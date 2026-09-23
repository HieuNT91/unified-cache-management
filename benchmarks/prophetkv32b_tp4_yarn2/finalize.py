"""CPU-only final report with prompt/source integrity and completed-run checks."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

from common import REPO, settings, load, dump
from suite import locked, live_supervisor, check_orphan, verify, report
from experiment import verify_frozen, EXTENSION, sha256


def main():
    cfg = settings()
    root = Path(cfg['root'])
    with locked(root):
        if live_supervisor(root):
            raise RuntimeError('Wait for the inference supervisor to exit')
        check_orphan(root)
        p = verify(cfg, root)
        verify_frozen(root)
        provenance = load(root / 'provenance.json')
        memory_audits = list((root / 'smoke').rglob('*.memory-audit.rank*.json')) + list((root / 'sessions').glob('*.memory-audit.rank*.json'))
        if not memory_audits:
            raise ValueError('Missing activation-memory numerical audits')
        for path in memory_audits:
            audit = load(path)
            if len(audit['layers']) != 64 or not all(x['passed'] and x['output_projection']['passed'] for x in audit['layers']):
                raise ValueError(f'Invalid memory adaptation audit: {path}')
            windows = audit.get('rope_windows', [])
            if len(windows) != 64 or not all(w['factor'] == 2 and w['table_positions'] == 65792
                    and w['original_entries_bitwise_equal'] and w['extended_formula_bitwise_equal'] for w in windows):
                raise ValueError(f'Invalid factor-2 position table audit: {path}')
        cleanup = load(root / 'cleanup.json')
        if not cleanup['owned_engine_groups_exited'] or not cleanup['cache_removed']:
            raise ValueError('Incomplete engine/cache cleanup')
        report(root, p)
        certificate = load(root / 'final/validation.json')
        assert certificate['complete'] and certificate['validated'] == 120
        records = [json.loads(line) for line in (root / 'final/raw_records.jsonl').read_text().splitlines()]
        rows = []
        lines = ['Qwen3-32B — ProphetKV UCM/vLLM port', '',
                 'Completed: 120/120 measured requests; 10 VT + 10 CWE prompts.',
                 'GPUs 1–4, TP=4, BF16, greedy, thinking off, max 128 new tokens.',
                 '64K-target data, 4096-token chunks, YaRN factor 2.', '',
                 '| Task | Method | Samples | Mean TTFT (s) | Accuracy (%) | Speedup |',
                 '|---|---|---:|---:|---:|---:|']
        for task in ('vt', 'cwe', 'combined'):
            group = [r for r in records if task == 'combined' or r['label'] == task]
            base = statistics.mean(r['ttft_seconds'] for r in group if r['case'] == 'baseline')
            for method in p['cases']:
                subset = [r for r in group if r['case'] == method]
                ttft = statistics.mean(r['ttft_seconds'] for r in subset)
                accuracy = 100 * statistics.mean(r['score'] for r in subset)
                rows.append(dict(task=task, method=method, samples=len(subset),
                    mean_ttft_seconds=ttft, accuracy_percent=accuracy, speedup=base/ttft,
                    length_limited=sum(r['finish_reason'] == 'length' for r in subset),
                    mean_cache_build_seconds=statistics.mean(r['cache_build_seconds'] for r in subset)))
                lines.append(f'| {task} | {method} | {len(subset)} | {ttft:.3f} | {accuracy:.1f} | {base/ttft:.2f}x |')
        lengths = [s['tokens'] for s in p['samples']]
        lines += ['', f'Actual prompt lengths: {min(lengths):,}–{max(lengths):,} tokens.',
                  f'Length-limited outputs: {sum(r["finish_reason"] == "length" for r in records)}/120.',
                  'Accuracy is the official RULER reference-substring score, averaged over prompts.',
                  'TTFT runs from engine submission to first token-bearing output; it includes online probing,',
                  'cache transfers, selection and fusion. Model loading, offline cache building and priming',
                  'are excluded. Cache-building costs and total generation times are saved separately.',
                  'Ratios refer to selected cached tokens after the exact first chunk and before the fresh suffix.',
                  'All methods share prompts, generation budgets and four physical GPUs. One timing per sample.',
                  'Memory adaptation on every method: 4096-token MLP and attention-output-projection tiles,',
                  'early release of dead attention tensors, expandable allocator segments, and 1029 KV blocks.',
                  'Tiling can change BF16 rounding and execution cost; these are memory-adapted local results.',
                  'Nominal factor-2 window is 65,536. The identical 65,792-token engine window uses',
                  '256 appended RoPE table positions with unchanged factor-2 frequencies and magnitude;',
                  'the first 65,536 table entries are bitwise preserved. This slightly exceeds nominal 2x.',
                  'This is a 10-sample/task study, not the full default RULER evaluation.',
                  'The 4B summary uses 100 samples/task and different model/RoPE/TP settings; its aggregate',
                  'accuracy and TTFT are not a paired comparison with this experiment.', '',
                  'Model configuration: https://huggingface.co/Qwen/Qwen3-32B',
                  f'Artifacts: {root / "final"}']
        text = '\n'.join(lines) + '\n'
        (root / 'final/RESULTS.md').write_text(text)
        (REPO / 'qwen3_32b_yarn2_results.txt').write_text(text)
        with (root / 'final/summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        dump(root / 'final/integrity-validation.json', dict(complete=True,
            frozen_artifacts_verified=len(provenance['sha256']), measured_requests=120,
            cleanup=cleanup, summary=rows))
        write_rope_comparison(root, p, records)
        print(text)


def write_rope_comparison(root, protocol, current):
    previous = [json.loads(line) for line in (EXTENSION / 'combined/raw_records.jsonl').read_text().splitlines()]
    old = {(r['sample_id'], r['case']): r for r in previous}
    new = {(r['sample_id'], r['case']): r for r in current}
    if len(old) != 120 or len(new) != 120 or set(old) != set(new):
        raise ValueError('RoPE comparisons are not paired')
    for key, record in new.items():
        for field in ('prompt_tokens', 'max_output_tokens', 'references', 'source_row', 'gpu_devices', 'tensor_parallel_size'):
            if record[field] != old[key][field]:
                raise ValueError(f'Paired protocol mismatch: {key} {field}')
    lines = ['Qwen3-32B: paired YaRN 4x versus 2x', '',
             '120 preserved 4x measurements versus 120 fresh 2x measurements on identical prompts.',
             'Ten prompts/task; native non-thinking chat, BF16 TP=4 on GPUs 1–4, 128-token cap, 4096-token chunks.', '',
             '| Task | Method | 4x TTFT (s) | 4x accuracy (%) | 2x TTFT (s) | 2x accuracy (%) |',
             '|---|---|---:|---:|---:|---:|']
    rows = []
    for task in ('vt', 'cwe', 'combined'):
        for case in protocol['cases']:
            keys = [key for key, r in new.items() if r['case'] == case and (task == 'combined' or r['label'] == task)]
            mean = lambda records, field: statistics.mean(records[key][field] for key in keys)
            ttft4, ttft2 = mean(old, 'ttft_seconds'), mean(new, 'ttft_seconds')
            acc4, acc2 = 100 * mean(old, 'score'), 100 * mean(new, 'score')
            rows.append(dict(task=task, method=case, paired_samples=len(keys),
                yarn4_ttft_seconds=ttft4, yarn2_ttft_seconds=ttft2,
                yarn4_accuracy_percent=acc4, yarn2_accuracy_percent=acc2,
                accuracy_change_pp=acc2-acc4, ttft_ratio_2x_over_4x=ttft2/ttft4))
            lines.append(f'| {task} | {case} | {ttft4:.3f} | {acc4:.1f} | {ttft2:.3f} | {acc2:.1f} |')
    lines += ['', 'TTFT excludes model load, offline cache building, readiness waits and priming.',
              'Accuracy is official RULER reference-substring credit; one timing per prompt/method.',
              'The 2x cohort ran later. All methods share the original memory tiling and engine allocation.',
              'The 2x table appends 256 positions beyond its nominal 65536 using unchanged frequencies',
              'to preserve exact prompts and the 65792-token engine window. Original entries are bitwise unchanged.',
              'Both factors passed their own native-prefix bitwise controls and all-layer tensor audits.',
              'Tiling may affect BF16 rounding and execution cost. This is ten samples/task, not full RULER.']
    directory = root / 'yarn-comparison'
    directory.mkdir(exist_ok=True)
    text = '\n'.join(lines) + '\n'
    (directory / 'REPORT.md').write_text(text)
    (REPO / 'qwen3_32b_yarn_comparison.txt').write_text(text)
    with (directory / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump(directory / 'validation.json', dict(complete=True, paired_measurements=120,
        yarn4_preserved=True, yarn2_validated=True,
        sha256={f.name: sha256(f) for f in directory.iterdir() if f.is_file() and f.name != 'validation.json'}))


if __name__ == '__main__':
    main()
