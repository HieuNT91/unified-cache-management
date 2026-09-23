"""CPU-only final report with prompt/source integrity and completed-run checks."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

from common import REPO, settings, load, dump
from suite import locked, live_supervisor, check_orphan, verify, report


def main():
    cfg = settings()
    root = Path(cfg['root'])
    with locked(root):
        if live_supervisor(root):
            raise RuntimeError('Wait for the inference supervisor to exit')
        check_orphan(root)
        p = verify(cfg, root)
        provenance = load(root / 'provenance.json')
        for name, wanted in provenance['sha256'].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != wanted:
                raise ValueError(f'Frozen artifact changed: {name}')
        for name in (root / 'source').rglob('*.py'):
            current = Path(__file__).parent / name.relative_to(root / 'source')
            if current.read_bytes() != name.read_bytes():
                raise ValueError(f'Executed source differs from snapshot: {current}')
        memory_audits = list((root / 'smoke').rglob('*.memory-audit.rank*.json')) + list((root / 'sessions').glob('*.memory-audit.rank*.json'))
        if not memory_audits:
            raise ValueError('Missing activation-memory numerical audits')
        for path in memory_audits:
            audit = load(path)
            if len(audit['layers']) != 64 or not all(x['passed'] and x['output_projection']['passed'] for x in audit['layers']):
                raise ValueError(f'Invalid memory adaptation audit: {path}')
        cleanup = load(root / 'cleanup.json')
        if not cleanup['owned_engine_groups_exited'] or not cleanup['cache_removed']:
            raise ValueError('Incomplete engine/cache cleanup')
        report(root, p)
        certificate = load(root / 'final/validation.json')
        assert certificate['complete'] and certificate['validated'] == 60
        records = [json.loads(line) for line in (root / 'final/raw_records.jsonl').read_text().splitlines()]
        rows = []
        lines = ['Qwen3-32B — ProphetKV UCM/vLLM port', '',
                 'Completed: 60/60 measured requests; 10 VT + 10 CWE prompts.',
                 'GPUs 1–4, TP=4, BF16, greedy, thinking off, max 128 new tokens.',
                 '64K-target data, 4096-token chunks, YaRN factor 4.', '',
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
                  f'Length-limited outputs: {sum(r["finish_reason"] == "length" for r in records)}/60.',
                  'Accuracy is the official RULER reference-substring score, averaged over prompts.',
                  'TTFT runs from engine submission to first token-bearing output; it includes online probing,',
                  'cache transfers, selection and fusion. Model loading, offline cache building and priming',
                  'are excluded. Cache-building costs and total generation times are saved separately.',
                  '5%/20% refers to selected cached tokens after the exact first chunk and before the fresh suffix.',
                  'All methods share prompts, generation budgets and four physical GPUs. One timing per sample.',
                  'Memory adaptation on every method: 4096-token MLP and attention-output-projection tiles,',
                  'early release of dead attention tensors, expandable allocator segments, and 1029 KV blocks.',
                  'Tiling can change BF16 rounding and execution cost; these are memory-adapted local results.',
                  'This is a 10-sample/task study, not the full default RULER evaluation.',
                  'The 4B summary uses 100 samples/task and different model/RoPE/TP settings; its aggregate',
                  'accuracy and TTFT are not a paired comparison with this experiment.', '',
                  'Model configuration: https://huggingface.co/Qwen/Qwen3-32B',
                  f'Artifacts: {root / "final"}']
        text = '\n'.join(lines) + '\n'
        (root / 'final/RESULTS.md').write_text(text)
        (REPO / 'qwen3_32b_results.txt').write_text(text)
        with (root / 'final/summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        dump(root / 'final/integrity-validation.json', dict(complete=True,
            frozen_artifacts_verified=len(provenance['sha256']), measured_requests=60,
            cleanup=cleanup, summary=rows))
        print(text)


if __name__ == '__main__':
    main()
