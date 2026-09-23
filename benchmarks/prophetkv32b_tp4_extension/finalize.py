"""Validate added ratios and report six methods without rerunning old cases."""
import csv
import json
from pathlib import Path
import statistics

from common import REPO, settings, load, dump
from extension import PARENT, ALL_CASES, verify_frozen, sha256
from suite import locked, live_supervisor, check_orphan, verify, report


def main():
    root = Path(settings()['root'])
    with locked(root):
        if live_supervisor(root):
            raise RuntimeError('Wait for the inference supervisor to exit')
        check_orphan(root)
        check_orphan(PARENT)
        verify_frozen(root)
        p = verify(settings(), root)
        audits = list((root / 'sessions').glob('*.memory-audit.rank*.json'))
        if not audits:
            raise ValueError('Missing memory adaptation audits')
        for path in audits:
            a = load(path)
            if len(a['layers']) != 64 or not all(x['passed'] and x['output_projection']['passed'] for x in a['layers']):
                raise ValueError(f'Invalid numerical audit: {path}')
        cleanup = load(root / 'cleanup.json')
        if not cleanup['owned_engine_groups_exited'] or not cleanup['cache_removed']:
            raise ValueError('Incomplete engine/cache cleanup')
        report(root, p)
        if not load(root / 'final/validation.json')['complete']:
            raise ValueError('Added cohort is incomplete')
        read_records = lambda path: [json.loads(s) for s in path.read_text().splitlines()]
        original = read_records(PARENT / 'final/raw_records.jsonl')
        added = read_records(root / 'final/raw_records.jsonl')
        if len(original) != 60 or len(added) != 60:
            raise ValueError('Expected 60 original and 60 added records')
        records = original + added
        expected = {(m['id'], case) for m in p['samples'] for case in ALL_CASES}
        if len(records) != len(expected) or {(r['sample_id'], r['case']) for r in records} != expected:
            raise ValueError('Missing or duplicate combined measurement')
        for sample in p['samples']:
            if Path(sample['input_path']).read_bytes() != (PARENT / 'samples' / (sample['id'] + '.json')).read_bytes():
                raise ValueError('Combined prompt mismatch')
        rows = []
        lines = ['Qwen3-32B — ProphetKV UCM/vLLM port', '',
            'Completed: 120/120 measurements; 10 VT + 10 CWE prompts per method.',
            'Original no-cache, 5% and 20% measurements retained unchanged; 30%, 40%, 50% added.',
            'GPUs 1–4, TP=4, BF16, native non-thinking chat, greedy, max 128 new tokens.',
            '64K-target data, 4096-token chunks, YaRN factor 4.', '',
            '| Task | Method | Samples | Mean TTFT (s) | Accuracy (%) | Speedup |',
            '|---|---|---:|---:|---:|---:|']
        for task in ('vt', 'cwe', 'combined'):
            group = [r for r in records if task == 'combined' or r['label'] == task]
            base = statistics.mean(r['ttft_seconds'] for r in group if r['case'] == 'baseline')
            for case in ALL_CASES:
                subset = [r for r in group if r['case'] == case]
                mean = lambda key: statistics.mean(r[key] for r in subset)
                ttft, accuracy = mean('ttft_seconds'), 100 * mean('score')
                rows.append(dict(task=task, method=case, samples=len(subset),
                    mean_ttft_seconds=ttft, accuracy_percent=accuracy, speedup=base/ttft,
                    mean_generation_seconds=mean('generation_seconds'),
                    length_limited=sum(r['finish_reason'] == 'length' for r in subset),
                    mean_cache_build_seconds=mean('cache_build_seconds'),
                    cohort='retained' if case in ('baseline', 'prophetkv-5', 'prophetkv-20') else 'extension'))
                lines.append(f'| {task} | {case} | {len(subset)} | {ttft:.3f} | {accuracy:.1f} | {base/ttft:.2f}x |')
        lengths = [m['tokens'] for m in p['samples']]
        lines += ['', f'Actual prompt lengths: {min(lengths):,}–{max(lengths):,} tokens.',
            f'Length-limited outputs: {sum(r["finish_reason"] == "length" for r in records)}/120.',
            'Accuracy is official RULER reference-substring credit averaged across prompts.',
            'TTFT is engine submission to first token-bearing output, including online probing,',
            'cache transfers, selection and fusion; model loading, cache building and priming are excluded.',
            'Cache-building costs, total generation times and per-method truncation counts are in summary.csv.',
            'Ratios select cached tokens after the exact first chunk and before the fresh suffix.',
            'All methods use identical prompts and budgets; one timing per prompt and method.',
            'Added ratios were measured later than the preserved original cohort; method order is fixed.',
            'Every method uses 4096-token MLP/output-projection tiling, early release of dead attention tensors,',
            'expandable allocator segments, and 1029 KV blocks. Tiling can affect BF16 rounding and timing.',
            'The extension removes only the unused startup checkpoint export; inference arithmetic is unchanged.',
            'Native-prefix bitwise controls are inherited from the original qualified implementation.',
            'Each new ratio/task passed A-B-A versus fresh-engine isolation and all-layer tensor/causality audits.',
            'This is a 10-sample/task study, not the full default RULER evaluation.',
            'The 4B summary has different sample counts/model/RoPE/TP settings and is not a paired comparison.', '',
            f'Original artifacts: {PARENT / "final"}', f'Combined artifacts: {root / "combined"}']
        combined = root / 'combined'
        combined.mkdir(exist_ok=True)
        text = '\n'.join(lines) + '\n'
        (combined / 'REPORT.md').write_text(text)
        (combined / 'raw_records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        with (combined / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        dump(combined / 'validation.json', dict(complete=True, retained=60, added=60, total=120,
            parent_integrity_certificate=str(PARENT / 'final/integrity-validation.json'),
            new_integrity_certificate=str(root / 'final/integrity-validation.json'),
            sha256={f.name: sha256(f) for f in combined.iterdir() if f.is_file() and f.name != 'validation.json'}))
        dump(root / 'final/integrity-validation.json', dict(complete=True, added_requests=60,
            retained_requests=60, combined_requests=120, frozen_artifacts_verified=len(load(root / 'provenance.json')['sha256']),
            parent_artifacts_verified=len(load(root / 'extension.json')['parent_sha256']), cleanup=cleanup))
        temporary = REPO / 'qwen3_32b_results.txt.tmp'
        temporary.write_text(text)
        temporary.replace(REPO / 'qwen3_32b_results.txt')
        print(text)


if __name__ == '__main__':
    main()
