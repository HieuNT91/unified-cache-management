#!/usr/bin/env python3
"""One-command status for `run.sh sweep` roots: progress, accuracy, TTFT.

Accuracy is shown two ways: this repo's `ruler_all` (reference substring
coverage) and sparse_attention's RULER metric (needle_score for NIAH,
multi_words for CWE) recomputed from the saved prediction, so the numbers can
be read next to sparse_attention tables.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
import string


def normalize_answer(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def postprocess_pred(s):
    s = s.strip().replace('<|eot_id|>', '').replace('</s>', '').replace('</s', '').replace('</', '')
    return re.sub(r'[\x00-\x1f]', '\n', s).strip()


def needle_score(prediction, truth):
    prediction, truth = normalize_answer(postprocess_pred(prediction)), normalize_answer(truth)
    score = float(prediction[:len(truth)] == truth)
    return max(float(truth in prediction.split()), score)


def multi_words(prediction, truth):
    words = re.findall(r'\b\w+\b', prediction.lower())
    return sum(t.lower() in words for t in truth) / len(truth)


def sparse_attention_score(task, prediction, references):
    return multi_words(prediction, references) if task == 'cwe' else needle_score(prediction, references[0])


def order(name):
    return (0, 0) if name == 'baseline' else (1, int(name.rsplit('-', 1)[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    args = parser.parse_args()
    cells = defaultdict(list)
    for root in args.roots:
        manifest = [json.loads(l) for l in (root / 'manifest.jsonl').read_text().splitlines() if l.strip()] \
            if (root / 'manifest.jsonl').exists() else None
        shards = sorted(root.glob('group-*'))
        print(f'== {root}')
        for group in shards:
            plan = json.loads((group / 'plan.json').read_text()) if (group / 'plan.json').exists() else {}
            state = 'complete' if (group / 'complete.json').exists() else \
                'FAILED: ' + json.loads((group / 'failure.json').read_text())['error'][:200] \
                if (group / 'failure.json').exists() else 'running'
            print(f"  {group.name}: {len(plan.get('prompt_ids', []))} prompts, {state}")
        for result in root.glob('*/[0-9]*/result.json'):
            r = json.loads(result.read_text())
            name = result.parent.parent.name
            answer = r.get('answer_text', r['prediction'])
            cells[(r['subtask'], name)].append(dict(
                ruler_all=r['accuracy'], sa=sparse_attention_score(r['subtask'], answer, r['references']),
                ttft=r['timings']['ttft_seconds'], gpus=tuple(r['gpu_uuids'])))
        if manifest:
            print(f'  manifest prompts: {len(manifest)}')
    tasks = sorted({t for t, _ in cells})
    names = sorted({n for _, n in cells}, key=order)
    print(f"\n{'task':<17}{'config':<15}{'n':>5}{'sa_metric':>11}{'ruler_all':>11}{'ttft_mean':>11}{'ttft_med':>10}")
    for task in tasks:
        for name in names:
            rows = cells.get((task, name), [])
            if not rows:
                continue
            ttft = [r['ttft'] for r in rows]
            print(f"{task:<17}{name:<15}{len(rows):>5}{statistics.mean(r['sa'] for r in rows):>11.3f}"
                  f"{statistics.mean(r['ruler_all'] for r in rows):>11.3f}{statistics.mean(ttft):>10.2f}s"
                  f"{statistics.median(ttft):>9.2f}s")
        print()


if __name__ == '__main__':
    main()
