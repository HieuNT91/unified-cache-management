"""Export all committed tree and copied no-cache outcomes without probes."""
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from scripts.experiment_task_breakdown import snapshot


def export(root, output=None, quiet=False):
    root = Path(root).resolve()
    output = Path(output).absolute() if output else root/'tree-prophetkv-outcomes.npz'
    if output.suffix != '.npz':
        raise ValueError('Output must end in .npz')
    if output.exists() or output.is_symlink():
        raise ValueError(f'Output exists; choose another --output: {output}')
    # This reader verifies original receipts, input/policy/token identity and
    # selects only the tree experiment's seeds from a copied baseline run.
    expected, data = snapshot(root)
    tasks = {r['prompt_id']: r['subtask'] for r in expected}
    if len(tasks) != len(expected):
        raise ValueError('Duplicate expected prompt IDs')
    seen, present = set(), set()
    fields = {k: [] for k in ('prompt_id', 'method', 'accuracy', 'ttft')}
    for method, records in data.items():
        for record in records:
            identity = record['prompt_id']
            key = (method, identity)
            if (key in seen or identity not in tasks
                    or record['subtask'] != tasks[identity]):
                raise ValueError('Duplicate/unexpected result or subtask mismatch')
            seen.add(key)
            present.add(identity)
            pid, seed = json.loads(record['prompt_id'])
            pid = f'{pid}--seed-{seed}' if seed is not None else pid
            accuracy, ttft = record['accuracy'], record['timings']['ttft_seconds']
            if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (accuracy, ttft))
                    or not 0 <= accuracy <= 1 or ttft <= 0):
                raise ValueError(f'Invalid accuracy/TTFT: {pid}, {method}')
            for key, value in zip(fields, (pid, method, accuracy, ttft)):
                fields[key].append(value)
    arrays = {k: np.asarray(v, dtype=np.float64 if k in ('accuracy', 'ttft') else str)
              for k, v in fields.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.tree-outcomes-', dir=output.parent) as tmp:
        staged = Path(tmp)/'outcomes.npz'
        np.savez_compressed(staged, **arrays)
        # Atomically publish without replacing any concurrent export.
        os.link(staged, output)
    if not quiet:
        baseline = 'included from baseline-source/' if 'nocache' in data else 'absent (tree methods only)'
        print(f'Saved {output}: {len(fields["prompt_id"])} outcomes, '
              f'{len(present)}/{len(expected)} prompt/seed pairs with at least one answer; '
              f'nocache {baseline}.\n'
              f'Available per method: { {method: len(rows) for method, rows in data.items()} }\n'
              'Fields: prompt_id, method, accuracy, ttft (seconds).', flush=True)
    return output
