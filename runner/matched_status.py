"""CPU-only snapshots of the exact prompts shared by all active methods."""
import json
from pathlib import Path

from runner.setups import file_hash


def match_records(by_method, expected):
    """Intersect identities, not counts: shards can advance in different orders."""
    tasks = {r['prompt_id']: r['subtask'] for r in expected}
    if len(tasks) != len(expected):
        raise ValueError('Duplicate expected prompt IDs')
    indexed = {}
    for method, records in by_method.items():
        indexed[method] = {}
        for record in records:
            pid = record['prompt_id']
            if pid in indexed[method] or pid not in tasks or record['subtask'] != tasks[pid]:
                raise ValueError('Duplicate/unexpected result or subtask mismatch')
            indexed[method][pid] = record
    common = set.intersection(*(set(rows) for rows in indexed.values())) if indexed else set()
    ids = [r['prompt_id'] for r in expected if r['prompt_id'] in common]
    metadata = dict(mode='same_count',matching='prompt ID intersection across all active methods',
        matched_samples=len(ids),prompt_ids=ids,
        per_subtask={t:sum(tasks[p] == t for p in ids) for t in sorted(set(tasks.values()))},
        available_counts={m:len(rs) for m,rs in indexed.items()})
    return {m:[rs[p] for p in ids] for m,rs in indexed.items()}, metadata


def committed_records(root, cases, protocol_sha256):
    """Read committed results and verify their pins; no model or diagnostic replay.

    Freeze receipt paths first so newly published answers wait for the next status.
    Workers publish validated.json last and never modify accepted results.
    """
    paths = {c:sorted((Path(root)/'records'/c).glob('*/validated.json')) for c in cases}
    result = {c:[] for c in cases}
    for case, receipts in paths.items():
        for path in receipts:
            receipt = json.loads(path.read_text())
            body = path.parent/'result.json'
            if (receipt.get('complete') is not True or receipt.get('protocol_sha256') != protocol_sha256
                    or receipt.get('files', {}).get('result.json') != file_hash(body)):
                raise ValueError('Accepted result pin or protocol mismatch')
            record = json.loads(body.read_text())
            if record['prompt_id'] != path.parent.name or record['method'] != case:
                raise ValueError('Accepted result identity mismatch')
            result[case].append(record)
    return result
