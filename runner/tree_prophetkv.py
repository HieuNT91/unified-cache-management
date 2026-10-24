"""CPU-safe protocol, immutable F3 records and native tree evidence checks."""
import json
import math
from pathlib import Path

METHOD = 'tree-prophetkv'
PROTOCOL = 'tree-prophetkv-f3-local16-source-tree-s4-in-budget-v1'
SOURCE_COMMIT = 'f96cb874c7becddd868f8f47872701f6111489ac'


def budget(boundaries, ratio):
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('Invalid tree-prophetkv ratio')
    prefix, end = boundaries[1], boundaries[-2]
    forced = [p+j for p in boundaries[1:-2] for j in range(4)]
    count = math.floor((end-prefix)*ratio)
    if count < len(forced):
        raise ValueError('tree-prophetkv budget is too small for four tokens per reused chunk (S4)')
    return count, forced


def f3_path(cache, tokens):
    from runner.layout import token_hash
    return Path(cache) / 'tree-f3' / (token_hash(list(tokens)) + '.json')


def read_f3(cache, namespace, tokens):
    from runner.layout import token_hash
    from runner.setups import fingerprint
    path = f3_path(cache, tokens)
    if path.is_symlink() or path.parent.is_symlink():
        raise RuntimeError('Symlinked F3 record')
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError('Missing or corrupt F3; build a fresh setup with --tree-prophetkv') from error
    if not isinstance(record, dict) or not isinstance(record.get('payload'), dict):
        raise RuntimeError('Invalid F3 record structure')
    payload = record['payload']
    values = payload.get('values', [])
    if (record.get('sha256') != fingerprint(payload)
            or payload.get('protocol') != PROTOCOL or payload.get('namespace') != namespace
            or payload.get('token_sha256') != token_hash(list(tokens))
            or payload.get('layers') != list(range(16)) or payload.get('window') != 16
            or payload.get('source_commit') != SOURCE_COMMIT
            or not isinstance(values, list) or len(values) != len(tokens)
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1.0001 for v in values)):
        raise RuntimeError('F3 identity, checksum or values mismatch')
    return values


def write_f3(cache, namespace, tokens, values):
    """Called only by construction rank zero under the setup/namespace owner."""
    from runner.layout import token_hash
    from runner.setups import atomic_json, fingerprint
    path = f3_path(cache, tokens)
    if path.exists() or path.is_symlink():
        # A resumed construction must preserve already committed records.
        read_f3(cache, namespace, tokens)
        return
    if path.parent.is_symlink():
        raise RuntimeError('Symlinked F3 directory')
    values = [float(v) for v in values]
    if len(values) != len(tokens) or any(not math.isfinite(v) or not 0 <= v <= 1.0001 for v in values):
        raise RuntimeError('Invalid construction-time F3')
    payload = dict(protocol=PROTOCOL, source_commit=SOURCE_COMMIT, namespace=namespace,
                   token_sha256=token_hash(list(tokens)), layers=list(range(16)), window=16, values=values)
    atomic_json(path, dict(payload=payload, sha256=fingerprint(payload)))


def snapshot_f3(cache, namespace, samples):
    from runner.setups import file_hash
    result = {}
    for sample in samples:
        for a, z in zip(sample['boundaries'][:-2], sample['boundaries'][1:-1]):
            tokens = sample['token_ids'][a:z]
            path = f3_path(cache, tokens)
            name = path.name
            if name not in result:
                read_f3(cache, namespace, tokens)
                result[name] = file_hash(path)
    return result


def validate_selection(event, sample, ratio):
    """Replay the source tree from native F3 evidence, then apply the S4 budget."""
    import numpy as np
    from ucm.sparse.prophetkv.tree_selection import select_tree
    bounds = sample['boundaries']
    count, forced = budget(bounds, ratio)
    links = event.get('f3_links', [])
    if (event.get('selection_method') != METHOD or event.get('tree_protocol') != PROTOCOL
            or event.get('ratio') != ratio or event.get('unit_budget') != count
            or event.get('forced') != forced or event.get('selected_count') != count
            or event.get('eligible_count') != bounds[-2]-bounds[1]
            or event.get('question_positions') != sample['question_positions']
            or event.get('probe_question_positions') != list(range(min(sample['question_positions']), bounds[-1]))
            or event.get('f3_layers') != list(range(16))
            or event.get('recompute_mask') != 'prefix-fresh-earlier-own-chunk-causal'
            or event.get('suffix_mask') != 'prefix-fresh-causal-suffix'
            or event.get('decode_mask') != 'native-unmasked'
            or len(links) != bounds[-2]-bounds[1]
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1.0001 for v in links)):
        raise RuntimeError('Invalid tree-prophetkv evidence')
    reference, _ = select_tree(np.asarray(links), np.asarray(event['scores']), bounds, ratio)
    if event.get('selected_positions') != reference:
        raise RuntimeError('Tree selection replay mismatch')
    return reference
