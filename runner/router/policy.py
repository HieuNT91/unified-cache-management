"""Frozen, interpretable attention router. CPU only; no training code imports."""
import hashlib
import json
import math
from pathlib import Path

FEATURES = ('top1_mass', 'top20_mass', 'top50_mass', 'group_agreement', 'group_top20_jaccard')
ACTIONS = ('prophetkv-all64-1', 'prophetkv-all64-20', 'prophetkv-all64-50', 'baseline')
RATIOS = dict(zip(ACTIONS, (.01, .20, .50, None)))
DEFINITIONS = dict(version=1, layers=64, groups=[list(range(32)), list(range(32,64))],
    selection='ascending FP32 layer sum / 64; native TP sum / TP size',
    normalization='all context keys; exclude first chunk before feature normalization',
    masses='float64 eligible scores / sum; stable descending top floor(N*pct/100)',
    agreement='TP mean per-layer FP32 scores in float64; group mean; cosine and top20 Jaccard',
    ties='ascending original position', missing='baseline')
SCHEMA = 'clean-attention-router-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def number(value):
    return type(value) in (float, int) and math.isfinite(value)


def validate(data):
    if (data.get('schema') != SCHEMA or data.get('primary') != 'router1'
            or data.get('feature_definitions') != DEFINITIONS
            or data.get('feature_names') != list(FEATURES)
            or data.get('actions') != list(ACTIONS)):
        raise ValueError('Incompatible router policy or feature definitions')
    if data.get('payload_sha256') != digest({k:v for k,v in data.items() if k != 'payload_sha256'}):
        raise ValueError('Router payload checksum mismatch')
    provenance = data.get('provenance', {})
    for key in ('training_protocol_sha256','policy_lock_sha256','cohort_sha256'):
        value = provenance.get(key, '')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing router provenance')
    rows = data.get('policies', [])
    if len(rows) != 3 or [r.get('id') for r in rows] != ['router1','router2','router3']:
        raise ValueError('Exactly three ordered policies are required')
    def node(n, depth=0):
        if not isinstance(n, dict) or depth > 3:
            raise ValueError('Tree exceeds depth 3')
        if 'action' in n:
            if set(n) != {'action','training_samples'} or n['action'] not in ACTIONS or type(n['training_samples']) is not int or n['training_samples'] < 1:
                raise ValueError('Invalid tree leaf')
            return 1
        if (set(n) != {'feature','threshold','le','gt'} or n['feature'] not in FEATURES
                or not number(n['threshold'])):
            raise ValueError('Invalid split')
        return node(n['le'],depth+1)+node(n['gt'],depth+1)
    for row in rows:
        source_hash=row.get('source_policy_sha256','')
        if len(source_hash)!=64 or any(c not in '0123456789abcdef' for c in source_hash):
            raise ValueError('Missing source policy provenance')
        costs = row.get('training_costs', [])
        if len(costs) != 4 or any(not number(x) or x <= 0 for x in costs):
            raise ValueError('Invalid frozen costs')
        if node(row['tree']) > 8:
            raise ValueError('Tree exceeds eight leaves')
    return data


def load_policy(path):
    path = Path(path)
    sidecar = Path(str(path)+'.sha256')
    if not sidecar.exists() or sidecar.read_text().split()[0] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Missing or mismatched policy SHA256 sidecar')
    return validate(json.loads(path.read_text()))


def decide(data, router_id, features):
    validate(data)  # corruption must never silently become dense fallback
    if router_id not in ('router1','router2','router3'):
        raise ValueError('Unknown router ID')
    missing = [name for name in FEATURES if not number(features.get(name))]
    if missing:
        return dict(action='baseline', ratio=None, fallback_reason='missing_required_feature', missing_features=missing)
    tree = data['policies'][int(router_id[-1])-1]['tree']
    while 'feature' in tree:
        tree = tree['le'] if features[tree['feature']] <= tree['threshold'] else tree['gt']
    action = tree['action']
    return dict(action=action, ratio=RATIOS[action], fallback_reason='dense_selected_by_policy' if action == 'baseline' else None)


def features_from_arrays(layers, scores, prefix, context_end):
    import numpy as np
    layers, scores = np.asarray(layers, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    if (layers.shape != (64,context_end) or scores.shape != (context_end-prefix,)
            or not np.isfinite(layers).all() or not np.isfinite(scores).all()
            or (layers < 0).any() or (scores < 0).any()):
        raise ValueError('Corrupt attention arrays')
    if scores.sum() <= 0:
        return dict.fromkeys(FEATURES)
    n = len(scores)
    ranked = np.argsort(-scores, kind='stable')
    probabilities = scores / scores.sum()
    result = {f'top{p}_mass':float(probabilities[ranked[:math.floor(n*p/100)]].sum()) for p in (1,20,50)}
    a, b = layers[:32,prefix:].mean(0), layers[32:,prefix:].mean(0)
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    result['group_agreement'] = float(np.dot(a,b)/denominator) if denominator > 0 else None
    count = math.floor(n*.2)
    left, right = np.argsort(-a,kind='stable')[:count], np.argsort(-b,kind='stable')[:count]
    result['group_top20_jaccard'] = len(np.intersect1d(left,right))/len(np.union1d(left,right)) if count else None
    return result


def rules(data):
    validate(data)
    lines = ['Primary: router1. Missing required feature -> baseline.', 'Training and clean inference runtimes differ. No test refitting.']
    def visit(tree, indent):
        if 'action' in tree:
            lines.append(indent+'return '+tree['action'])
        else:
            lines.append(indent+f"if {tree['feature']} <= {tree['threshold']!r}:")
            visit(tree['le'],indent+'    ')
            lines.append(indent+'else:')
            visit(tree['gt'],indent+'    ')
    for row in data['policies']:
        lines.append(row['id']+':')
        visit(row['tree'],'    ')
    return '\n'.join(lines)+'\n'
