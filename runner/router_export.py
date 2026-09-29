"""One-shot CPU export of finalized training data; never launches a consumer."""
import argparse
import json
import math
from pathlib import Path
from runner.router_policy import ACTIONS, FEATURES, DEFINITIONS, SCHEMA, canonical, digest, decide, rules, validate
from runner.setups import atomic_json, file_hash


def source_action(policy, values):
    node = policy['tree']
    while 'feature' in node:
        node = node['left'] if values[FEATURES[node['feature']]] <= node['threshold'] else node['right']
    if policy['family'] == 'cost':
        index = node['action']
    else:
        eligible = [i for i in range(3) if node['loss'][i] <= policy['hyperparameters']['threshold']] + [3]
        index = min(eligible, key=lambda i:(policy['costs'][i],i))
    return ACTIONS[index]


def compile_policy(policy):
    if (policy['version'] != 1 or policy['feature_names'] != list(FEATURES)
            or policy['actions'] != list(ACTIONS) or policy['family'] not in ('loss','cost')):
        raise ValueError('Incompatible source policy')
    def walk(node, depth=0):
        if depth > 3 or node['n'] < 1 or len(node['loss']) != 3 or any(
                not isinstance(x,(int,float)) or not math.isfinite(x) or not 0 <= x <= 1+1e-12 for x in node['loss']):
            raise ValueError('Invalid source tree')
        if 'feature' in node:
            index = node['feature']
            if type(index) is not int or not 0 <= index < 5:
                raise ValueError('Invalid source feature')
            return dict(feature=FEATURES[index],threshold=node['threshold'],
                        le=walk(node['left'],depth+1),gt=walk(node['right'],depth+1))
        leaf_policy = dict(policy,tree=node)
        return dict(action=source_action(leaf_policy,{}),training_samples=node['n'])
    return dict(id=f"router{policy['rank']}", tree=walk(policy['tree']), training_costs=policy['costs'],
                source_policy_sha256=digest(policy),family=policy['family'],
                oof={k:v for k,v in policy.items() if k.startswith('oof_')})


def export(root, output, cohort):
    from runner.router_process import alive, group_alive
    root, output, cohort = Path(root), Path(output), Path(cohort)
    read = lambda name:json.loads((root/name).read_text())
    final, lock, cleanup = read('final-validation.json'), read('policies/lock.json'), read('cleanup.json')
    if (not final['complete'] or final['validated_answers'] != 780 or final['tuning_probes'] != 195
            or not cleanup['complete'] or not lock['complete'] or lock['training_answers'] != 780
            or lock['training_probes'] != 195 or lock['primary'] != 'router-1'):
        raise ValueError('Training has not completed and locked all policies')
    supervisor = read('supervisor.json')
    if supervisor['state'] != 'complete' or alive(supervisor):
        raise ValueError('Training supervisor must exit successfully')
    for path in root.glob('active-group*.json'):
        state = json.loads(path.read_text())
        if alive(state) or group_alive(state['pid']):
            raise ValueError('Owned training engine is still alive')
    for receipt in (lock,final):
        for name, expected in receipt['sha256'].items():
            if file_hash(name) != expected:
                raise ValueError('Finalized training artifact changed: '+name)
    if len(lock['training_evidence'])!=975:
        raise ValueError('Expected 780 answer and 195 probe acceptance receipts')
    for name, expected in lock['training_evidence'].items():
        if file_hash(name) != expected:
            raise ValueError('Training evidence changed')
        evidence=json.loads(Path(name).read_text())
        if not evidence['complete']:
            raise ValueError('Incomplete training evidence')
        for artifact,checksum in evidence['sha256'].items():
            if file_hash(artifact)!=checksum:
                raise ValueError('Training evidence artifact changed: '+artifact)
    spec = json.loads(cohort.read_text())
    if file_hash(root/'protocol.json') != spec['training_protocol_sha256']:
        raise ValueError('Wrong training protocol')
    policies = [read(f'policies/{i}.json') for i in (1,2,3)]
    if any(p['rank'] != i or p['primary'] != (i == 1) for i,p in enumerate(policies,1)):
        raise ValueError('Policy rank changed')
    result = dict(schema=SCHEMA,primary='router1',feature_names=list(FEATURES),feature_definitions=DEFINITIONS,
        actions=list(ACTIONS),policies=[compile_policy(p) for p in policies],
        provenance=dict(training_protocol_sha256=file_hash(root/'protocol.json'),policy_lock_sha256=file_hash(root/'policies/lock.json'),cohort_sha256=file_hash(cohort)))
    result['payload_sha256'] = digest(result)
    validate(result)
    matrices=read('policies/training-matrices.json')
    features=matrices['features']
    if len(set(matrices['ids']))!=195 or len(matrices['ids'])!=195:
        raise ValueError('Invalid training matrix IDs')
    for pid,values in zip(matrices['ids'],features):
        if read(f'records/{pid}/tuning-probe.json')['routing']['features']!=values:
            raise ValueError('Saved feature matrix differs from its source probe')
    if len(features) != 195:
        raise ValueError('Expected 195 frozen training feature rows')
    checked = 0
    for p in policies:
        probes = list(features)
        def boundaries(n, values):
            if 'feature' not in n:
                return
            name, threshold = FEATURES[n['feature']],n['threshold']
            for v in (math.nextafter(threshold,-math.inf),threshold,math.nextafter(threshold,math.inf)):
                probes.append(dict(values,**{name:v}))
            boundaries(n['left'],dict(values,**{name:threshold}))
            boundaries(n['right'],dict(values,**{name:math.nextafter(threshold,math.inf)}))
        boundaries(p['tree'],dict.fromkeys(FEATURES,.5))
        for row in probes:
            actual = decide(result,f"router{p['rank']}",row)['action']
            missing = any(row.get(k) is None or not math.isfinite(row[k]) for k in FEATURES)
            expected = 'baseline' if missing else source_action(p,row)
            if actual != expected:
                raise ValueError('Export decision differs from frozen source policy')
            checked += 1
    blob = canonical(result)+b'\n'
    if output.exists() and output.read_bytes() != blob:
        raise FileExistsError('Refusing to replace a different policy')
    output.parent.mkdir(parents=True,exist_ok=True)
    temp = output.with_suffix('.tmp');temp.write_bytes(blob);temp.replace(output)
    Path(str(output)+'.sha256').write_text(file_hash(output)+'  '+output.name+'\n')
    output.with_suffix('.txt').write_text(rules(result))
    atomic_json(output.with_suffix('.verification.json'),dict(decisions_checked=checked,training_rows=195,policies=3,sha256=file_hash(output)))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cohort',type=Path,default=Path(__file__).resolve().parents[1]/'scripts/router_cohort.json')
    a=p.parse_args();export(a.training_root,a.output,a.cohort)
