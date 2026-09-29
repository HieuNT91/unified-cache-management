"""Versioned task-independent trees, separate from the frozen router_policy v1."""
import json
import re
from pathlib import Path
from runner.router_policy import FEATURES, DEFINITIONS, digest, number
from runner.setups import atomic_json, file_hash
SCHEMA = 'ruler-shared-tree-v1'
FEATURE_DEFINITIONS = dict(DEFINITIONS, version='attention-five-v1', missing='nocache')
DEFAULT_ACTIONS = [dict(id='nocache', method='baseline', ratio=None)] + [
    dict(id=f'prophetkv-{p}', method='prophetkv', ratio=p/100) for p in (1,20,40)]


def actions(value=None):
    value = DEFAULT_ACTIONS if value is None else value
    if not isinstance(value,list) or len(value)<2:raise ValueError('Inventory requires nocache and sparse actions')
    seen=set();ratios=set()
    for item in value:
        if (not isinstance(item,dict) or set(item)!={'id','method','ratio'} or
                not isinstance(item['id'],str) or item['id'] in ('probe','router','control') or not re.fullmatch('[a-z][a-z0-9-]*',item['id']) or item['id'] in seen):
            raise ValueError('Invalid or duplicate action definition')
        seen.add(item['id'])
        if item['id']=='nocache':
            if item!=dict(id='nocache',method='baseline',ratio=None):raise ValueError('Invalid nocache')
        elif (item['method']!='prophetkv' or not number(item['ratio']) or
              not 0<item['ratio']<=1 or item['ratio'] in ratios):raise ValueError('Unsupported or duplicate sparse action')
        else:ratios.add(item['ratio'])
    if 'nocache' not in seen:raise ValueError('Inventory must contain nocache')
    return {item['id']:dict(item) for item in value}


def validate(data):
    if (data.get('schema')!=SCHEMA or data.get('feature_names')!=list(FEATURES) or
            data.get('feature_definitions')!=FEATURE_DEFINITIONS):raise ValueError('Unsupported tree/feature version')
    inventory=actions(data.get('actions',[]))
    if data.get('payload_sha256')!=digest({k:v for k,v in data.items() if k!='payload_sha256'}):
        raise ValueError('Tree checksum mismatch')
    provenance=data.get('provenance',{})
    for field in ('snapshot_sha256','training_run_sha256','corpus_protocol_sha256'):
        if not re.fullmatch('[a-f0-9]{64}',provenance.get(field,'')):raise ValueError('Missing training provenance')
    train=data.get('training_ids',[]);test=data.get('heldout_ids',[])
    if not train or len(set(train))!=len(train) or len(set(test))!=len(test) or set(train)&set(test):
        raise ValueError('Invalid frozen train/test membership')
    def visit(node,depth=0):
        if not isinstance(node,dict) or depth>3:raise ValueError('Invalid tree depth')
        if 'action' in node:
            if (set(node)!={'action','training_samples'} or node['action'] not in inventory or
                    type(node['training_samples']) is not int or node['training_samples']<1):raise ValueError('Invalid tree leaf')
        elif (set(node)!={'feature','threshold','le','gt'} or node['feature'] not in FEATURES or
              not number(node['threshold'])):raise ValueError('Invalid tree split')
        else:visit(node['le'],depth+1);visit(node['gt'],depth+1)
    visit(data['tree'])
    costs=data.get('training_costs',[])
    if len(costs)!=len(inventory) or any(not number(v) or v<=0 for v in costs):raise ValueError('Invalid frozen action costs')
    return data


def load(path):
    path=Path(path);side=Path(str(path)+'.sha256')
    if not side.exists() or side.read_text().split()[0]!=file_hash(path):raise ValueError('Missing or corrupt tree checksum sidecar')
    return validate(json.loads(path.read_text()))


def decide(data,features):
    validate(data)
    missing=[f for f in FEATURES if not number(features.get(f))];node=data['tree']
    if missing:action='nocache'
    else:
        while 'feature' in node:node=node['le'] if features[node['feature']]<=node['threshold'] else node['gt']
        action=node['action']
    return dict(action=action,ratio=actions(data['actions'])[action]['ratio'],fallback_reason=
                'missing_required_feature' if missing else 'dense_selected_by_policy' if action=='nocache' else None)


def rules(data):
    validate(data);lines=['Missing required feature -> nocache. Costs frozen on training hardware.']
    def visit(n,indent=''):
        if 'action' in n:lines.append(indent+'return '+n['action'])
        else:
            lines.append(indent+f"if {n['feature']} <= {n['threshold']!r}:")
            visit(n['le'],indent+'    ');lines.append(indent+'else:');visit(n['gt'],indent+'    ')
    visit(data['tree']);return '\n'.join(lines)+'\n'


def export(path,data):
    path=Path(path);data=dict(data);data['payload_sha256']=digest({k:v for k,v in data.items() if k!='payload_sha256'})
    validate(data)
    if path.exists():raise FileExistsError('Tree exports are immutable')
    atomic_json(path,data);Path(str(path)+'.sha256').write_text(file_hash(path)+'  '+path.name+'\n')
    path.with_suffix('.txt').write_text(rules(data));return data
