"""Study-only variable-feature policy; old five-feature schemas stay unchanged."""
import math
from runner.gpu_study_features import REGISTERED
from runner.setups import fingerprint
SCHEMA='single-probe-study-policy-v1'


def validate(policy):
    if policy.get('schema')!=SCHEMA or policy.get('sha256')!=fingerprint({k:v for k,v in policy.items() if k!='sha256'}):raise ValueError('Study policy checksum/schema mismatch')
    names=policy['feature_names']
    if len(set(names))!=len(names) or not set(names)<=set(REGISTERED):raise ValueError('Unregistered features')
    used=set()
    def walk(n,d=0):
        if d>3:raise ValueError('Study tree exceeds depth3')
        if 'action' in n:
            if n['action'] not in policy['actions']:raise ValueError('Unknown action')
        else:
            if n['feature'] not in names or not math.isfinite(n['threshold']):raise ValueError('Invalid split')
            used.add(n['feature']);walk(n['le'],d+1);walk(n['gt'],d+1)
    walk(policy['tree'])
    if set(names)!=used:raise ValueError('Policy must compute only used features')
    return policy


def make(tree,actions,provenance):
    used=set()
    def walk(n):
        if 'feature' in n:used.add(n['feature']);walk(n['le']);walk(n['gt'])
    walk(tree)
    p=dict(schema=SCHEMA,tree=tree,actions=actions,feature_names=[f for f in REGISTERED if f in used],provenance=provenance)
    p['sha256']=fingerprint(p)
    return validate(p)


def decide(policy,features):
    validate(policy)
    if any(type(features.get(f)) not in (int,float) or not math.isfinite(features[f]) for f in policy['feature_names']):return 'nocache'
    n=policy['tree']
    while 'feature' in n:n=n['le'] if features[n['feature']]<=n['threshold'] else n['gt']
    return n['action']


class Handoff:
    """Immutable CPU score bytes, bound once to a unique final request."""
    def __init__(self,binding,scores,source_event):
        import numpy as np
        required={'input_sha256','cache_identity','runtime_sha256','gpu_uuids','source_request'}
        if set(binding)!=required or not all(binding.values()):raise ValueError('Incomplete score binding')
        a=np.asarray(scores)
        if a.dtype!=np.float32 or a.ndim!=1 or not np.isfinite(a).all() or (a<0).any():raise ValueError('Corrupt native scores')
        self.binding=__import__('copy').deepcopy(binding);self.bytes=a.tobytes();self.event=dict(source_event);self.target=None;self.used=False
    def arm(self,binding,target):
        if self.used or self.target is not None or binding!=self.binding or target==binding['source_request']:raise ValueError('Stale/mismatched handoff')
        if target.split(':',1)[0]!=binding['source_request'].split(':',1)[0]:raise ValueError('Cache namespace changed')
        self.target=target
    def consume(self,binding,target):
        import numpy as np
        if self.used or binding!=self.binding or target!=self.target or target is None:raise ValueError('Handoff isolation failure')
        self.used=True
        return np.frombuffer(self.bytes,dtype=np.float32).copy(),dict(self.event)


def check_boundaries(policy,rows):
    """Independent recursive evaluation, including ancestor-compatible witnesses."""
    import numpy as np
    checks=0
    def reference(node,row):
        if 'action' in node:return node['action']
        return reference(node['le'] if row[node['feature']]<=node['threshold'] else node['gt'],row)
    for row in rows:
        if decide(policy,row)!=reference(policy['tree'],row):raise ValueError('Compiled policy mismatch')
        checks+=1
    def visit(node,subset):
        nonlocal checks
        if 'action' in node:return
        if not subset:raise ValueError('Unreachable tree split')
        f=node['feature'];t=node['threshold']
        for v in (float(np.nextafter(t,-np.inf)),t,float(np.nextafter(t,np.inf))):
            row=dict(subset[0],**{f:v})
            if decide(policy,row)!=reference(policy['tree'],row):raise ValueError('Threshold boundary mismatch')
            checks+=1
        visit(node['le'],[r for r in subset if r[f]<=t]);visit(node['gt'],[r for r in subset if r[f]>t])
    visit(policy['tree'],rows)
    for f in policy['feature_names']:
        for value in (None,True,float('inf'),-float('inf'),float('nan')):
            if decide(policy,dict(rows[0],**{f:value}))!='nocache':raise ValueError('Missing-feature fallback mismatch')
            checks+=1
        row=dict(rows[0]);del row[f]
        if decide(policy,row)!='nocache':raise ValueError('Absent feature fallback mismatch')
        checks+=1
    return checks
