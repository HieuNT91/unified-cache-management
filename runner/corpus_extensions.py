"""Discover append-only action batches and describe their combined read-only view."""
import json
from pathlib import Path
from runner.corpus_records import protocol_identity
from runner.tree_policy import actions


def discover(root, requested=None):
    root=Path(root)
    found=[]
    for path in sorted((root/'extensions').glob('add*/protocol.json')):
        protocol=json.loads(path.read_text())
        if protocol.get('schema')!='ruler-inplace-actions-v2':continue
        if requested is None or set(protocol['added_actions'])&set(requested):
            found.append((path.parent.relative_to(root),protocol))
    return found


def joined_protocol(original,extensions):
    if not extensions:return original
    if len(extensions)==1:return extensions[0][1]
    inventory={a['id']:a for a in original['actions']}
    sources={}
    for state,protocol in extensions:
        for name in protocol['added_actions']:
            if name in inventory:raise ValueError('Action registered in multiple sources: '+name)
            inventory[name]=next(a for a in protocol['actions'] if a['id']==name)
        sources[str(state)]=protocol_identity(protocol)
    ordered=sorted(inventory.values(),key=lambda a:a['ratio'] or 0)
    actions(ordered)
    return dict(original,schema='ruler-multi-extension-view-v1',actions=ordered,
                extension_protocols=sources,base_protocol_sha256=protocol_identity(original))
