"""Atomic measurement publications; accepted artifacts are immutable."""
import json
import math
import re
from pathlib import Path
from runner.corpus import relative
from runner.setups import atomic_json,file_hash,fingerprint
from runner.tree_policy import actions

NATIVE_ANSWER_VALIDATION = 'native-answer-diagnostics-v1'
PROBE_ANSWER_VALIDATION = 'independent-probe-replay-v1'


def answer_validation(protocol):
    mode=protocol.get('answer_validation',PROBE_ANSWER_VALIDATION)
    if mode not in (NATIVE_ANSWER_VALIDATION,PROBE_ANSWER_VALIDATION):
        raise ValueError('Unknown answer validation protocol')
    return mode


def record_dir(root,case,prompt_id):
    if any(not isinstance(s,str) or not re.fullmatch('[a-zA-Z0-9_-]+',s) for s in (case,prompt_id)):
        raise ValueError('Invalid record identity')
    return Path(root)/'records'/case/prompt_id


def protocol_identity(protocol):
    # Deployment paths may move without changing accepted data identity.
    return fingerprint({k:v for k,v in protocol.items() if k not in ('prepared','model','cache_root')})


def accepted(root,case,row,protocol,prepared=None,full=False):
    from runner.tensor_parallel import protocol_tp
    tp=protocol_tp(protocol)
    mode=answer_validation(protocol)
    if mode==NATIVE_ANSWER_VALIDATION and case in ('probe','router'):
        raise ValueError('Native answer controls do not use independent probes or routers')
    folder=record_dir(root,case,row['id']);path=folder/'validated.json'
    if not path.exists():return None
    saved=json.loads(path.read_text())
    if saved.get('complete') is not True or saved['protocol_sha256']!=protocol_identity(protocol):
        raise ValueError('Accepted record protocol changed')
    if not {'result.json','diagnostics.json'}<=set(saved['files']):raise ValueError('Incomplete receipt')
    for name,expected in saved['files'].items():
        artifact=relative(folder,name)
        if not artifact.is_file() or ((full or name=='result.json') and file_hash(artifact)!=expected):raise ValueError('Corrupt accepted artifact: '+name)
    result=json.loads((folder/'result.json').read_text())
    if result.get('answer_validation',PROBE_ANSWER_VALIDATION)!=mode:
        raise ValueError('Accepted answer validation mode changed')
    group=row['ordinal']%len(protocol['groups'])
    if (result['prompt_id']!=row['id'] or result['method']!=case or result['input_sha256']!=row['sha256'] or
            result['group']!=group or result['gpu_uuids']!=protocol['groups'][group] or not result['cache_immutable']):
        raise ValueError('Accepted identity mismatch')
    retired=result['retirement']
    if sorted(r['rank'] for r in retired)!=list(range(tp)) or any(not r['quiescent'] or r['transfers']['pending'] or r['request_bookkeeping'] for r in retired):
        raise ValueError('Missing retirement evidence')
    if not relative(root,result['initialization']).is_file() or file_hash(relative(root,result['initialization']))!=result['initialization_sha256']:
        raise ValueError('Initialization evidence changed')
    initial=json.loads(relative(root,result['initialization']).read_text())
    if not initial.get('validated'):raise ValueError('Initialization was not validated')
    if case=='probe':
        if result['internal_tokens']!=1 or len(result.get('artifacts',[]))!=tp:raise ValueError('Incomplete independent probe')
        if any(saved['files'].get(a['path'])!=a['sha256'] for a in result['artifacts']):raise ValueError('Unpinned attention')
    else:
        if case!='router' and result['executed_action']!=case:raise ValueError('Accepted action differs from inventory')
        if not math.isfinite(result['accuracy']) or not 0<=result['accuracy']<=1:raise ValueError('Invalid score')
        if case=='nocache' and (result['num_cached_tokens']!=0 or initial['engine_config'].get('kv_transfer_config')):
            raise ValueError('Baseline is not connector-free')
    if any(not isinstance(t,(int,float)) or not math.isfinite(t) or t<0
           for k,t in result['timings'].items()
           if not (k=='first_answer_content_seconds' and t is None and result.get('generated_answer_tokens')==0)):
        raise ValueError('Invalid timing')
    if protocol.get('execution_profile')=='ruler-thinking' and case!='probe':
        from runner.thinking_budget import KEY, PROTOCOL, validate_thinking_outcome
        if result.get(KEY)!=row[KEY] or result.get('evaluation_protocol')!=PROTOCOL:
            raise ValueError('Thinking result policy/provenance changed')
        validate_thinking_outcome(dict(result,ttft_seconds=result['timings']['ttft_seconds'],
            first_answer_content_seconds=result['timings'].get('first_answer_content_seconds')),row[KEY])
    if full and case=='router':validate_routed(root,folder,row,protocol,result)
    if full and case!='router':
        from runner.generation import verify_diagnostics
        from runner.corpus import load_attention,match_answer
        sample=json.loads(relative(prepared or protocol['prepared'],row['prepared']).read_text())
        ds=json.loads((folder/'diagnostics.json').read_text())
        definition=dict(method='prophetkv',ratio=.01) if case=='probe' else actions(protocol['actions'])[case]
        verify_diagnostics(ds,sample,definition['method'],definition['ratio'],tp,range(64))
        if case=='probe':
            heads=protocol.get('feature_profile',{}).get('head_layers')
            load_attention(folder,result,sample,protocol['actions'],heads,tp=tp)
        elif case!='nocache' and mode==PROBE_ANSWER_VALIDATION:
            probe=accepted(root,'probe',row,protocol,prepared)
            if probe is None:raise ValueError('Missing independent probe')
            attention=load_attention(record_dir(root,'probe',row['id']),probe,sample,protocol['actions'],tp=tp)
            match_answer(ds,sample,case,attention,protocol['actions'],tp=tp)
    return result


def publish(root,case,row,protocol,result,diagnostics):
    import fcntl
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    folder=record_dir(root,case,row['id'])
    # Serialize duplicate checking, body publication and the atomic commit marker.
    with (root/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'validated.json').exists():raise FileExistsError('Accepted records cannot be replaced')
        atomic_json(folder/'result.json',result);atomic_json(folder/'diagnostics.json',diagnostics)
        files={str(p.relative_to(folder)):file_hash(p) for p in folder.rglob('*') if p.is_file() and not p.name.endswith('.tmp')}
        atomic_json(folder/'validated.json',dict(complete=True,protocol_sha256=protocol_identity(protocol),files=files))
    return result


def validate_routed(root,folder,row,protocol,result):
    """Revalidate stored live controls; changing any evidence halts resume."""
    from runner.tree_policy import load,decide
    from runner.corpus_runtime import paired
    tree=load(Path(root)/'tree.json')
    if result['tree_sha256']!=tree['payload_sha256'] or decide(tree,result['probe']['features'])!=result['decision']:
        raise ValueError('Accepted tree decision changed')
    control=json.loads((folder/'control.json').read_text())
    if result['executed_action']!=result['decision']['action'] or control['executed_action']!=result['decision']['action']:
        raise ValueError('Stored control action differs from tree')
    control_ds=json.loads((folder/'control-diagnostics.json').read_text())
    ds=json.loads((folder/'diagnostics.json').read_text())
    if paired(result,ds,control,control_ds)!=result['paired_validation']:raise ValueError('Invalid stored control comparison')
    if result['decision']['action']=='nocache':
        baseline=accepted(root,'nocache',row,protocol)
        if baseline is None or baseline!=control:raise ValueError('Dense control is not the connector-free baseline')
        checks=result['dense_receipts'];names=sorted(f'model.layers.{i}.self_attn.attn' for i in range(64))
        if sorted(c['rank'] for c in checks)!=list(range(protocol.get('tp',4))) or any(c['native_layers']!=names or c['store_operations']!=dict(lookup=0,load=0,store=0) for c in checks):
            raise ValueError('Dense fallback native/zero-store audit failed')
