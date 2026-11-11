"""Atomic measurement publications; accepted artifacts are immutable."""
import json
import math
import re
from pathlib import Path
from runner.corpus.dataset import relative
from runner.cache.setups import atomic_json,file_hash,fingerprint

NATIVE_ANSWER_VALIDATION = 'native-answer-diagnostics-v1'


def record_dir(root,case,prompt_id,seed=None):
    if any(not isinstance(s,str) or not re.fullmatch('[a-zA-Z0-9_-]+',s) for s in (case,prompt_id)):
        raise ValueError('Invalid record identity')
    folder = Path(root)/'records'/case/prompt_id
    if seed is not None:
        if type(seed) is not int or seed < 0:
            raise ValueError('Invalid answer seed identity')
        folder = folder/f'seed-{seed}'
    return folder


def protocol_identity(protocol):
    # Deployment paths may move without changing accepted data identity.
    return fingerprint({k:v for k,v in protocol.items() if k not in ('prepared','model','cache_root')})


def accepted(root,case,row,protocol,prepared=None,full=False,*,seed=None):
    if case not in ('adatree-v1', 'nocache'):
        raise ValueError('Only AdaTree and no-cache records are supported')
    if protocol.get('answer_validation') != NATIVE_ANSWER_VALIDATION:
        raise ValueError('Expected native answer diagnostics')
    from runner.runtime.tensor_parallel import protocol_tp
    tp = protocol_tp(protocol)
    from scripts.collection.collection_v3 import enabled, seeds
    if seed not in seeds(protocol, case):
        raise ValueError('Answer seed differs from frozen schedule')
    folder=record_dir(root,case,row['id'],seed);path=folder/'validated.json'
    if not path.exists():return None
    saved=json.loads(path.read_text())
    if saved.get('complete') is not True or saved['protocol_sha256']!=protocol_identity(protocol):
        raise ValueError('Accepted record protocol changed')
    if not {'result.json','diagnostics.json'}<=set(saved['files']):raise ValueError('Incomplete receipt')
    for name,expected in saved['files'].items():
        artifact=relative(folder,name)
        if not artifact.is_file() or ((full or name=='result.json') and file_hash(artifact)!=expected):raise ValueError('Corrupt accepted artifact: '+name)
    result=json.loads((folder/'result.json').read_text())
    if enabled(protocol) and (type(result.get('seed')) is not int or result['seed'] != seed or result.get('sequence_id') != row['id']):
        raise ValueError('Accepted answer seed/sequence changed')
    if result.get('answer_validation') != NATIVE_ANSWER_VALIDATION:
        raise ValueError('Accepted answer validation mode changed')
    group=row['ordinal']%len(protocol['groups'])
    if (result['prompt_id']!=row['id'] or result['method']!=case or result['input_sha256']!=row['sha256'] or
            result['group']!=group or result['gpu_uuids']!=protocol['groups'][group] or not result['cache_immutable']):
        raise ValueError('Accepted identity mismatch')
    if enabled(protocol):
        from runner.inputs.layout import PROMPT_PROTOCOL
        if (result.get('prompt_protocol') != PROMPT_PROTOCOL
                or result.get('evaluation_protocol') != row.get('evaluation_protocol')
                or not re.fullmatch('[0-9a-f]{64}', result.get('token_sha256', ''))):
            raise ValueError('Accepted prompt/generation provenance mismatch')
        if (result.get('subtask') != row['subtask'] or type(result.get('output_cap_reached')) is not bool
                or any(type(result.get(k)) is not int or result[k] < 0 for k in
                       ('thinking_tokens', 'answer_tokens', 'control_tokens', 'output_tokens', 'forced_tokens'))
                or result['output_tokens'] != sum(result[k] for k in ('thinking_tokens', 'answer_tokens', 'control_tokens'))):
            raise ValueError('Invalid committed answer metrics')
        from runner.runtime.output_budget import OUTPUT_CAP
        from scripts.data.ruler import CAPS
        cap = CAPS.get(row['subtask']) if protocol['generation_protocol'] == 'ruler-nonthinking-v1' else OUTPUT_CAP
        if protocol.get('answer_generation') and (case == 'adatree-v1' or case == 'nocache' and protocol.get('followup_nocache')):
            cap = protocol['answer_generation']['output_reserve']
        if (result.get('max_output_tokens') != cap or result['output_tokens'] > cap
                or len(result.get('output_token_ids', [])) != result['output_tokens']):
            raise ValueError('Invalid committed answer output budget')
    retired=result['retirement']
    if sorted(r['rank'] for r in retired)!=list(range(tp)) or any(not r['quiescent'] or r['transfers']['pending'] or r['request_bookkeeping'] for r in retired):
        raise ValueError('Missing retirement evidence')
    if not relative(root,result['initialization']).is_file() or file_hash(relative(root,result['initialization']))!=result['initialization_sha256']:
        raise ValueError('Initialization evidence changed')
    initial=json.loads(relative(root,result['initialization']).read_text())
    if not initial.get('validated'):raise ValueError('Initialization was not validated')
    if case=='adatree-v1':
        from runner.router.adatree import validate_record
        validate_record(protocol, result)
    elif result['executed_action']!=case:raise ValueError('Accepted action differs from inventory')
    if not math.isfinite(result['accuracy']) or not 0<=result['accuracy']<=1:raise ValueError('Invalid score')
    if case=='nocache' and (result['num_cached_tokens']!=0 or initial['engine_config'].get('kv_transfer_config')):
        raise ValueError('Baseline is not connector-free')
    if any(not isinstance(t,(int,float)) or not math.isfinite(t) or t<0
           for k,t in result['timings'].items()
           if not (k=='first_answer_content_seconds' and t is None and result.get('generated_answer_tokens')==0)):
        raise ValueError('Invalid timing')
    if (protocol.get('execution_profile')=='ruler-thinking' or (enabled(protocol) and protocol['dataset']=='longbench-v2')):
        from runner.runtime.output_budget import KEY, validate_outcome
        from runner.inputs.prepared_policy import RULER_PROTOCOL as PROTOCOL
        expected_protocol = row.get('evaluation_protocol') if enabled(protocol) else protocol.get('evaluation_protocol', PROTOCOL)
        expected_policy = row[KEY]
        if protocol.get('answer_generation') and (case == 'adatree-v1' or case == 'nocache' and protocol.get('followup_nocache')):
            from runner.router.adatree import generation_policy
            expected_policy = generation_policy(row[KEY], protocol)
            if result.get('thinking_loop_breaker') != protocol['answer_generation']['loop_breaker']:
                raise ValueError('Thinking loop breaker differs from configured policy')
            if result.get('prepared_generation_policy') != row[KEY]:
                raise ValueError('Prepared generation provenance changed')
        if result.get(KEY)!=expected_policy or result.get('evaluation_protocol')!=expected_protocol:
            raise ValueError('Thinking result policy/provenance changed')
        validate_outcome(dict(result,ttft_seconds=result['timings']['ttft_seconds'],
            first_answer_content_seconds=result['timings'].get('first_answer_content_seconds')),expected_policy)
    if full:
        from runner.runtime.generation import verify_diagnostics
        sample=json.loads(relative(prepared or protocol['prepared'],row['prepared']).read_text())
        ds=json.loads((folder/'diagnostics.json').read_text())
        from runner.tree.collection import inventory
        definition=inventory(protocol)[case]
        if case == 'adatree-v1':
            from runner.router.adatree import validate_routing
            from runner.tree.collection import runtime_sample
            sample = runtime_sample(sample, protocol)
            if validate_routing(ds, sample, result['routing']['request_id'], tp,
                                chunk_size=protocol.get('kv_chunk_size') or 4096) != result['routing']:
                raise ValueError('Accepted AdaTreeV1 routing evidence changed')
        else:
            verify_diagnostics(ds,sample,definition['method'],definition['ratio'],tp)
    return result


def publish(root,case,row,protocol,result,diagnostics,*,seed=None):
    import fcntl
    if case not in ('adatree-v1', 'nocache') or protocol.get('answer_validation') != NATIVE_ANSWER_VALIDATION:
        raise ValueError('Only native AdaTree and no-cache answers can be published')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    from scripts.collection.collection_v3 import enabled, seeds
    if seed not in seeds(protocol, case):
        raise ValueError('Answer seed differs from frozen schedule')
    if enabled(protocol) and (result.get('seed') != seed or result.get('sequence_id') != row['id']):
        raise ValueError('Answer seed/sequence mismatch')
    folder=record_dir(root,case,row['id'],seed)
    # Serialize duplicate checking, body publication and the atomic commit marker.
    with (root/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'validated.json').exists():raise FileExistsError('Accepted records cannot be replaced')
        atomic_json(folder/'result.json',result);atomic_json(folder/'diagnostics.json',diagnostics)
        files={str(p.relative_to(folder)):file_hash(p) for p in folder.rglob('*') if p.is_file() and not p.name.endswith('.tmp')}
        atomic_json(folder/'validated.json',dict(complete=True,protocol_sha256=protocol_identity(protocol),files=files))
    return result
