"""Native two-request routing, shared by temporary and persistent clean readers."""
import json
import time
from pathlib import Path
from runner.router_policy import decide, features_from_arrays, digest
from runner.setups import atomic_json, file_hash


def validate_profile(sample, tp):
    from runner.config import validate_sample
    validate_sample(sample,4096,65536)
    if tp != 4 or sample['thinking'] or len(sample['token_ids'])+sample['max_output_tokens']>65920:
        raise ValueError('Router requires BF16 TP4 and non-thinking')


def sync_mode(llm, rid, action, tp, capture=False):
    from runner.worker import router_mode, arm
    replies = llm.collective_rpc(router_mode,kwargs=dict(request_id=rid,action=action,capture=capture))
    if sorted(r['rank'] for r in replies) != list(range(tp)) or any(
            r != dict(rank=r['rank'],request_id=rid,action=action,capture=capture) for r in replies):
        raise RuntimeError('TP router decision synchronization failed')
    return replies


def arm_request(llm, sample, rid, action, tp, capture=False):
    from runner.worker import arm
    sync = sync_mode(llm,rid,action,tp,capture)
    if action != 'baseline':
        llm.collective_rpc(arm,kwargs=dict(request_id=rid,boundaries=sample['boundaries'],
                                         question_positions=sample['question_positions']))
    return sync


def verify_retired(llm, tp, rid, scheduler_path):
    from runner.setups import retirement
    receipts = retirement(llm,tp)
    if scheduler_path is not None:
        state = json.loads(Path(scheduler_path).read_text())
        if state != dict(request_id=rid,requests_blend_meta=0,requests_meta=0):
            raise RuntimeError('Router scheduler request did not retire')
    return receipts


def extract_features(artifacts, sample, tp):
    import numpy as np
    if sorted(a['rank'] for a in artifacts) != list(range(tp)):
        raise RuntimeError('Missing probe rank')
    layers, scores = [], None
    for artifact in sorted(artifacts,key=lambda a:a['rank']):
        if file_hash(artifact['path']) != artifact['sha256']:
            raise ValueError('Corrupt probe export')
        with np.load(artifact['path'],allow_pickle=False) as array:
            current = array['scores']
            local = array['layers']
            if local.dtype != np.float32 or local.shape != (64,sample['boundaries'][-2]):
                raise ValueError('Expected native FP32 all64 layer capture')
            replay=np.zeros(local.shape[1],dtype=np.float32)
            for layer in local:
                np.add(replay,layer,out=replay)
            replay/=np.float32(64)
            if not np.array_equal(replay,array['local_mean']):
                raise ValueError('Ascending native FP32 all64 replay mismatch')
            if scores is not None and not np.array_equal(scores,current):
                raise RuntimeError('TP native attention scores differ')
            scores = current.copy()
            layers.append(array['layers'].astype(np.float64))
    return features_from_arrays(np.mean(layers,axis=0),scores,sample['boundaries'][1],sample['boundaries'][-2])


def measure(llm, sample, request_id, policy, router_id, output, tp=4, unchanged=lambda:None, scheduler_path=None):
    from run import generate, verify_diagnostics
    from runner.worker import router_export, drain
    validate_profile(sample,tp)
    from runner.layout import PROMPT_PROTOCOL
    if policy.get("prompt_protocol") != PROMPT_PROTOCOL or policy.get("evaluation_protocol") != sample.get("evaluation_protocol"):
        raise ValueError("Policy is incompatible with this prompt/evaluation protocol; no automatic refit")
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    probe_id = request_id+'-probe'
    arm_request(llm,sample,probe_id,'prophetkv-all64-1',tp,capture=True)
    internal, probe_ttft, _ = generate(llm.llm_engine,sample['token_ids'],1,probe_id,False,sample=sample)
    if len(internal.outputs[0].token_ids) != 1:
        raise RuntimeError('Router probe must produce exactly one discarded internal token')
    del internal
    export_started = time.perf_counter()
    artifacts = llm.collective_rpc(router_export,kwargs=dict(output=str(output/'probe')))
    diagnostics = llm.collective_rpc(drain)
    verify_diagnostics(diagnostics,sample,'prophetkv',.01,tp,range(64))
    atomic_json(output/'probe-diagnostics.json',diagnostics)
    exported = time.perf_counter()-export_started
    retired_at = time.perf_counter()
    retired = verify_retired(llm,tp,probe_id,scheduler_path)
    unchanged()
    retirement_seconds = time.perf_counter()-retired_at
    features_at = time.perf_counter()
    features = extract_features(artifacts,sample,tp)
    feature_seconds = time.perf_counter()-features_at
    decision_at = time.perf_counter()
    decision = decide(policy,router_id,features)
    decision_seconds = time.perf_counter()-decision_at
    answer_id = request_id+('|router-dense' if decision['action'] == 'baseline' else '')
    sync_at = time.perf_counter()
    replies = arm_request(llm,sample,answer_id,decision['action'],tp)
    sync_seconds = time.perf_counter()-sync_at
    overhead = time.perf_counter()-started
    result, answer_ttft, elapsed = generate(llm.llm_engine,sample['token_ids'],sample['max_output_tokens'],answer_id,False,sample=sample)
    # The elapsed interval before generate includes all routing work, including CPU export and barriers.
    return result, overhead+answer_ttft, overhead+elapsed, dict(router_id=router_id,
        primary=router_id=='router1',policy_sha256=policy['payload_sha256'],features=features,
        decision=decision,probe_request_id=probe_id,answer_request_id=answer_id,internal_tokens=1,
        probe_artifacts=artifacts,probe_diagnostics_sha256=file_hash(output/'probe-diagnostics.json'),
        probe_retirement=retired,decision_sync=replies,probe_ttft_seconds=probe_ttft,
        export_seconds=exported,retirement_seconds=retirement_seconds,feature_seconds=feature_seconds,
        decision_seconds=decision_seconds,tp_sync_seconds=sync_seconds,
        routing_overhead_seconds=overhead,answer_engine_ttft_seconds=answer_ttft)


def validate_answer(llm, diagnostics, result, sample, routing, tp):
    from run import verify_diagnostics
    from runner.worker import router_dense_receipt
    decision = routing['decision']
    dense = decision['action'] == 'baseline'
    verify_diagnostics(diagnostics,sample,'baseline' if dense else 'prophetkv',decision['ratio'],tp,range(64))
    if dense:
        if result.num_cached_tokens != 0:
            raise RuntimeError('Dense fallback reused KV')
        receipts = llm.collective_rpc(router_dense_receipt)
        if sorted(r['rank'] for r in receipts) != list(range(tp)):
            raise RuntimeError('Missing dense audit rank')
        routing['dense_receipts'] = receipts
