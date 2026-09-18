"""Per-request semantic validation; receipts do not require engine shutdown."""
import json,math,re,threading,time
from pathlib import Path
from prophetkv_gpu import ROOT,GPU_UUIDS
from prophetkv_common import sha,dump,load_sample,score
from cacheblend_prophetkv import CASES,TIMING
FATAL=re.compile(r'Traceback|\[UC\]\[E\]|load kv cache failed|dump kv cache failed|CUDA out of memory|EngineDeadError|ERROR\s|\b(?:RuntimeError|AssertionError|ValueError):')
PROGRESS=threading.Lock()
def validate(path,meta,case,smoke=False):
    r=json.loads(path.read_text());log=path.with_suffix('.log').read_text()
    if FATAL.search(log) or 'REQUEST_COMPLETE' not in log:raise ValueError('Unclean worker log')
    expected=dict(sample_id=meta['id'],case=case,input_sha256=meta['input_sha256'],prompt_sha256=meta['prompt_sha256'],
        protocol_sha256=sha(ROOT/'protocol.json'),physical_gpu=meta['physical_gpu'],gpu_uuid=GPU_UUIDS[meta['physical_gpu']],
        prompt_tokens=meta['tokens'],smoke=smoke,max_output_tokens=16 if smoke else 128,timing_source=TIMING,
        cache_unchanged=True,online_mask_reused=False)
    if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('Record identity/protocol mismatch')
    if any(not math.isfinite(r[k]) or r[k]<=0 for k in ('ttft_seconds','generation_seconds')) or r['ttft_seconds']>r['generation_seconds']:
        raise ValueError('Invalid timing')
    if not 0<r['output_tokens']<=r['max_output_tokens'] or r['output_tokens']!=len(r['output_token_ids']) or r['finish_reason'] not in ('length','stop'):
        raise ValueError('Invalid generation')
    sample=load_sample(meta['input_path'])
    if r['references']!=sample['source_metadata']['references'] or score(sample,r['prediction'])[0]!=r['score']:raise ValueError('Scoring mismatch')
    if any(w['visible_uuid']!=r['gpu_uuid'] or not Path(w['ucm_path']).is_relative_to(ROOT/'private_ucm') for w in r['worker_imports']):raise ValueError('Worker provenance mismatch')
    dp=path.with_suffix('.diagnostics.json')
    if sha(dp)!=r['diagnostics_sha256']:raise ValueError('Diagnostics changed')
    diag=[d for w in json.loads(dp.read_text()) for d in w['diagnostics']]
    if r['namespace'] != r['request_id'].split(':')[0] or r['max_model_len'] != json.loads((ROOT/'protocol.json').read_text())['max_model_len']:
        raise ValueError('Namespace/capacity mismatch')
    if r['cache_files_after_retirement'] != 0 or not all(w['quiescent'] and w['transfers']['pending']==0 and w['request_bookkeeping']==0 for w in r['retirement']['workers']):
        raise ValueError('Incomplete retirement')
    if case=='baseline':
        if diag or r['num_cached_tokens']!=0:raise ValueError('Baseline cache reuse')
        return r
    rid=r['request_id']
    measured=log.split('MEASURE_BEGIN',1)[1].split('MEASURE_END',1)[0]
    hits=re.findall(rf'request_id: {re.escape(rid)},.*?req_stage: BlendStage\.(\w+), first chunk prefix hit: (\d+), chunks cache total hit: (\d+)',measured)
    b=meta['boundaries'];eligible=b[-2]-b[1];ratio=int(case.split('-')[-1])/100;count=int(eligible*ratio)
    if hits!=[('CACHE_BLEND',str(b[1]//64),str(eligible//64))]:raise ValueError('Incomplete measured cache hit evidence')
    if r['cache_verification']['namespace'] != r['namespace']:raise ValueError('Wrong cache namespace')
    if not r['cache_verification']['complete']:raise ValueError('Cache verification missing')
    layers=[d for d in diag if d.get('kind') in ('layer_counts','fusion_audit')]
    selection=[d for d in diag if d not in layers]
    if len(selection)!=1 or len(layers)!=36:raise ValueError('Missing request or layer diagnostics')
    d=selection[0]
    if d['request_id']!=rid or d['eligible_count']!=eligible or d['selected_count']!=count:raise ValueError('Invalid selection budget')
    selected=d['selected_positions'];scores=d['scores'];chosen=set(selected)
    if len(chosen)!=count or not chosen<=set(range(b[1],b[-2])) or len(scores)!=eligible or any(not math.isfinite(x) or x<0 for x in scores):raise ValueError('Invalid selection')
    if 0<count<eligible and min(scores[i-b[1]] for i in chosen)<max(s for i,s in enumerate(scores) if i+b[1] not in chosen):raise ValueError('Selection not top-ranked')
    if case.startswith('prophetkv'):
        if d.get('suffix_tokens')!=b[-1]-b[-2] or d.get('probe_tokens_per_layer')!=b[-1]-b[-2]:raise ValueError('Probe suffix mismatch')
        if d['kind']!='prophetkv_selection' or d['question_positions']!=meta['query']['positions'] or d['probe_layers']!=36 or d['alignment_count']!=36:raise ValueError('Probe mismatch')
        expected_selected=sorted(sorted(range(eligible),key=lambda i:(-scores[i],i))[:count])
        if selected!=[i+b[1] for i in expected_selected]:raise ValueError('Unstable tie handling')
    for i,event in enumerate(layers):
        if event['layer']!=f'model.layers.{i}.self_attn.attn':raise ValueError('Layer coverage mismatch')
        suffix=b[-1]-b[-2]
        projection=eligible+suffix if case.startswith('cacheblend') and i<=1 else count+suffix
        attention=eligible+suffix if case.startswith('cacheblend') and i==0 else count+suffix
        if event['projection_tokens']!=projection or event['attention_tokens']!=attention or event['ffn_tokens']!=attention:
            raise ValueError('Actual compute count mismatch')
        if smoke and not all(event.get(k) is True for k in ('k_write_verified','v_write_verified','skipped_preserved','prefix_preserved','suffix_verified','causal_attention_verified')):
            raise ValueError('Numerical cache/model audit failed')
    return r

def valid(path,meta,case,smoke=False):
    try:
        receipt=json.loads(path.with_suffix('.validated.json').read_text())
        if any(sha(path.with_suffix(suffix))!=receipt[key] for suffix,key in [('.json','record_sha256'),('.log','log_sha256'),('.diagnostics.json','diagnostics_sha256')]):return False
        validate(path,meta,case,smoke);return True
    except (OSError,ValueError,KeyError,TypeError):return False

def all_records(p):
    rows=[]
    for meta in p['samples']:
        for case in CASES:
            path=ROOT/'records'/meta['id']/f'{case}.json'
            if valid(path,meta,case):rows.append(json.loads(path.read_text()))
    return rows

def progress(p,state):
    with PROGRESS:
        # Counts use validation receipts; full semantic validation occurs on acceptance/resume/report.
        n=sum(1 for m in p['samples'] for c in CASES if (ROOT/'records'/m['id']/f'{c}.validated.json').exists())
        dump(ROOT/'progress.json',dict(state=state,validated=n,target=p['measured_requests'],updated_at=time.time()))
        return n

