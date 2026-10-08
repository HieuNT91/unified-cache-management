"""One native clean engine/configuration on one TP4 group; resumable accepted rows."""
from runner.layout import sample_provenance
from runner.reporting import scoring_text
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

from runner.config import engine_config
from runner.router_policy import RATIOS, load_policy
from runner.router_process import identity
from runner.setups import atomic_json, file_hash, fingerprint, check_environment, start_engine, retirement
from runner.router_measure import measure, validate_answer, arm_request, validate_profile, verify_retired
from runner.sweep import PromptCache

CASES = ('baseline','prophetkv-all64-1','prophetkv-all64-20','prophetkv-all64-50','router1','router2','router3')


def record_dir(root, case, prompt_id):
    if case not in CASES or '/' in prompt_id or '..' in prompt_id:
        raise ValueError('Invalid record identity')
    return Path(root)/'records'/case/prompt_id


def selection_signature(diagnostics):
    return [dict(rank=w['rank'],selections=[{k:e[k] for k in ('scores','selected_positions','scoring_layers')}
        for e in w['diagnostics'] if e['kind']=='prophetkv_selection']) for w in sorted(diagnostics,key=lambda w:w['rank'])]


def paired_check(record, diagnostics, fixed, fixed_diagnostics):
    if (record['prompt_id'] != fixed['prompt_id'] or record['group'] != fixed['group']
            or record['gpu_uuids'] != fixed['gpu_uuids'] or record['input_sha256'] != fixed['input_sha256']
            or record['output_token_ids'] != fixed['output_token_ids']):
        raise ValueError('Router output differs from its same-group fixed action')
    action=record['routing']['decision']['action']
    if action != fixed['method']:
        raise ValueError('Paired fixed action differs from tree decision')
    if selection_signature(diagnostics) != selection_signature(fixed_diagnostics):
        raise ValueError('Router scores/masks differ from fixed action')
    return dict(action=action,output_tokens_equal=True,scores_masks_equal=True,group=record['group'])


def accepted(root, case, row, protocol, full=False):
    folder=record_dir(root,case,row['id']);receipt=folder/'validated.json'
    if not receipt.exists():
        return None
    saved=json.loads(receipt.read_text())
    if not saved.get('complete') or saved['protocol_sha256'] != fingerprint(protocol):
        raise ValueError('Accepted record protocol changed')
    for name,expected in saved['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid accepted artifact path')
        if not (folder/relative).is_file() or ((full or name=='result.json') and file_hash(folder/relative) != expected):
            raise ValueError('Accepted artifact changed: '+str(folder/relative))
    record=json.loads((folder/'result.json').read_text())
    if (record['prompt_id'] != row['id'] or record['method'] != case or record['group'] != row['ordinal']%2
            or record['gpu_uuids'] != protocol['groups'][record['group']]
            or record['input_sha256'] != row['sha256'] or not record['cache_immutable']
            or len(record['retirement']) != 4 or not all(r['quiescent'] and r['transfers']['pending']==0
                and r['request_bookkeeping']==0 for r in record['retirement'])):
        raise ValueError('Accepted record identity or retirement mismatch')
    if full:
        initialization=Path(root)/record['session']/'initialization.json'
        if file_hash(initialization)!=record['initialization_sha256']:
            raise ValueError('Scheduled initialization/warmup evidence changed')
        from runner.generation import verify_diagnostics
        diagnostics=json.loads((folder/'diagnostics.json').read_text())
        sample=json.loads((Path(protocol['prepared'])/row['prepared']).read_text())
        routing=record.get('routing')
        action=routing['decision']['action'] if routing else case
        verify_diagnostics(diagnostics,sample,'baseline' if action=='baseline' else 'prophetkv',RATIOS[action],4,range(64))
        if action == 'baseline' and record['num_cached_tokens'] != 0:
            raise ValueError('Accepted dense request reused tokens')
        if routing:
            if routing['decision']['action']=='baseline':
                checks=routing.get('dense_receipts',[])
                names=sorted(f'model.layers.{i}.self_attn.attn' for i in range(64))
                if sorted(c['rank'] for c in checks)!=list(range(4)) or any(
                        c['native_layers']!=names or c['store_operations']!=dict(lookup=0,load=0,store=0) for c in checks):
                    raise ValueError('Missing accepted native dense bypass evidence')
            from runner.router_policy import decide
            if decide(load_policy(Path(root)/'trees.json'),case,routing['features']) != routing['decision']:
                raise ValueError('Accepted router decision changed')
            probe=json.loads((folder/'routing/probe-diagnostics.json').read_text())
            verify_diagnostics(probe,sample,'prophetkv',.01,4,range(64))
            baseline=accepted(root,action,row,protocol,full=False)
            if baseline is None:
                raise ValueError('Missing paired fixed action')
            paired_check(record,diagnostics,baseline,json.loads((record_dir(root,action,row['id'])/'diagnostics.json').read_text()))
    return record


def run_group(root, group, case, attempt):
    from runner.generation import generate, verify_diagnostics
    from runner.worker import setup, arm, drain
    from runner.reporting import OutputAnalyzer, evaluation_metadata, score_answer
    from scripts.router_inputs import load_prepared
    from runner.cache import wait_for_cache
    from ucm.sparse.prophetkv.lifecycle import seed_value, delete_retired_files
    root=Path(root).resolve();protocol=json.loads((root/'protocol.json').read_text())
    rows=[r for r in load_prepared(protocol['prepared']) if r['ordinal']%2==group]
    devices=check_environment(4)
    if devices != protocol['groups'][group]:
        raise ValueError('GPU group changed')
    pending=[r for r in rows if accepted(root,case,r,protocol) is None]
    session_root=root/'sessions'/f'{case}-group{group}-attempt{attempt}'
    session_root.mkdir(parents=True,exist_ok=True)
    if not pending:
        atomic_json(session_root/'complete.json',dict(complete=True,skipped=len(rows),engine_started=False));return
    with (root/f'group{group}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cached=case!='baseline';is_router=case.startswith('router')
        cache=Path(protocol['cache_root'])/fingerprint(str(root))[:16]/f'group{group}'/uuid.uuid4().hex
        cache.mkdir(parents=True,exist_ok=False)
        atomic_json(session_root/'ownership.json',dict(cache=str(cache),group=group,pid=os.getpid(),identity=identity(os.getpid())))
        samples=[]
        for row in pending:
            path=Path(protocol['prepared'])/row['prepared']
            if file_hash(path)!=row['sha256']:raise ValueError('Prepared input changed')
            samples.append((row,json.loads(path.read_text())))
        for _,s in samples:validate_profile(s,4)
        policy=load_policy(root/'trees.json') if is_router else None
        if is_router:
            from runner.layout import validate_policy_protocol
            for _,s in samples:validate_policy_protocol(policy,s)
        first=samples[0][1]
        cfg=engine_config(protocol['model'],'prophetkv' if cached else 'baseline',
            ratio=.01 if is_router else RATIOS[case],tp=4,cache_dir=cache,
            end_token=None,router_profile=True)
        if cached:
            cfg['kv_transfer_config']['kv_connector_extra_config']['temporary_layouts']={
                r['id']:s['boundaries'] for r,s in samples}
        scheduler=session_root/'scheduler.json'
        os.environ['PROPHETKV_SCHEDULER_RECEIPT']=str(scheduler)
        atomic_json(session_root/'config.json',cfg)
        start=time.perf_counter();llm=start_engine(cfg)
        session=dict(model_load_seconds=time.perf_counter()-start)
        try:
            session['setup']=llm.collective_rpc(setup)
            for receipt in session['setup']:
                if receipt['kv_tokens']!=65920 or receipt['kv_blocks']!=1031 or receipt['block_size']!=64:
                    raise ValueError('Independent KV allocation audit failed')
                if len(receipt['rope'])!=64 or any(not a['formula_bitwise_equal'] or a['table_positions']!=131072 for a in receipt['rope']):
                    raise ValueError('Scheduled YaRN4 initialization audit failed')
            analyzer=OutputAnalyzer(llm.get_tokenizer())
            start=time.perf_counter()
            warm_namespace=uuid.uuid4().hex
            warm_id=f"{warm_namespace}:{pending[0]['id']}:populate:warmup"
            warm=[100,200,300,400]*32
            generate(llm.llm_engine,warm,1,warm_id,phase='populate' if cached else None)
            if cached:
                ready_args=SimpleNamespace(model_path=Path(protocol['model']),tensor_parallel_size=4,cache_dir=cache,
                    cache_ready_timeout_seconds=600,hash_seed=seed_value(warm_namespace))
                session['warmup_readiness']=wait_for_cache(ready_args,[warm])
                if session['warmup_readiness']['verified_shards']!=8:
                    raise ValueError('Expected eight committed TP4 warmup shards')
            retirement(llm,4)
            if cached:
                files={str(p.relative_to(cache)) for p in (cache/'kv').rglob('*') if p.is_file()}
                delete_retired_files(cache,files)
            session['warmup_seconds']=time.perf_counter()-start
            atomic_json(session_root/'initialization.json',dict(session=session,engine_config=cfg))
            initialization_sha256=file_hash(session_root/'initialization.json')
            for row,sample in samples:
                folder=record_dir(root,case,row['id'])
                # Failed attempts retain their diagnostics; accepted results are never replaced.
                if folder.exists():
                    history=root/'incomplete'/f'{case}-{row["id"]}-{uuid.uuid4().hex}'
                    history.parent.mkdir(exist_ok=True);folder.rename(history)
                folder.mkdir(parents=True)
                namespace=uuid.uuid4().hex;tag=row['id'];pc=None
                timings={};construction=None
                if cached:
                    start=time.perf_counter();pc=PromptCache(cache,Path(protocol['model']),4,namespace,sample)
                    for i,chunk in enumerate(pc.chunks):
                        generate(llm.llm_engine,list(chunk),1,f'{namespace}:{tag}:populate:{i}',phase='populate')
                        retirement(llm,4)
                    timings['construction_seconds']=time.perf_counter()-start
                    start=time.perf_counter();construction=pc.ready()
                    timings['readiness_seconds']=time.perf_counter()-start
                else:
                    timings.update(construction_seconds=0.,readiness_seconds=0.)
                prime=f'{namespace}:{tag}:read:prime'
                start=time.perf_counter()
                if cached:
                    arm_request(llm,sample,prime,'prophetkv-all64-1' if is_router else case,4)
                generate(llm.llm_engine,sample['token_ids'],1,prime,False,sample=sample)
                llm.collective_rpc(drain);retirement(llm,4)
                timings['priming_seconds']=time.perf_counter()-start
                if pc:pc.unchanged()
                rid=f'{namespace}:{tag}:read:measured';routing=None
                if is_router:
                    result,ttft,elapsed,routing=measure(llm,sample,rid,policy,case,folder/'routing',4,pc.unchanged,scheduler)
                    rid=routing['answer_request_id']
                else:
                    if cached:arm_request(llm,sample,rid,case,4)
                    result,ttft,elapsed=generate(llm.llm_engine,sample['token_ids'],sample['max_output_tokens'],rid,False,sample=sample)
                timings.update(ttft_seconds=ttft,generation_seconds=elapsed,
                    answer_engine_ttft_seconds=routing['answer_engine_ttft_seconds'] if routing else ttft,
                    routing_overhead_seconds=routing['routing_overhead_seconds'] if routing else 0.)
                start=time.perf_counter();diagnostics=llm.collective_rpc(drain)
                if routing:validate_answer(llm,diagnostics,result,sample,routing,4)
                else:verify_diagnostics(diagnostics,sample,'prophetkv' if cached else 'baseline',RATIOS[case],4,range(64))
                if not cached and result.num_cached_tokens!=0:
                    raise ValueError('Standalone baseline reused KV')
                atomic_json(folder/'diagnostics.json',diagnostics)
                timings['validation_export_seconds']=time.perf_counter()-start
                start=time.perf_counter();retired=verify_retired(llm,4,rid,scheduler if cached else None)
                if pc:pc.unchanged()
                timings['answer_retirement_seconds']=time.perf_counter()-start
                output=result.outputs[0]
                metrics=analyzer.analyze(list(output.token_ids),sample['token_ids'],False)
                evaluation=evaluation_metadata({},row)
                record=dict(**sample_provenance(sample),prompt_id=row['id'],method=case,group=group,gpu_uuids=devices,input_sha256=row['sha256'],
                    prompt_tokens=len(sample['token_ids']),output_token_ids=list(output.token_ids),prediction=output.text,
                    finish_reason=output.finish_reason,max_output_tokens=sample['max_output_tokens'],num_cached_tokens=result.num_cached_tokens,
                    output_cap_reached=len(output.token_ids)>=sample['max_output_tokens'],timings=timings,routing=routing,
                    retirement=retired,cache_immutable=True,construction=construction,
                    session=str(session_root.relative_to(root)),initialization_sha256=initialization_sha256,**metrics,**evaluation,
                    accuracy=score_answer(scoring_text(output.text,metrics,evaluation),evaluation))
                if routing:
                    action=routing['decision']['action'];fixed=accepted(root,action,row,protocol)
                    if fixed is None:raise ValueError('Fixed action must be accepted before router measurement')
                    record['paired_validation']=paired_check(record,diagnostics,fixed,
                        json.loads((record_dir(root,action,row['id'])/'diagnostics.json').read_text()))
                if pc:record['cache_deletion']=pc.delete()
                atomic_json(folder/'result.json',record)
                files={str(p.relative_to(folder)):file_hash(p) for p in folder.rglob('*') if p.is_file()}
                atomic_json(folder/'validated.json',dict(complete=True,protocol_sha256=fingerprint(protocol),files=files))
                atomic_json(root/f'progress-group{group}.json',dict(case=case,prompt_id=row['id'],accepted_at=time.time(),pid=os.getpid()))
                print(f'{case} {row["id"]}: accepted; total TTFT {ttft:.3f}s',flush=True)
        finally:
            llm.llm_engine.engine_core.shutdown()
        atomic_json(session_root/'complete.json',dict(complete=True,engine_shutdown=True,session=session))
        # Only this attempt's empty directory is eligible for cleanup.
        if any((cache/'kv').rglob('*')):
            # Empty prefix directories may remain; no KV files may remain.
            if any(p.is_file() for p in (cache/'kv').rglob('*')):
                raise ValueError('Cache files remain after retirement')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--group',type=int,choices=(0,1),required=True)
    p.add_argument('--case',choices=CASES,required=True);p.add_argument('--attempt',required=True)
    a=p.parse_args()
    try:run_group(a.root,a.group,a.case,a.attempt)
    except (TimeoutError,ConnectionError,OSError):
        import traceback;traceback.print_exc();sys.exit(75)
    except Exception:
        import traceback;traceback.print_exc();sys.exit(76)


if __name__=='__main__':
    import sys
    main()
