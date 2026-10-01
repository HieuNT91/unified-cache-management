"""Shared collection and cross-dataset inference lifecycle; GPU imports are lazy."""
from runner.reporting import scoring_text
import json
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from runner.setups import atomic_json,file_hash,fingerprint,retirement,start_engine
from runner.router_measure import verify_retired
from runner.corpus import load_attention,match_answer,selection
from runner.tree_policy import actions,decide
from runner.router_policy import features_from_arrays
from runner.tree_profiles import config,validate,validate_initialization
from runner.corpus_records import record_dir,accepted,publish

PROBE=dict(id='independent-probe',method='prophetkv',ratio=.01)


def arm_tree(llm,sample,rid,definition,capture=False):
    from runner.corpus_worker import tree_mode
    from runner.worker import arm
    replies=llm.collective_rpc(tree_mode,kwargs=dict(request_id=rid,definition=definition,capture=capture))
    if sorted(r['rank'] for r in replies)!=list(range(4)) or any(
            r!=dict(rank=r['rank'],request_id=rid,definition=definition,capture=capture) for r in replies):
        raise ValueError('TP action synchronization failed')
    if definition['method']!='baseline':
        llm.collective_rpc(arm,kwargs=dict(request_id=rid,boundaries=sample['boundaries'],question_positions=sample['question_positions']))
    return replies


def clean_capture(llm):
    from runner.corpus_worker import capture_state
    replies=llm.collective_rpc(capture_state)
    if sorted(r['rank'] for r in replies)!=list(range(4)) or any(r['capture'] or r['arrays'] for r in replies):
        raise ValueError('Attention capture leaked into timed answer')


class Engine:
    def __init__(self,root,protocol,group,phase,attempt,rows):
        from runner.worker import setup
        from run import generate
        from runner.reporting import OutputAnalyzer
        from runner.router_process import identity
        from runner.cache import wait_for_cache
        from ucm.sparse.prophetkv.lifecycle import seed_value,delete_retired_files
        self.root=Path(root);self.protocol=protocol;self.group=group;self.cached=phase=='cached';self.pc=None
        self.session=self.root/'sessions'/f'{phase}-group{group}-{attempt}'
        self.session.mkdir(parents=True,exist_ok=False)
        self.cache=Path(protocol['cache_root'])/fingerprint(str(self.root))[:16]/f'group{group}'/uuid.uuid4().hex
        self.cache.mkdir(parents=True,exist_ok=False)
        atomic_json(self.session/'ownership.json',dict(cache=str(self.cache),group=group,pid=os.getpid(),identity=identity(os.getpid())))
        self.scheduler=self.session/'scheduler.json'
        os.environ['PROPHETKV_SCHEDULER_RECEIPT']=str(self.scheduler)
        first=json.loads((Path(protocol['prepared'])/rows[0]['prepared']).read_text());validate(first,protocol['dataset'])
        for row in rows:
            sample=json.loads((Path(protocol['prepared'])/row['prepared']).read_text())
            from runner.tree_profiles import allocation
            if len(sample['token_ids'])+sample['max_output_tokens']>allocation(protocol['dataset'],protocol.get('hardware_profile','server'))['kv_tokens']:
                raise ValueError('Complete input plus output reserve exceeds assigned hardware KV capacity')
        cfg=config(protocol['model'],protocol['dataset'],self.cached,self.cache,None,
                   protocol.get('hardware_profile','server'))
        if self.cached:
            cfg['kv_transfer_config']['kv_connector_extra_config']['temporary_layouts']={
                r['id']:json.loads((Path(protocol['prepared'])/r['prepared']).read_text())['boundaries'] for r in rows}
        start=time.perf_counter();self.llm=start_engine(cfg)
        try:
            init=dict(model_load_seconds=time.perf_counter()-start,engine_config=cfg)
            init['setup']=self.llm.collective_rpc(setup);validate_initialization(init['setup'],protocol['dataset'],protocol.get('hardware_profile','server'))
            self.analyzer=OutputAnalyzer(self.llm.get_tokenizer())
            warmup_started=time.perf_counter()
            warm_namespace=uuid.uuid4().hex;warm=[100,200,300,400]*32
            generate(self.llm.llm_engine,warm,1,f"{warm_namespace}:{rows[0]['id']}:populate:warmup",phase='populate')
            if self.cached:
                args=SimpleNamespace(model_path=Path(protocol['model']),tensor_parallel_size=4,cache_dir=self.cache,
                    cache_ready_timeout_seconds=600,hash_seed=seed_value(warm_namespace))
                init['warmup_readiness']=wait_for_cache(args,[warm])
                if init['warmup_readiness']['verified_shards']!=8:raise ValueError('Expected eight committed TP4 warmup shards')
            retirement(self.llm,4)
            if self.cached:delete_retired_files(self.cache,{str(p.relative_to(self.cache)) for p in (self.cache/'kv').rglob('*') if p.is_file()})
            init['warmup_seconds']=time.perf_counter()-warmup_started
            init['validated']=True;atomic_json(self.session/'initialization.json',init)
        except BaseException:
            self.llm.llm_engine.engine_core.shutdown();raise

    def begin(self,row):
        from runner.sweep import PromptCache
        from run import generate
        from runner.worker import drain
        self.row=row;self.sample=json.loads((Path(self.protocol['prepared'])/row['prepared']).read_text())
        validate(self.sample,self.protocol['dataset']);self.namespace=uuid.uuid4().hex;self.construction={}
        if file_hash(Path(self.protocol['prepared'])/row['prepared'])!=row['sha256']:raise ValueError('Input changed')
        start=time.perf_counter()
        if self.cached:
            self.pc=PromptCache(self.cache,Path(self.protocol['model']),4,self.namespace,self.sample)
            for i,chunk in enumerate(self.pc.chunks):
                generate(self.llm.llm_engine,list(chunk),1,self.rid(f'populate:{i}'),phase='populate')
                retirement(self.llm,4)
            self.construction['construction_seconds']=time.perf_counter()-start;start=time.perf_counter()
            self.construction['readiness']=self.pc.ready()
            self.construction['readiness_seconds']=time.perf_counter()-start
        rid=self.rid('read:prime');start=time.perf_counter()
        if self.cached:arm_tree(self.llm,self.sample,rid,PROBE)
        generate(self.llm.llm_engine,self.sample['token_ids'],1,rid,self.sample['thinking'],sample=self.sample)
        self.llm.collective_rpc(drain);verify_retired(self.llm,4,rid,self.scheduler if self.cached else None)
        if self.pc:self.pc.unchanged();clean_capture(self.llm)
        self.construction['priming_seconds']=time.perf_counter()-start

    def rid(self,suffix):return f"{self.namespace}:{self.row['id']}:{suffix}"

    def common(self,case,timings,retired):
        from runner.corpus_worker import alignment_state
        path=self.session/'initialization.json'
        alignment=self.llm.collective_rpc(alignment_state) if self.cached else []
        if self.cached and (sorted(a['rank'] for a in alignment)!=list(range(4)) or
                any(not a['normalized'] or a['delta_amplitude']!=1. or a['aliases_native_table'] for a in alignment)):
            raise ValueError('YaRN delta normalization was lost during generation')
        from runner.layout import sample_provenance
        return dict(**sample_provenance(self.sample),prompt_id=self.row['id'],method=case,group=self.group,gpu_uuids=self.protocol['groups'][self.group],
            hardware=self.protocol.get('hardware',{}),input_sha256=self.row['sha256'],cache_immutable=True,
            retirement=retired,initialization=str(path.relative_to(self.root)),initialization_sha256=file_hash(path),
            timings=timings,construction=self.construction,alignment_audit=alignment)

    def probe(self,folder):
        import numpy as np
        from run import generate,verify_diagnostics
        from runner.worker import drain
        from runner.corpus_worker import export as export_attention
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        started=time.perf_counter();rid=self.rid('read:probe-'+uuid.uuid4().hex)
        arm_tree(self.llm,self.sample,rid,PROBE,True)
        sync_seconds=time.perf_counter()-started
        generation_start=time.perf_counter()
        internal,ttft,_=generate(self.llm.llm_engine,self.sample['token_ids'],1,rid,False,sample=self.sample)
        generation_seconds=time.perf_counter()-generation_start
        if len(internal.outputs[0].token_ids)!=1:raise ValueError('Probe must produce one discarded token')
        del internal
        export_start=time.perf_counter()
        artifacts=self.llm.collective_rpc(export_attention,kwargs=dict(output=str(folder),sample=self.sample,inventory=self.protocol['actions']))
        export_seconds=time.perf_counter()-export_start
        validation_start=time.perf_counter()
        ds=self.llm.collective_rpc(drain);verify_diagnostics(ds,self.sample,'prophetkv',.01,4,range(64))
        diagnostics_seconds=time.perf_counter()-validation_start
        retirement_start=time.perf_counter()
        retired=verify_retired(self.llm,4,rid,self.scheduler);self.pc.unchanged();clean_capture(self.llm)
        retirement_seconds=time.perf_counter()-retirement_start
        replay_start=time.perf_counter()
        attention=load_attention(folder,dict(artifacts=artifacts),self.sample,self.protocol['actions'])
        # Probe selection may be absent from the answer inventory.
        for worker in ds:
            event=next(e for e in worker['diagnostics'] if e['kind']=='prophetkv_selection')
            if (not np.array_equal(np.asarray(event['scores'],np.float32),attention['scores']) or
                    not np.array_equal(event['selected_positions'],selection(attention['scores'],self.sample['boundaries'][1],.01))):
                raise ValueError('Probe diagnostics differ from archived native scores')
        replay_seconds=time.perf_counter()-replay_start
        features_start=time.perf_counter()
        features=features_from_arrays(attention['layers'].astype(np.float64).mean(0),attention['scores'],
            self.sample['boundaries'][1],self.sample['boundaries'][-2])
        features_seconds=time.perf_counter()-features_start
        overhead=time.perf_counter()-started
        result=self.common('probe',dict(probe_ttft_seconds=ttft,routing_overhead_seconds=overhead,
            archive_writing_seconds=max(a['serialization_seconds'] for a in artifacts),
            tp_sync_seconds=sync_seconds,probe_generation_seconds=generation_seconds,
            export_rpc_seconds=export_seconds,diagnostics_seconds=diagnostics_seconds,
            retirement_and_cache_check_seconds=retirement_seconds,attention_replay_seconds=replay_seconds,
            feature_extraction_seconds=features_seconds),retired)
        result.update(artifacts=artifacts,features=features,internal_tokens=1)
        return result,ds,attention

    def answer(self,definition,case,attention=None,overhead=0.,route_started=None):
        from run import generate,verify_diagnostics
        from runner.worker import drain,router_dense_receipt
        from runner.reporting import evaluation_metadata,score_answer
        rid=self.rid('read:'+case+('-'+uuid.uuid4().hex))
        dense=definition['method']=='baseline'
        if self.cached and dense:rid+='|router-dense'
        start=time.perf_counter()
        if self.cached:
            clean_capture(self.llm);sync=arm_tree(self.llm,self.sample,rid,definition)
        else:sync=[]
        sync_seconds=time.perf_counter()-start
        routing_time=time.perf_counter()-route_started if route_started is not None else overhead+sync_seconds
        output,ttft,elapsed=generate(self.llm.llm_engine,self.sample['token_ids'],self.sample['max_output_tokens'],rid,self.sample['thinking'],sample=self.sample)
        ds=self.llm.collective_rpc(drain)
        verify_diagnostics(ds,self.sample,definition['method'],definition['ratio'],4,range(64))
        if dense and output.num_cached_tokens!=0:raise ValueError('Dense request reused KV')
        audits=self.llm.collective_rpc(router_dense_receipt) if dense and self.cached else []
        if not dense:
            if attention is None:raise ValueError('Sparse answer requires independent probe replay')
            match_answer(ds,self.sample,definition['id'],attention,self.protocol['actions'])
        retired=verify_retired(self.llm,4,rid,self.scheduler if self.cached else None)
        if self.pc:self.pc.unchanged();clean_capture(self.llm)
        tokens=list(output.outputs[0].token_ids);metrics=self.analyzer.analyze(tokens,self.sample['token_ids'],self.sample['thinking'])
        evaluation=evaluation_metadata({},self.row)
        record=self.common(case,dict(answer_engine_ttft_seconds=ttft,ttft_seconds=routing_time+ttft,
            generation_seconds=elapsed,routing_overhead_seconds=routing_time,tp_sync_seconds=sync_seconds),retired)
        record.update(executed_action=definition['id'],output_token_ids=tokens,prediction=output.outputs[0].text,num_cached_tokens=output.num_cached_tokens,
            output_cap_reached=len(tokens)>=self.sample['max_output_tokens'],max_output_tokens=self.sample['max_output_tokens'],
            dense_receipts=audits,decision_sync=sync,**metrics,**evaluation,accuracy=score_answer(scoring_text(output.outputs[0].text,metrics,evaluation),evaluation))
        return record,ds

    def end(self):
        if self.pc:
            deletion=self.pc.delete();self.pc=None;return deletion
        return dict(deleted_shards=0)

    def close(self):
        self.llm.llm_engine.engine_core.shutdown()
        atomic_json(self.session/'complete.json',dict(complete=True,engine_shutdown=True))


def paired(routed,diagnostics,control,control_diagnostics):
    from runner.router_sweep import selection_signature
    if any(routed[k]!=control[k] for k in ('prompt_id','group','gpu_uuids','input_sha256','output_token_ids')):
        raise ValueError('Routed output differs from same-group fixed control')
    if routed.get('executed_action')!=control.get('executed_action'):raise ValueError('Routed/control actions differ')
    if selection_signature(diagnostics)!=selection_signature(control_diagnostics):raise ValueError('Routed scores/selections differ from fixed control')
    return dict(output_tokens_equal=True,scores_masks_equal=True,group=routed['group'])


def infer_prompt(engine,tree,folder,baseline,baseline_diagnostics):
    """Fresh probe -> retirement/features -> decision -> routed answer -> control."""
    from runner.layout import validate_policy_protocol
    validate_policy_protocol(tree,engine.sample)
    started=time.perf_counter()
    probe,probe_ds,attention=engine.probe(Path(folder)/'probe')
    decision=decide(tree,probe['features']);definition=actions(tree['actions'])[decision['action']]
    overhead=time.perf_counter()-started
    routed,diagnostics=engine.answer(definition,'router',attention,overhead,route_started=started)
    if decision['action']=='nocache':control,control_ds=baseline,baseline_diagnostics
    else:control,control_ds=engine.answer(definition,'control',attention)
    routed.update(decision=decision,tree_sha256=tree['payload_sha256'],probe=probe,
                  paired_validation=paired(routed,diagnostics,control,control_ds))
    atomic_json(Path(folder)/'probe-diagnostics.json',probe_ds)
    atomic_json(Path(folder)/'control.json',control);atomic_json(Path(folder)/'control-diagnostics.json',control_ds)
    return routed,diagnostics
