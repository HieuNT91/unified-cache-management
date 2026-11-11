"""Shared collection and cross-dataset inference lifecycle; GPU imports are lazy."""
import json
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from runner.cache.setups import atomic_json,file_hash,fingerprint,retirement,start_engine
from runner.runtime.capacity import allocation,config,validate,validate_initialization


def clean_capture(llm,tp=4):
    from runner.corpus.worker import capture_state
    replies=llm.collective_rpc(capture_state)
    if sorted(r['rank'] for r in replies)!=list(range(tp)) or any(r['capture'] or r['arrays'] for r in replies):
        raise ValueError('Attention capture leaked into timed answer')


class Engine:
    def __init__(self,root,protocol,group,phase,attempt,rows):
        from runner.runtime.worker import setup
        from runner.runtime.generation import generate
        from runner.reporting.aggregate import OutputAnalyzer
        from runner.runtime.process import identity
        from runner.cache.lifecycle import wait_for_cache
        from ucm.sparse.prophetkv.lifecycle import seed_value,delete_retired_files
        from runner.runtime.tensor_parallel import protocol_tp
        from runner.tree.collection import inventory, configure_engine, runtime_sample
        from runner.router.adatree import generation_sample
        if phase not in ('cached', 'baseline'):
            raise ValueError('Expected cached or baseline phase')
        inventory(protocol)
        self.tp=protocol_tp(protocol)
        self.profile=protocol.get('execution_profile',protocol['dataset'])
        self.root=Path(root);self.protocol=protocol;self.group=group;self.cached=phase=='cached';self.pc=None
        self.session=self.root/'sessions'/f'{phase}-group{group}-{attempt}'
        self.session.mkdir(parents=True,exist_ok=False)
        self.cache=Path(protocol['cache_root'])/fingerprint(str(self.root))[:16]/f'group{group}'/uuid.uuid4().hex
        self.cache.mkdir(parents=True,exist_ok=False)
        atomic_json(self.session/'ownership.json',dict(cache=str(self.cache),group=group,pid=os.getpid(),identity=identity(os.getpid())))
        self.scheduler=self.session/'scheduler.json'
        os.environ['PROPHETKV_SCHEDULER_RECEIPT']=str(self.scheduler)
        for row in rows:
            sample=json.loads((Path(protocol['prepared'])/row['prepared']).read_text())
            validate(sample, self.profile)
            effective = generation_sample(sample, protocol)
            if len(effective['token_ids'])+effective['max_output_tokens']>allocation(self.profile)['kv_tokens']:
                raise ValueError('Complete input plus output reserve exceeds assigned hardware KV capacity')
        cfg=config(protocol['model'],self.profile,self.cached,self.cache,
                   tp=self.tp,memory=protocol.get('gpu_memory_utilization',.95))
        if self.cached:
            configure_engine(cfg, self.cache, protocol)
            if 'position_attention_kernel' in protocol:
                cfg['kv_transfer_config']['kv_connector_extra_config']['position_attention_kernel']=protocol['position_attention_kernel']
            def context_layout(row):
                sample = json.loads((Path(protocol['prepared'])/row['prepared']).read_text())
                return runtime_sample(sample, protocol)['boundaries']
            cfg['kv_transfer_config']['kv_connector_extra_config']['temporary_layouts'] = {
                r['id']: context_layout(r) for r in rows}
        start=time.perf_counter();self.llm=start_engine(cfg)
        try:
            from runner.operations import code_hashes
            init=dict(model_load_seconds=time.perf_counter()-start,engine_config=cfg,execution_code=code_hashes())
            init['setup']=self.llm.collective_rpc(setup);validate_initialization(init['setup'],self.profile,tp=self.tp)
            self.analyzer=OutputAnalyzer(self.llm.get_tokenizer())
            warmup_started=time.perf_counter()
            warm_namespace=uuid.uuid4().hex;warm=[100,200,300,400]*32
            generate(self.llm.llm_engine,warm,f"{warm_namespace}:{rows[0]['id']}:populate:warmup",phase='populate')
            if self.cached:
                args=SimpleNamespace(model_path=Path(protocol['model']),tensor_parallel_size=self.tp,cache_dir=self.cache,
                    cache_ready_timeout_seconds=600,hash_seed=seed_value(warm_namespace))
                init['warmup_readiness']=wait_for_cache(args,[warm])
                if init['warmup_readiness']['verified_shards']!=2*self.tp:raise ValueError('Expected two committed warmup shards per TP rank')
            retirement(self.llm,self.tp)
            if self.cached:delete_retired_files(self.cache,{str(p.relative_to(self.cache)) for p in (self.cache/'kv').rglob('*') if p.is_file()})
            if self.cached:
                from runner.tree.collection import retire_warmup
                retire_warmup(self.cache, warm)
            init['warmup_seconds']=time.perf_counter()-warmup_started
            init['validated']=True;atomic_json(self.session/'initialization.json',init)
        except BaseException:
            self.llm.llm_engine.engine_core.shutdown();raise

    def begin(self,row):
        """Construct and commit context KV; the next read is the requested work."""
        from runner.tree.collection import TreePromptCache as PromptCache, runtime_sample
        from runner.runtime.generation import generate
        self.row=row;self.sample=json.loads((Path(self.protocol['prepared'])/row['prepared']).read_text())
        validate(self.sample,self.protocol.get('execution_profile',self.protocol['dataset']));self.namespace=uuid.uuid4().hex;self.construction={}
        if file_hash(Path(self.protocol['prepared'])/row['prepared'])!=row['sha256']:raise ValueError('Input changed')
        if self.sample['model_config_sha256']!=file_hash(Path(self.protocol['model'])/'config.json'):
            raise ValueError('Prepared model mismatch')
        start=time.perf_counter()
        if self.cached:
            from runner.tree.prophetkv import budget
            original_bounds = list(self.sample['boundaries'])
            self.sample = runtime_sample(self.sample, self.protocol)
            if self.protocol.get('kv_chunk_size') is not None:
                self.construction['layout'] = dict(kv_chunk_size=self.protocol['kv_chunk_size'],
                    prepared_boundaries=original_bounds, runtime_boundaries=list(self.sample['boundaries']),
                    token_sha256=self.sample['token_sha256'])
            for percentage in self.protocol['percentages']:
                budget(self.sample['boundaries'], percentage/100)
            self.pc=PromptCache(self.cache,Path(self.protocol['model']),self.tp,self.namespace,self.sample)
            for i,chunk in enumerate(self.pc.chunks):
                generate(self.llm.llm_engine,list(chunk),self.rid(f'populate:{i}'),phase='populate')
                retirement(self.llm,self.tp)
            self.construction['construction_seconds']=time.perf_counter()-start;start=time.perf_counter()
            self.construction['readiness']=self.pc.ready()
            self.construction['readiness_seconds']=time.perf_counter()-start
            self.pc.unchanged()
            clean_capture(self.llm,self.tp)

    def rid(self,suffix):return f"{self.namespace}:{self.row['id']}:{suffix}"

    def common(self,case,timings,retired):
        from runner.corpus.worker import alignment_state
        path=self.session/'initialization.json'
        alignment=self.llm.collective_rpc(alignment_state) if self.cached else []
        if self.cached and (sorted(a['rank'] for a in alignment)!=list(range(self.tp)) or
                any(not a['normalized'] or a['delta_amplitude']!=1. or a['aliases_native_table'] for a in alignment)):
            raise ValueError('YaRN delta normalization was lost during generation')
        from runner.inputs.layout import sample_provenance
        return dict(**sample_provenance(self.sample),prompt_id=self.row['id'],method=case,group=self.group,gpu_uuids=self.protocol['groups'][self.group],
            hardware=self.protocol.get('hardware',{}),input_sha256=self.row['sha256'],cache_immutable=True,
            retirement=retired,initialization=str(path.relative_to(self.root)),initialization_sha256=file_hash(path),
            timings=timings,construction=self.construction,alignment_audit=alignment)

    def answer(self, definition, case, *, seed):
        from runner.tree.collection import inventory
        from runner.router.adatree import answer
        if (inventory(self.protocol).get(case) != definition
                or self.cached != (case == 'adatree-v1')):
            raise ValueError('AdaTree requires cached KV; no-cache requires a connector-free engine')
        return answer(self, seed, baseline=case == 'nocache')

    def end(self):
        if self.pc:
            deletion=self.pc.delete();self.pc=None;return deletion
        return dict(deleted_shards=0)

    def record_failure(self,error,phase,case=None):
        """Preserve diagnostic metadata in the session before cache cleanup."""
        import traceback
        from runner.cache.lifecycle import failure_inventory
        receipt=dict(phase=phase,case=case,group=self.group,
                     prompt_id=getattr(self,'row',{}).get('id'),
                     namespace=getattr(self,'namespace',None),
                     error_type=type(error).__name__,error=str(error),
                     traceback=''.join(traceback.format_exception(type(error),error,error.__traceback__)),
                     initialization=str(self.session/'initialization.json'))
        if self.cached:
            receipt['inventory']=failure_inventory(self.cache,self.pc.files if self.pc else ())
        path=self.session/'failure.json'
        atomic_json(path,receipt)
        return path

    def close(self):
        self.llm.llm_engine.engine_core.shutdown()
        atomic_json(self.session/'complete.json',dict(complete=True,engine_shutdown=True))
