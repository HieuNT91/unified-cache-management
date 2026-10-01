"""Study collection and routed answers, with serialization outside live TTFT."""
import json
import time
import uuid
from pathlib import Path
import numpy as np
from runner.corpus_runtime import Engine,arm_tree,PROBE,clean_capture
from runner.gpu_study_features import CPU_FEATURES,GPU_FEATURES,REGISTERED,reduce_statistics,additional_features,base_features
from runner.router_policy import FEATURES,features_from_arrays
from runner.gpu_study_policy import decide
from runner.setups import atomic_json,file_hash
from runner.corpus import selection,reduction_matches,match_answer
from runner import gpu_study_worker as rpc
from runner.gpu_study_transport import unpack


def native_validate(ranks,sample,saved):
    if sorted(r['rank'] for r in ranks)!=list(range(4)):raise ValueError('Missing ranks')
    prefix,end=sample['boundaries'][1],sample['boundaries'][-2];means=[]
    for r in ranks:
        a=r['layers'];scores=r['scores']
        if a.shape!=(64,end) or a.dtype!=np.float32 or not np.isfinite(a).all() or (a<0).any():raise ValueError('Invalid native layer capture')
        total=np.zeros(end,np.float32)
        for layer in a:np.add(total,layer,out=total)
        total/=np.float32(64)
        if not np.array_equal(total,r['local_mean']) or not np.array_equal(scores,saved['scores']):raise ValueError('Instrumentation changed native scores')
        if r['calls']!=1 or r['scratch_bound_bytes']>256*2**20:raise ValueError('Probe/scratch gate failed')
        means.append(total)
    if not reduction_matches(means,ranks[0]['scores'],prefix):raise ValueError('Native TP replay failed')


def reference_validate(ranks,prefix):
    """Independent CPU FP32 reference for one full head and one full query/layer/rank."""
    checked=0;max_error=0.
    for r in ranks:
        for layer,q,k,head,query in r['references']:
            q=q.astype(np.float32);k=k.astype(np.float32);nq,nh,d=q.shape;group=nh//k.shape[1]
            def probabilities(query,keys):
                logits=np.matmul(query,keys.T)/np.float32(d**.5)
                logits-=logits.max(-1,keepdims=True);p=np.exp(logits);p/=p.sum(-1,keepdims=True)
                return p[:,prefix:]
            h=probabilities(q[:,0],k[:,0]).mean(0)
            z=np.zeros(len(k)-prefix,np.float32)
            for i in range(nh):z+=probabilities(q[:1,i],k[:,i//group])[0]/np.float32(nh)
            for actual,expected in ((head,h),(query,z)):
                if actual is None:continue
                error=float(np.max(np.abs(actual-expected)));max_error=max(max_error,error)
                if not np.allclose(actual,expected,rtol=3e-4,atol=2e-7):raise ValueError('Bounded FP32 statistics/reference mismatch')
                checked+=1
    if checked!=64:raise ValueError('Missing scheduled reference checks')
    return dict(checks=checked,max_abs_error=max_error,rtol=3e-4,atol=2e-7)


def resident_features(ranks,sample,required):
    start=time.perf_counter();prefix,end=sample['boundaries'][1],sample['boundaries'][-2]
    # Original equal-rank float64 reduction order is preserved exactly.
    layers=np.stack([r['layers'] for r in ranks]).astype(np.float64).mean(0)
    aggregation=time.perf_counter()-start;start=time.perf_counter();features={}
    if set(required)&set(FEATURES):features.update(base_features(layers,ranks[0]['scores'],prefix,end,[f for f in required if f in FEATURES]))
    base=time.perf_counter()-start;start=time.perf_counter()
    extra=[f for f in required if f in CPU_FEATURES]
    if extra:features.update(additional_features(layers,ranks[0]['scores'],prefix,end,extra))
    cpu=time.perf_counter()-start;start=time.perf_counter()
    gpu=[f for f in required if f in GPU_FEATURES]
    if gpu:features.update(reduce_statistics(ranks,gpu))
    reduction=time.perf_counter()-start
    return {f:features[f] for f in required},dict(aggregation_seconds=aggregation,base_cpu_seconds=base,cpu_extra_seconds=cpu,gpu_scalar_reduction_seconds=reduction),layers


class StudyEngine(Engine):
    def probe_study(self,required,reference=False):
        from run import generate,verify_diagnostics
        from runner.worker import drain
        from runner.router_measure import verify_retired
        rid=self.rid('read:study-probe-'+uuid.uuid4().hex)
        binding=dict(input_sha256=self.row['sha256'],cache_identity=self.namespace,runtime_sha256=self.protocol['runtime_sha256'],
                     gpu_uuids=self.protocol['groups'][0],source_request=rid)
        start=time.perf_counter()
        arm_tree(self.llm,self.sample,rid,PROBE,True)
        self.llm.collective_rpc(rpc.arm_probe,kwargs=dict(binding=binding,required=required,reference=reference))
        output,ttft,elapsed=generate(self.llm.llm_engine,self.sample['token_ids'],1,rid,False,sample=self.sample)
        if len(output.outputs[0].token_ids)!=1:raise ValueError('Expected one discarded probe token')
        del output
        export_start=time.perf_counter();ranks=unpack(self.llm.collective_rpc(rpc.finish_probe));export_seconds=time.perf_counter()-export_start
        feature_start=time.perf_counter();features,costs,layers=resident_features(ranks,self.sample,required)
        feature_seconds=time.perf_counter()-feature_start
        ds=self.llm.collective_rpc(drain);verify_diagnostics(ds,self.sample,'prophetkv',.01,4,range(64))
        retirement_start=time.perf_counter();retired=verify_retired(self.llm,4,rid,self.scheduler);self.pc.unchanged();clean_capture(self.llm)
        retirement_seconds=time.perf_counter()-retirement_start
        record=self.common('study-probe',dict(probe_ttft_seconds=ttft,probe_generation_seconds=elapsed,
            feature_seconds=feature_seconds,export_rpc_seconds=export_seconds,retirement_seconds=retirement_seconds,
            total_seconds=time.perf_counter()-start,**costs),retired)
        record.update(binding=binding,features=features,internal_tokens=1,rank_costs=[r['timings'] for r in ranks],
            scratch_bounds=[r['scratch_bound_bytes'] for r in ranks])
        return record,ds,ranks,layers

    def routed(self,policy,label):
        from run import generate,verify_diagnostics
        from runner.worker import drain,router_dense_receipt
        from runner.router_measure import verify_retired
        from runner.reporting import evaluation_metadata,score_answer,scoring_text
        from runner.layout import validate_policy_protocol
        validate_policy_protocol(policy,self.sample)
        started=time.perf_counter()
        probe,probe_ds,ranks,layers=self.probe_study(policy['feature_names'])
        decision_start=time.perf_counter();action=decide(policy,probe['features']);decision_seconds=time.perf_counter()-decision_start
        definition=next(a for a in self.protocol['actions'] if a['id']==action);dense=action=='nocache'
        rid=self.rid('read:'+label+'-'+uuid.uuid4().hex)+('|router-dense' if dense else '')
        self.llm.collective_rpc(rpc.handoff,kwargs=dict(binding=probe['binding'],target=rid,dense=dense))
        sync=arm_tree(self.llm,self.sample,rid,definition)
        routing_seconds=time.perf_counter()-started
        output,ttft,elapsed=generate(self.llm.llm_engine,self.sample['token_ids'],self.sample['max_output_tokens'],rid,False,sample=self.sample)
        ds=self.llm.collective_rpc(drain);verify_diagnostics(ds,self.sample,definition['method'],definition['ratio'],4,range(64))
        counts=self.llm.collective_rpc(rpc.completed,kwargs=dict(dense=dense))
        audits=self.llm.collective_rpc(router_dense_receipt) if dense else []
        if dense and (output.num_cached_tokens!=0 or len(audits)!=4 or any(a['store_operations']!=dict(lookup=0,load=0,store=0) or len(a['native_layers'])!=64 for a in audits)):
            raise ValueError('Dense bypass gate failed')
        retired=verify_retired(self.llm,4,rid,self.scheduler);self.pc.unchanged();clean_capture(self.llm)
        tokens=list(output.outputs[0].token_ids);metrics=self.analyzer.analyze(tokens,self.sample['token_ids'],False);evaluation=evaluation_metadata({},self.row)
        result=self.common(label,dict(ttft_seconds=routing_seconds+ttft,answer_engine_ttft_seconds=ttft,
            routing_seconds=routing_seconds,decision_seconds=decision_seconds,generation_seconds=elapsed),retired)
        result.update(executed_action=action,output_token_ids=tokens,prediction=output.outputs[0].text,num_cached_tokens=output.num_cached_tokens,
            output_cap_reached=len(tokens)>=self.sample['max_output_tokens'],dense_receipts=audits,decision_sync=sync,**metrics,**evaluation,
            accuracy=score_answer(scoring_text(output.outputs[0].text,metrics,evaluation),evaluation),probe=probe,policy_sha256=policy['sha256'],one_probe_audit=counts)
        return result,ds,probe_ds,ranks


def save_arrays(folder,ranks,sample):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);artifacts=[]
    for r in ranks:
        path=folder/f'attention.rank{r["rank"]}.npz';start=time.perf_counter()
        # Only compact sufficient statistics accompany the native-score provenance.
        np.savez_compressed(path,layers=r['layers'],scores=r['scores'],local_mean=r['local_mean'])
        artifacts.append(dict(rank=r['rank'],path=path.name,sha256=file_hash(path),serialization_seconds=time.perf_counter()-start))
    return artifacts
