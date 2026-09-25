import sys,unittest,types,math,importlib.util,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from transformers import AutoTokenizer
from prepare import H,D
sys.path.insert(0,str(H/'runtime'))
from common import load,CASES,model_limit
from validate import select
from reporting import summarize
from query_config import configuration,positions,schedule,TASKS,BUDGETS
# Load primitives without importing the GPU runtime package initializer.
pkg=types.ModuleType('selective_test');pkg.__path__=[str(H/'private_ucm/ucm/sparse/prophetkv')];sys.modules[pkg.__name__]=pkg
from selective_test import selection,runtime
from rope_window import ensure_rope_window

class Tests(unittest.TestCase):
    def test_exact_budget_ties(self):
        scores=torch.tensor([.5,.9,.9,.5,.9,.1])
        for ratio in [0,.01,.05,.1,.2,.3,.4,.5,.6,1]:
            expected=select(scores.numpy(),ratio,4096)
            self.assertTrue(np.array_equal(selection.select(scores,4096,ratio),expected))
            self.assertEqual(len(expected),math.floor(len(scores)*ratio))
        self.assertEqual(selection.select(scores,10,.5).tolist(),[11,12,14])
        with self.assertRaises(ValueError):selection.select(torch.tensor([float('nan')]),0,.2)
    def test_all_context_normalization(self):
        q=torch.ones(2,2,4);k=torch.ones(10,1,4)
        scores=selection.context_importance(q,k)
        torch.testing.assert_close(scores,torch.full((10,),.1))
    def test_request_alignment_and_fresh(self):
        calls=[];guard=selection.LayerAlignment()
        for _ in range(2):
            for i in range(64):guard.apply(str(i),calls.append)
        self.assertEqual(len(calls),64)
        guard.reset();guard.apply('0',calls.append);self.assertEqual(len(calls),65)
        pos=torch.arange(64,256);picked=torch.tensor([66,70])
        mask=selection.request_mask(pos,picked,64,192,torch.ones(2))
        self.assertEqual(pos[mask].tolist(),[66,70]+list(range(192,256)))
    def test_probe_dependencies_fp32_and_original(self):
        from contextlib import ExitStack
        for method,expected in [('selective_prophetkv',[45,48,50,56,58]),('prophetkv',list(range(64)))]:
            project_indices=[];importance_indices=[];forwards=[];waits=[];saved={}
            class Layer:
                def __init__(self,i):
                    self.i=i;self.self_attn=types.SimpleNamespace(o_proj=lambda x:(x,None))
                def post_attention_layernorm(self,h,r):return h,r
                def mlp(self,h):forwards.append(self.i);return h
            layers=[Layer(i) for i in range(64)]
            def project(layer,h,r,p):
                project_indices.append(layer.i)
                return torch.arange(8,dtype=torch.float32).reshape(4,1,2),torch.ones(4,1,2),torch.ones(4,1,2),torch.ones(4,2)
            def importance(q,k):
                i=project_indices[-1];importance_indices.append(i)
                self.assertEqual(q.tolist(),[[[2.,3.]]])
                return torch.full((128,),np.float32(1/(i+1)),dtype=torch.float32)
            def global_selection(scores,prefix,ratio):
                saved['scores']=scores.clone();return scores[prefix:],selection.select(scores[prefix:],prefix,ratio)
            connector=types.SimpleNamespace(prophet_aligned=set())
            def wait(name):
                if name not in connector.prophet_aligned:waits.append(name);connector.prophet_aligned.add(name)
            connector.wait_for_layer_load=wait
            sparse=types.SimpleNamespace(method=method,model=types.SimpleNamespace(layers=layers),ratio=.2,
                query_scope='focus',layer_scope='all64' if method=='prophetkv' else 'selected5',
                request=types.SimpleNamespace(boundaries=(0,64,128,132),question_positions=(129,),request_id='test'),
                attn_metadata=types.SimpleNamespace(block_table=torch.tensor([[0,1,2]])),connector=connector,selection_diagnostics=[])
            context=types.SimpleNamespace(virtual_engine=0,no_compile_layers={f'model.layers.{i}.self_attn.attn':types.SimpleNamespace(kv_cache=[torch.ones(2,3,64,1,2)]) for i in range(64)})
            fake=types.ModuleType('vllm.forward_context');fake.get_forward_context=lambda:context
            with patch.dict(sys.modules,{'vllm.forward_context':fake}),patch.object(runtime,'project',project),patch.object(runtime,'context_importance',importance),patch.object(runtime,'global_selection',global_selection),patch.object(torch.cuda,'synchronize',lambda:None):
                runtime.probe(sparse,torch.arange(64,132),torch.ones(68,2))
            self.assertEqual(importance_indices,expected)
            self.assertEqual(project_indices,list(range(expected[-1]+1)))
            self.assertEqual(forwards,list(range(58 if method=='selective_prophetkv' else 64)))
            self.assertEqual(len(waits),64)
            expected_sum=torch.zeros(128,dtype=torch.float32)
            for i in expected:expected_sum+=np.float32(1/(i+1))
            if method=='prophetkv':expected_sum/=64
            self.assertTrue(torch.equal(saved['scores'],expected_sum))
    def test_tp_average(self):
        fake=types.ModuleType('vllm.distributed');fake.tensor_model_parallel_all_reduce=lambda x:x*4
        fake.get_tp_group=lambda:types.SimpleNamespace(world_size=4,broadcast=lambda x,src:x)
        with patch.dict(sys.modules,{'vllm.distributed':fake}):
            scores,chosen=runtime.global_selection(torch.arange(10,dtype=torch.float32),2,.5)
        self.assertEqual(scores.tolist(),list(range(2,10)));self.assertEqual(chosen.tolist(),[6,7,8,9])
    def test_allocation_and_rope(self):
        self.assertEqual(model_limit({'samples':[{'context_target':65536,'tokens':65664}]},{'context_target':65536}),65920)
        self.assertEqual(65920//64+1,1031)
        with self.assertRaises(ValueError):model_limit({'samples':[{'context_target':65536,'tokens':65665}]},{'context_target':65536})
        inv=torch.tensor([.5,.125]);pos=torch.arange(65536,dtype=torch.float32);f=torch.einsum('i,j->ij',pos,inv)
        rope=types.SimpleNamespace(scaling_factor=2.,max_position_embeddings=32768,mscale=1.1,_compute_inv_freq=lambda _:inv,
                                 cos_sin_cache=torch.cat((f.cos()*1.1,f.sin()*1.1),-1))
        old=rope.cos_sin_cache.clone();receipt=ensure_rope_window(rope,65920)
        self.assertTrue(torch.equal(old,rope.cos_sin_cache[:65536]));self.assertEqual(receipt['added_positions'],384)
    def test_configuration_routing(self):
        import worker
        vllm=types.ModuleType('vllm');vllm.LLM=lambda **kw:kw
        config=types.ModuleType('vllm.config');config.KVTransferConfig=lambda **kw:kw
        memory=types.ModuleType('memory_worker');memory.loader_options=lambda:{}
        sample=dict(context_target=65536,tokens=65664,token_ids=[0,99],boundaries=[0,2,4])
        p=dict(model='model',samples=[sample],gpu_memory_utilization=.95,tensor_parallel_size=4,rope_scaling_64k={'factor':2})
        with patch.dict(sys.modules,{'vllm':vllm,'vllm.config':config,'memory_worker':memory}):
            for case in ('baseline',*CASES):
                cfg=worker.build(types.SimpleNamespace(case=case,cache_dir=Path('/tmp/test'),smoke=False,persistent=True),sample,p)
                self.assertEqual(cfg['num_gpu_blocks_override'],1031);self.assertEqual(cfg['max_model_len'],65920)
                self.assertFalse(cfg['enable_prefix_caching'])
                if case=='baseline':self.assertNotIn('kv_transfer_config',cfg)
                else:
                    sparse=cfg['kv_transfer_config']['kv_connector_extra_config']['ucm_sparse_config']['ProphetKV']
                    for key in ['method','query_scope','layer_scope']:self.assertEqual(sparse[key],configuration(case)[key])
    def test_report_gate(self):
        import reporting
        with patch.object(reporting,'load',return_value={'state':'running'}):
            with self.assertRaises(AssertionError):reporting.finalize()

    def test_spans_immutable_and_suffix(self):
        from spans import extract,map_span
        from transformers import AutoTokenizer
        from prepare import sha,VANILLA
        p=load(D/'protocol.json');tok=AutoTokenizer.from_pretrained(p['model'],local_files_only=True)
        self.assertEqual(len(p['samples']),15)
        self.assertEqual([m['label'] for m in p['samples']],list(TASKS)*5)
        self.assertEqual([m['source_row'] for m in p['samples']],sum(([i]*3 for i in range(5)),[]))
        sources={}
        for m in p['samples']:
            sample=load(m['input_path']);focus=m['focus']
            self.assertEqual(sha(m['input_path']),m['input_sha256'])
            self.assertEqual(Path(m['input_path']).read_bytes(),(VANILLA/'inputs'/Path(m['input_path']).name).read_bytes())
            if m['source_path'] not in sources:sources[m['source_path']]=[json.loads(x) for x in Path(m['source_path']).read_text().splitlines()]
            row=sources[m['source_path']][m['source_row']]
            self.assertEqual(map_span(tok,m,sample,row),focus)
            # Reference changes cannot alter inference spans.
            self.assertEqual(map_span(tok,m,sample,row|dict(outputs=['WRONG'])),focus)
            self.assertTrue(set(focus['absolute_positions'])<=set(m['query']['positions']))
            self.assertTrue(min(focus['absolute_positions'])>=m['boundaries'][-2])
            self.assertEqual([sample['token_ids'][i] for i in focus['absolute_positions']],focus['token_ids'])
            if m['label']=='cwe':self.assertEqual(focus['span_text'],'10 most common words');self.assertEqual(len(focus['token_ids']),5)
            elif m['label'].endswith('_3'):
                self.assertEqual(focus['span_text'].count('-'),4)
                self.assertTrue(28<=len(focus['token_ids'])<=32)
                self.assertEqual(tok.decode(focus['token_ids']).strip(),focus['span_text'])
            else:self.assertTrue(3<=len(focus['token_ids'])<=6)
        with self.assertRaises(ValueError):extract('niah_multikey_3','What for 123 mentioned?')
        with self.assertRaises(ValueError):extract('cwe','10 most common words and 10 most common words')
    def test_schedule_reuse(self):
        ms=load(D/'protocol.json')['samples']
        retained=[dict(sample_id=m['id'],case=c) for m in ms for c in CASES if c.startswith('full_question-')]
        fresh=schedule(ms,retained)
        self.assertEqual(len(fresh),210)
        self.assertTrue(all(m['case'].startswith('focus-') for m in fresh))
        missing=retained.pop(0);fresh=schedule(ms,retained)
        self.assertEqual(len(fresh),211)
        self.assertIn((missing['sample_id'],missing['case']),[(m['id'],m['case']) for m in fresh])
        self.assertEqual([configuration(m['case'])['budget'] for m in fresh],sorted(configuration(m['case'])['budget'] for m in fresh))
        self.assertEqual(list(dict.fromkeys(m['case'] for m in fresh))[:3],['focus-all64-5','focus-selected5-5','full_question-all64-5'])
        for case in CASES:
            self.assertEqual(positions(ms[0],case),ms[0]['focus']['absolute_positions'] if case.startswith('focus-') else ms[0]['query']['positions'])
    def test_focus_query_rows_used(self):
        # The probe test above asserts native layer arithmetic; additionally verify exact query slicing.
        import inspect
        self.assertIn('context_importance(q[qp],ck)',inspect.getsource(runtime.probe))
        # Uniform keys retain normalization across *all* keys with either query length.
        for count in [1,5,32]:
            scores=selection.context_importance(torch.ones(count,2,4),torch.ones(13,1,4))
            torch.testing.assert_close(scores,torch.full((13,),1/13))
    def test_evidence_metrics(self):
        from diagnosis import evidence_metrics,overlap
        ann=dict(spans=[dict(kind=k,label=k,text=k,token_runs=[span]) for k,span in [('key',[5,7]),('answer',[9,11]),('evidence',[5,11])]])
        m=dict(evidence_path='mock',boundaries=[0,4,12,16],label='niah_multikey_2')
        with patch('diagnosis.load',return_value=ann):
            metrics=evidence_metrics(m,[5,6,9],np.ones(8)/12,'all64')
            self.assertEqual(metrics['key_recall'],1);self.assertEqual(metrics['answer_recall'],.5)
            self.assertAlmostEqual(metrics['harmonic_recall'],2/3);self.assertEqual(metrics['complete_statement_retention'],0)
            self.assertAlmostEqual(metrics['key_attention_mass'],1/6)
            scaled=evidence_metrics(m,[5,6,9],np.ones(8)*5/12,'selected5')
            self.assertAlmostEqual(scaled['key_attention_mass'],metrics['key_attention_mass'])
        self.assertEqual(overlap([1,2],[2,3])['mask_jaccard'],1/3)
    def test_priming_observer_and_historical_gate(self):
        import tempfile
        import prime_compare
        for query in ['focus','full_question']:
            for layer in ['all64','selected5']:
                indices=list(range(64)) if layer=='all64' else [45,48,50,56,58]
                prior=np.full((64,8),1/8,dtype=np.float32)
                value=torch.full((8,),1/8)
                if query=='focus':value[0]+=.01;value[1]-=.01
                rt=types.SimpleNamespace(context_importance=lambda *args:value.clone())
                native=dict(kind='prophetkv_selection',scores=torch.ones(4),selected_positions=torch.tensor([4]),question_positions=[9])
                mod=types.ModuleType('ucm.sparse.prophetkv');mod.runtime=rt
                state=types.ModuleType('ucm.sparse.state');state.get_ucm_sparse=lambda:types.SimpleNamespace(selection_diagnostics=[native])
                dist=types.ModuleType('vllm.distributed');dist.get_tensor_model_parallel_rank=lambda:0
                with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'ucm.sparse.prophetkv':mod,'ucm.sparse.state':state,'vllm.distributed':dist}):
                    np.savez(Path(tmp)/'prior.rank0.npz',layers=prior)
                    worker=types.SimpleNamespace();original=rt.context_importance
                    prime_compare.begin_capture(worker)
                    for _ in indices:rt.context_importance()
                    result=prime_compare.end_capture(worker,str(Path(tmp)/'prior'),str(Path(tmp)/'prime'),query,layer)
                    self.assertIs(rt.context_importance,original);self.assertTrue(result['observer_removed'])
                    self.assertEqual(result['historical_match_required'],query=='full_question')
                    self.assertEqual(result['prior_close'],query=='full_question')
                    self.assertFalse(hasattr(worker,'_prime_layers'))
                    # Deliberately mismatched full-question priming must fail, with the observer removed.
                    prime_compare.begin_capture(worker);worker._prime_layers=[value+1 for _ in indices]
                    with self.assertRaises(AssertionError):prime_compare.end_capture(worker,str(Path(tmp)/'prior'),str(Path(tmp)/'bad'),'full_question',layer)
                    self.assertIs(rt.context_importance,original)
    def test_full_report(self):
        import tempfile
        from reporting import publish,check_completeness
        from prepare import sha
        p=load(D/'protocol.json');records=[];diagnostics=[]
        for m in p['samples']:
            for case in ('baseline',*CASES):
                r=dict(sample_id=m['id'],label=m['label'],case=case,ttft_seconds=2 if case=='baseline' else 1,
                    score=.5,output_tokens=256,output_token_ids=[1]*256,finish_reason='length',prompt_tokens=m['tokens'],
                    measurement_origin='fresh' if case.startswith('focus-') else 'retained',timing_cohort='synthetic',
                    prediction='synthetic',references=['synthetic'])
                records.append(r)
                if case=='baseline':continue
                cfg=configuration(case)
                metrics={k:0.5 for k in ['key_recall','answer_recall','harmonic_recall','complete_statement_retention','evidence_recall','key_attention_mass','answer_attention_mass','evidence_attention_mass']}
                if m['label']=='cwe':
                    for k in ['key_recall','answer_recall','harmonic_recall','complete_statement_retention','key_attention_mass','answer_attention_mass']:metrics[k]=None
                diagnostics.append(dict(sample_id=m['id'],case=case,selected_positions=[4096,4097],**metrics))
        with self.assertRaises(AssertionError):check_completeness(records[:-1],p)
        with self.assertRaises(AssertionError):check_completeness(records[:-1]+[records[0]],p)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);summary,pairs=publish(out,records,p,diagnostics)
            self.assertEqual(len(summary),116);self.assertEqual(len(pairs),210)
            self.assertEqual(len(list((out/'masks').rglob('*.npy'))),420)
            self.assertEqual(len(list(out.glob('*.png'))),4);self.assertEqual(len(list(out.glob('*.pdf'))),4)
            self.assertTrue(all(r['accuracy_difference_pp']==0 for r in pairs))
            self.assertEqual(summary[-1]['baseline_to_method_mean_ttft_speedup'],2)
            self.assertEqual(summary[-1]['output_cap_count'],15)
            from common import dump
            dump(D/'cpu-report-validation.json',dict(complete=True,synthetic_records=435,method_records=420,paired=210,plots=8,
                artifacts={f.name:sha(f) for f in out.iterdir() if f.is_file()}))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():sys.exit(1)
    from common import dump
    dump(D/'cpu-validation.json',dict(complete=True,tests=result.testsRun))
