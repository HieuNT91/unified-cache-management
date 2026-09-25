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
from query_config import configuration,positions,oracle_positions,STAGES,TASKS,TARGETS,COUNTS,ORACLE_CASES,HIGH_CASES
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
        for method,expected,mode in [(method,layers,mode) for method,layers in [('selective_prophetkv',[45,48,50,56,58]),('prophetkv',list(range(64)))] for mode in ['prophetkv','target_union']]:
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
            sparse=types.SimpleNamespace(method=method,selection_mode=mode,model=types.SimpleNamespace(layers=layers),ratio=.2,
                query_scope='focus',layer_scope='all64' if method=='prophetkv' else 'selected5',
                request=types.SimpleNamespace(boundaries=(0,64,128,132),question_positions=(129,),request_id='test',oracle_positions=(127,) if mode=='target_union' else ()),
                attn_metadata=types.SimpleNamespace(block_table=torch.tensor([[0,1,2]])),connector=connector,selection_diagnostics=[])
            context=types.SimpleNamespace(virtual_engine=0,no_compile_layers={f'model.layers.{i}.self_attn.attn':types.SimpleNamespace(kv_cache=[torch.ones(2,3,64,1,2)]) for i in range(64)})
            fake=types.ModuleType('vllm.forward_context');fake.get_forward_context=lambda:context
            with patch.dict(sys.modules,{'vllm.forward_context':fake}),patch.object(runtime,'project',project),patch.object(runtime,'context_importance',importance),patch.object(runtime,'global_selection',global_selection),patch.object(torch.cuda,'synchronize',lambda:None):
                chosen=runtime.probe(sparse,torch.arange(64,132),torch.ones(68,2))
            native=selection.select(saved['scores'][64:],64,.2)
            expected_mask=selection.combine_positions(native,torch.tensor([127] if mode=='target_union' else [],dtype=torch.long),mode)
            self.assertTrue(torch.equal(chosen,expected_mask))
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
                    for key in ['method','query_scope','layer_scope','selection_mode']:self.assertEqual(sparse[key],configuration(case)[key])
    def test_union_reference_and_actual_budgets(self):
        from validate import expected_mask
        rng=np.random.default_rng(3)
        for n in [7,51,60800]:
            scores=rng.integers(0,8,size=n).astype(np.float32);oracle=sorted(set([64,64+n//2,63+n]))
            m=dict(boundaries=[0,64,64+n,320+n],oracle={'eligible_positions':oracle})
            for case in CASES:
                cfg=configuration(case);native,chosen=expected_mask(scores,m,case)
                expected_native=[] if cfg['selection_mode']=='target_only' else sorted(64+i for i in sorted(range(n),key=lambda i:(-scores[i],i))[:math.floor(n*cfg['budget']/100)])
                expected_oracle=[] if cfg['selection_mode']=='prophetkv' else oracle
                self.assertEqual(chosen.tolist(),sorted(set(expected_native)|set(expected_oracle)))
                actual=selection.combine_positions(torch.tensor(native,dtype=torch.long),torch.tensor(expected_oracle,dtype=torch.long),cfg['selection_mode'])
                self.assertEqual(actual.tolist(),chosen.tolist())
                self.assertEqual(len(native),len(expected_native));self.assertLessEqual(len(chosen),n)
        with self.assertRaises(ValueError):selection.combine_positions(torch.tensor([1]),torch.tensor([2]),'prophetkv')
        self.assertEqual(selection.combine_positions(torch.tensor([4,5]),torch.tensor([5,9]),'target_union').tolist(),[4,5,9])
    def test_target_only_skips_scoring_and_aligns(self):
        for oracle in [(),(70,100)]:
            connector=types.SimpleNamespace(prophet_aligned=set())
            connector.wait_for_layer_load=lambda name:connector.prophet_aligned.add(name)
            sparse=types.SimpleNamespace(model=types.SimpleNamespace(layers=[None]*64),selection_mode='target_only',query_scope='full_question',layer_scope='all64',connector=connector,
                request=types.SimpleNamespace(boundaries=(0,64,128,132),question_positions=(129,),oracle_positions=oracle,request_id='only'),selection_diagnostics=[])
            fake=types.ModuleType('vllm.forward_context');fake.get_forward_context=lambda:types.SimpleNamespace()
            with patch.dict(sys.modules,{'vllm.forward_context':fake}),patch.object(torch.cuda,'synchronize',lambda:None),patch.object(runtime,'project',side_effect=AssertionError('Unexpected projection')),patch.object(runtime,'context_importance',side_effect=AssertionError('Unexpected scoring')):
                chosen=runtime.probe(sparse,torch.arange(64,132),torch.ones(68,2))
            self.assertEqual(chosen.tolist(),list(oracle));self.assertEqual(len(connector.prophet_aligned),64)
            self.assertEqual(sparse.selection_diagnostics[0]['probe_layers'],0)
            self.assertEqual(sparse.selection_diagnostics[0]['scores'].numel(),0)
    def test_metadata_and_oracle_leakage(self):
        metadata=selection.RequestMetadata('r',(0,64,128,384),(200,),(64,100))
        self.assertEqual(metadata.oracle_positions,(64,100))
        for oracle in [(63,),(128,),(70,70),(80,70)]:
            with self.assertRaises(ValueError):selection.RequestMetadata('r',(0,64,128,384),(200,),oracle)
        m=dict(query={'positions':[200]},oracle={'eligible_positions':[64]})
        for c in HIGH_CASES:self.assertEqual(oracle_positions(m,c),[]);self.assertEqual(positions(m,c),[200])
    def test_exact_frozen_cohorts_and_oracle_spans(self):
        from prepare import sha,annotate
        sources={};tok=AutoTokenizer.from_pretrained(load(D/'oracle/protocol.json')['model'],local_files_only=True)
        for stage in STAGES:
            p=load(D/stage/'protocol.json');schedule=load(D/stage/'schedule.json')
            self.assertEqual(len(schedule),TARGETS[stage]);self.assertEqual(len(p['samples']),COUNTS[stage]*3)
            self.assertEqual([m['label'] for m in p['samples']],list(TASKS[stage])*COUNTS[stage])
            for m in p['samples']:
                self.assertEqual(sha(m['input_path']),m['input_sha256']);self.assertEqual(sha(m['original_input_path']),m['input_sha256'])
                sample=load(m['input_path']);self.assertLessEqual(sample['tokens']+256,65920)
                if stage=='highbudget':self.assertNotIn('oracle',m);continue
                if m['source_path'] not in sources:sources[m['source_path']]=[json.loads(x) for x in Path(m['source_path']).read_text().splitlines()]
                a=annotate(tok,m,sample,sources[m['source_path']][m['source_row']]);self.assertEqual(a,m['oracle'])
                self.assertTrue(a['mapping_verified'])
                self.assertEqual(sorted(a['eligible_positions']+a['already_exact_prefix_positions']),a['all_positions'])
                for span in a['spans']:self.assertEqual([sample['token_ids'][i] for i in span['positions']],span['token_ids'])
            if stage=='oracle':
                prefix=[m for m in p['samples'] if m['oracle']['already_exact_prefix_positions']]
                self.assertEqual([m['id'] for m in prefix],['niah_multikey_2-65536-0008'])
                self.assertEqual(len(prefix[0]['oracle']['already_exact_prefix_positions']),10)
                self.assertEqual(prefix[0]['oracle']['eligible_positions'],[])
                self.assertNotIn('cwe',TASKS[stage])
        self.assertEqual(list(dict.fromkeys(configuration(c)['budget'] for c in HIGH_CASES)),[90,80,95])
        self.assertEqual(sum(TARGETS.values()),1230)
    def test_priming_observer_modes(self):
        import tempfile,prime_compare
        for mode,scope in [('target_only','all64'),('target_union','all64'),('target_union','selected5'),('prophetkv','selected5')]:
            indices=[] if mode=='target_only' else list(range(64)) if scope=='all64' else [45,48,50,56,58]
            prior=np.full((64,8),1/8,dtype=np.float32);value=torch.full((8,),1/8)
            rt=types.SimpleNamespace(context_importance=lambda *args:value.clone())
            native=dict(kind='prophetkv_selection',scores=torch.ones(4),selected_positions=torch.tensor([4,5]),base_selected_positions=torch.tensor([4]),question_positions=[9],prefix_tokens=4,eligible_count=4)
            mod=types.ModuleType('ucm.sparse.prophetkv');mod.runtime=rt
            state=types.ModuleType('ucm.sparse.state');state.get_ucm_sparse=lambda:types.SimpleNamespace(selection_diagnostics=[native])
            dist=types.ModuleType('vllm.distributed');dist.get_tensor_model_parallel_rank=lambda:0
            with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'ucm.sparse.prophetkv':mod,'ucm.sparse.state':state,'vllm.distributed':dist}):
                np.savez(Path(tmp)/'prior.rank0.npz',layers=prior);worker=types.SimpleNamespace();original=rt.context_importance
                prime_compare.begin_capture(worker)
                for _ in indices:rt.context_importance()
                result=prime_compare.end_capture(worker,str(Path(tmp)/'prior'),str(Path(tmp)/'prime'),mode,scope)
                self.assertIs(rt.context_importance,original);self.assertTrue(result['observer_removed']);self.assertEqual(result['layers'],indices)
                self.assertEqual(result['historical_match_required'],bool(indices))
    def test_phase_handoff_gate(self):
        import tempfile,run
        from common import dump
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'D',Path(tmp)),patch.object(run,'group_alive',return_value=False):
            root=Path(tmp)/'oracle';root.mkdir();self.assertFalse(run.phase_ready('oracle'))
            dump(root/'final-validation.json',dict(complete=True,validated=329,artifact_sha256={}))
            with self.assertRaises(AssertionError):run.phase_ready('oracle')
            dump(root/'final-validation.json',dict(complete=True,validated=330,artifact_sha256={}))
            dump(root/'cleanup.json',dict(complete=True));dump(root/'stage.json',dict(last_engine_pid=1))
            self.assertTrue(run.phase_ready('oracle'))
            with patch.object(run,'group_alive',return_value=True):
                with self.assertRaises(AssertionError):run.phase_ready('oracle')
    def test_reports_and_completeness(self):
        import tempfile,reporting
        from prepare import sha
        from common import dump
        details=[]
        for stage in STAGES:
            p=load(D/stage/'protocol.json');records=[]
            for m in p['samples']:
                for case in p['cases']:
                    cfg=configuration(case);eligible=m['boundaries'][-2]-m['boundaries'][1];only=cfg['selection_mode']=='target_only'
                    oracle=len(oracle_positions(m,case));native=math.floor(eligible*cfg['budget']/100);chosen=oracle if only else min(eligible,native+oracle)
                    r=dict(sample_id=m['id'],label=m['label'],case=case,selection_mode=cfg['selection_mode'],layer_scope=cfg['layer_scope'],budget=cfg['budget'],
                        ttft_seconds=2,generation_seconds=3,score=.5,output_tokens=256,output_token_ids=[1]*256,finish_reason='length',prompt_tokens=m['tokens'],
                        prediction='synthetic',references=['synthetic'],selected_context_tokens=chosen,eligible_context_tokens=eligible,effective_recompute_ratio=chosen/eligible,
                        native_selected_tokens=native,oracle_eligible_tokens=oracle,oracle_added_tokens=chosen-native,oracle_exact_prefix_tokens=len(m.get('oracle',{}).get('already_exact_prefix_positions',[])))
                    records.append(r)
            with self.assertRaises(AssertionError):reporting.check_complete(records[:-1],p)
            with self.assertRaises(AssertionError):reporting.check_complete(records[:-1]+[records[0]],p)
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);summary,pairs=reporting.publish(out,records,p)
                self.assertEqual(len(summary),len(p['cases'])*4)
                self.assertEqual(len(pairs),450)
                self.assertEqual(summary[-1]['output_cap_count'],len(p['samples']))
                self.assertEqual(len(list(out.glob('*.png'))),1);self.assertEqual(len(list(out.glob('*.pdf'))),1)
                self.assertTrue(all(r['accuracy_difference_pp']==0 for r in pairs))
                details.append(dict(stage=stage,records=len(records),paired=len(pairs),artifacts={f.name:sha(f) for f in out.iterdir() if f.is_file()}))
        dump(D/'cpu-report-validation.json',dict(complete=True,stages=details))
    def test_report_live_engine_gate(self):
        import reporting
        with patch.object(reporting,'load',return_value={'state':'running'}):
            with self.assertRaises(AssertionError):reporting.finalize(D/'oracle')

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():sys.exit(1)
    from common import dump
    dump(D/'cpu-validation.json',dict(complete=True,tests=result.testsRun))
