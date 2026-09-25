import sys,unittest,types,math,importlib.util,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from prepare import H,D
sys.path.insert(0,str(H/'runtime'))
from common import load,CASES,model_limit
from validate import select
from reporting import summarize
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
                return torch.ones(4,1,2),torch.ones(4,1,2),torch.ones(4,1,2),torch.ones(4,2)
            def importance(q,k):
                i=project_indices[-1];importance_indices.append(i)
                return torch.full((128,),np.float32(1/(i+1)),dtype=torch.float32)
            def global_selection(scores,prefix,ratio):
                saved['scores']=scores.clone();return scores[prefix:],selection.select(scores[prefix:],prefix,ratio)
            connector=types.SimpleNamespace(prophet_aligned=set())
            def wait(name):
                if name not in connector.prophet_aligned:waits.append(name);connector.prophet_aligned.add(name)
            connector.wait_for_layer_load=wait
            sparse=types.SimpleNamespace(method=method,model=types.SimpleNamespace(layers=layers),ratio=.2,
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
            for case in [*CASES,'prophetkv-20']:
                cfg=worker.build(types.SimpleNamespace(case=case,cache_dir=Path('/tmp/test'),smoke=False,persistent=True),sample,p)
                self.assertEqual(cfg['num_gpu_blocks_override'],1031);self.assertEqual(cfg['max_model_len'],65920)
                self.assertFalse(cfg['enable_prefix_caching'])
                if case=='baseline':self.assertNotIn('kv_transfer_config',cfg)
                else:self.assertEqual(cfg['kv_transfer_config']['kv_connector_extra_config']['ucm_sparse_config']['ProphetKV']['method'],case.rsplit('-',1)[0])
    def test_cohort_and_report(self):
        p=load(D/'protocol.json');self.assertEqual(len(p['samples'])*len(CASES),360)
        self.assertEqual(p['max_output_tokens'],256);self.assertEqual(p['samples_per_task_length'],5)
        self.assertEqual({m['label'] for m in p['samples']},{m['task'] for m in p['scope']})
        self.assertEqual([m['source_row'] for m in p['samples']],sum(([i]*8 for i in range(5)),[]))
        records=[dict(sample_id=m['id'],label=m['label'],case=c,ttft_seconds=2 if c=='baseline' else 1,
                      score=.5,output_tokens=256,finish_reason='length') for c in CASES for m in p['samples']]
        rows,pairs=summarize(records,CASES,list(dict.fromkeys(m['label'] for m in p['samples'])))
        self.assertEqual(len(rows),81);self.assertEqual(len(pairs),360)
        self.assertEqual(rows[-1]['baseline_to_method_mean_ttft_speedup'],2)
        self.assertEqual(rows[-1]['output_cap_count'],40)
    def test_report_gate(self):
        import reporting
        with patch.object(reporting,'load',return_value={'state':'running'}):
            with self.assertRaises(AssertionError):reporting.finalize()

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():sys.exit(1)
    from common import dump
    dump(D/'cpu-validation.json',dict(complete=True,tests=result.testsRun))
