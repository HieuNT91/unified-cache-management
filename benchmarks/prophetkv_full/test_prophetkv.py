"""Independent references and regressions for ProphetKV selection/causality."""
import json
import os
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'method'))
from prophetkv.selection import RequestMetadata,RequestState,LayerAlignment,context_importance,request_mask,select
from prophetkv.attention import dense_reference,is_contiguous_suffix


def reference_scores(q,k):
    repeated=k.float().repeat_interleave(q.shape[1]//k.shape[1],dim=1)
    logits=torch.einsum('qhd,khd->hqk',q.float(),repeated)/(q.shape[-1]**.5)
    return logits.softmax(-1).mean((0,1))

class Selection(unittest.TestCase):
    def test_native_backend_only_for_original_contiguous_suffix(self):
        self.assertTrue(is_contiguous_suffix(torch.tensor([5,6,7]),8))
        self.assertTrue(is_contiguous_suffix(torch.arange(8),8))
        self.assertFalse(is_contiguous_suffix(torch.tensor([1,4,7]),8))
        self.assertFalse(is_contiguous_suffix(torch.tensor([4,6,7]),8))

    def test_model_audit_artifact_keeps_tensor_payload(self):
        from types import SimpleNamespace
        import tempfile
        from cacheblend_prophetkv import audit_capture
        class Layer(torch.nn.Module):
            def forward(self,x):return x,x/2
        layers=torch.nn.ModuleList([Layer() for _ in range(36)])
        worker=SimpleNamespace(model_runner=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=layers))))
        audit_capture(worker,True)
        for layer in layers:layer(torch.ones(300,8))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audit.pt'
            receipt=audit_capture(worker,False,str(path))
            saved=torch.load(path,weights_only=True)[0]
            self.assertEqual(receipt['layers'],36)
            self.assertTrue(all(isinstance(v,torch.Tensor) for v in saved.values()))
            torch.testing.assert_close(saved[0],torch.full((256,8),1.5))
            self.assertTrue(all(not isinstance(v,torch.Tensor) for v in receipt.values()))

    def test_request_isolation(self):
        state=RequestState();meta=RequestMetadata('warmup',(0,512,1024,1280),(1100,))
        state.arm(meta)
        with self.assertRaises(RuntimeError):state.arm(meta)
        with self.assertRaises(RuntimeError):state.take('measured')
        self.assertEqual(state.take('warmup'),meta)
        with self.assertRaises(RuntimeError):state.take('measured')
        measured=RequestMetadata('measured',(0,512,1024,1280),(1101,))
        state.arm(measured);self.assertEqual(state.take('measured'),measured)

    def test_alignment_exactly_once_per_request(self):
        guard=LayerAlignment();key=torch.tensor([1.])
        def rotate(name):key.mul_(2)
        guard.apply('layer0',rotate);guard.apply('layer0',rotate)
        self.assertEqual(key.item(),2)
        guard.reset();guard.apply('layer0',rotate)
        self.assertEqual(key.item(),4)

    def test_tiled_gqa_softmax(self):
        torch.manual_seed(5)
        q=torch.randn(19,8,16);k=torch.randn(131,2,16)
        actual=context_importance(q,k,key_tile=23,query_tile=7)
        torch.testing.assert_close(actual,reference_scores(q,k),atol=1e-7,rtol=1e-5)
        self.assertAlmostEqual(float(actual.sum()),1.,places=6)

    def test_context_denominator_includes_prefix(self):
        q=torch.ones(1,2,1);k=torch.tensor([100.,0.,0.])[:,None,None]
        s=context_importance(q,k)
        self.assertLess(float(s[1:].sum()),1e-30)

    def test_budget_and_ties(self):
        scores=torch.ones(11)
        for rate in (0,.1,.2,.5,.9,1):
            self.assertEqual(select(scores,512,rate).tolist(),list(range(512,512+int(11*rate))))
        with self.assertRaises(ValueError):select(torch.tensor([float('nan')]),0,.2)

    def test_variable_query_suffix(self):
        meta=RequestMetadata('long-query',(0,4096,8192,9216),(8200,9100))
        self.assertEqual(meta.boundaries[-1]-meta.boundaries[-2],1024)
        with self.assertRaises(ValueError):RequestMetadata('short',(0,4096,8192,8256),(8200,))

    def test_longbench_official_answer_extraction(self):
        from prophetkv_common import extract_answer,score
        self.assertEqual(extract_answer('**The correct answer is (B)**'), 'B')
        self.assertEqual(extract_answer('The correct answer is C'), 'C')
        self.assertIsNone(extract_answer('Maybe A or B'))
        self.assertEqual(score({'dataset':'longbench_v2','source_metadata':{'answer':'D'}},'The correct answer is (D)')[0],1)

    def test_metadata(self):
        m=RequestMetadata('request',(0,512,1024,1280),(1100,1101))
        self.assertEqual(m.boundaries[-1],1280)
        for b,q in [((0,513,1024,1280),(1100,)),((0,512,1024,1280),(1000,)),((0,512,1024,1280),(1101,1100))]:
            with self.assertRaises(ValueError):RequestMetadata('a',b,q)

    def test_misses_and_decode_fresh(self):
        positions=torch.arange(64,256)
        mask=request_mask(positions,torch.tensor([64]),64,192,torch.tensor([1,0]))
        self.assertEqual(positions[mask].tolist(),[64]+list(range(128,256)))
        self.assertTrue(request_mask(torch.tensor([260]),torch.tensor([],dtype=torch.long),64,192,torch.tensor([True,True])).all())

    def test_compaction_counterexample(self):
        q=torch.zeros(3,4,8);k=torch.zeros(8,2,8)
        v=torch.arange(8).float()[:,None,None].expand(8,2,8)
        a=dense_reference(q,k,v,torch.tensor([1,4,7]))
        wrong=dense_reference(q,k,v,torch.tensor([5,6,7]))
        torch.testing.assert_close(a[:,0,0],torch.tensor([.5,2.,3.5]))
        self.assertGreater((a-wrong).abs().max(),1)

    def test_future_k_and_v_invariance(self):
        torch.manual_seed(42)
        q=torch.randn(3,8,8);k=torch.randn(40,2,8);v=torch.randn_like(k)
        pos=torch.tensor([0,17,39]);a=dense_reference(q,k,v,pos)
        k[18:]*=100;v[18:]+=100
        b=dense_reference(q,k,v,pos)
        torch.testing.assert_close(a[:2],b[:2],atol=0,rtol=0)


def gpu_tests():
    from prophetkv_gpu import check,GPU_UUIDS
    gpu=int(os.environ['SELECTOR_PHYSICAL_GPU']);check(gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUIDS[gpu]:raise RuntimeError('GPU visibility mismatch')
    from prophetkv.causal_kernel import ragged_positions_attention_fwd
    # Independently compute delta rotation with the actual UCM kernel.
    if os.environ.get('SELECTOR_UCM_ROOT'):sys.path.insert(0,os.environ['SELECTOR_UCM_ROOT'])
    from ucm.sparse.blend.blockwise_rope import block_wise_rope_forward
    key=torch.randn(2,64,8,128,device='cuda',dtype=torch.bfloat16)
    original=key.clone();theta=torch.randn(1025,64,device='cuda')
    table=torch.cat((theta.cos(),theta.sin()),-1).to(torch.bfloat16)
    offsets=torch.tensor([0,1024],device='cuda');blocks=torch.arange(2,device='cuda')
    left,right=original.float().chunk(2,-1)
    cos,sin=table[offsets].float().chunk(2,-1)
    expected=torch.cat((left*cos[:,None,None,:]-right*sin[:,None,None,:],
                        right*cos[:,None,None,:]+left*sin[:,None,None,:]),-1)
    guard=LayerAlignment()
    operation=lambda name:block_wise_rope_forward(key,blocks,offsets,table)
    guard.apply('layer',operation);once=key.clone();guard.apply('layer',operation)
    torch.testing.assert_close(key,once,atol=0,rtol=0)
    torch.testing.assert_close(key.float(),expected,atol=.03,rtol=.03)
    records=[dict(rope_max_abs_error=float((key.float()-expected).abs().max()),double_rotation=False)]
    torch.manual_seed(55)
    for length in (131,8192,32768,65536):
        k=torch.randn(length,8,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        q=torch.randn(23,32,128,device='cuda',dtype=torch.bfloat16)
        actual=context_importance(q,k)
        reference=reference_scores(q,k)
        torch.testing.assert_close(actual,reference,atol=2e-6,rtol=.02)
        positions=torch.linspace(0,length-1,len(q),device='cuda').long()
        out=torch.empty_like(q);zero=torch.zeros(1,device='cuda',dtype=torch.int32)
        qlen=torch.tensor([len(q)],device='cuda',dtype=torch.int32)
        klen=torch.tensor([length],device='cuda',dtype=torch.int32)
        ragged_positions_attention_fwd(q,k,v,out,zero,qlen,zero,klen,positions,len(q))
        expected=dense_reference(q,k,v,positions)
        torch.testing.assert_close(out.float(),expected,atol=.03,rtol=.03)
        before=out.clone();cut=int(positions[10])+1;k[cut:]*=10;v[cut:]+=100
        ragged_positions_attention_fwd(q,k,v,out,zero,qlen,zero,klen,positions,len(q))
        torch.testing.assert_close(out[:11],before[:11],atol=0,rtol=0)
        records.append(dict(tokens=length,score_max_abs_error=float((actual-reference).abs().max()),
            causal_max_abs_error=float((before.float()-expected).abs().max()),future_kv_leakage=False))
    print(json.dumps(dict(complete=True,gpu_uuid=GPU_UUIDS[gpu],checks=records)),flush=True)

if __name__=='__main__':
    if '--gpu' in sys.argv:gpu_tests()
    else:unittest.main()
