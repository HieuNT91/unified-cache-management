"""Kernel routing and optional CPU Triton-interpreter numerical checks."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
import torch._dynamo
from ucm.sparse.prophetkv import attention

ROOT = Path(__file__).resolve().parents[1]


class AttentionDispatchTests(unittest.TestCase):
    def test_default_suffix_still_matches_original_position_reference(self):
        torch.manual_seed(7)
        q=torch.randn(3,4,16);k=torch.randn(7,2,16);v=torch.randn_like(k);positions=torch.arange(4,7)
        actual=attention.suffix_attention(NS(),q,k,v,positions)
        torch.testing.assert_close(actual,attention.dense_reference(q,k,v,positions),atol=1e-6,rtol=1e-5)

    def test_new_suffix_forwards_original_positions_and_gqa_without_rewriting_tokens(self):
        kernel=Mock(side_effect=lambda q,k,v,o,*a,**kw:o.fill_(3))
        module=NS(ragged_positions_attention_fwd=kernel)
        q=torch.zeros(3,4,16);k=torch.zeros(7,2,16);v=torch.zeros_like(k);positions=torch.arange(4,7)
        with patch.dict(sys.modules,{'ucm.sparse.prophetkv.causal_fa2_kernel':module}):
            out=attention.suffix_attention(NS(position_attention_kernel='_ragged_positions_fa2_kernel'),q,k,v,positions)
        args=kernel.call_args.args
        self.assertIs(args[0],q);self.assertIs(args[1],k);self.assertIs(args[2],v)
        self.assertEqual(args[5].tolist(),[3]);self.assertEqual(args[7].tolist(),[7])
        self.assertEqual(args[8].tolist(),[4,5,6]);self.assertTrue(kernel.call_args.kwargs['is_causal'])
        self.assertTrue((out==3).all())
        with self.assertRaises(ValueError):attention.positional_kernel(NS(position_attention_kernel='unknown'))

    def test_fusion_uses_fa2_for_sparse_and_contiguous_queries_decode_stays_native(self):
        for positions in ([0,3,6],[4,5,6]):
            with self.subTest(positions=positions):
                native=Mock(); impl=type('FlashAttentionImpl',(),{})()
                impl.forward=native;impl.kv_cache_dtype='auto';impl.kv_sharing_target_layer_name=None
                impl.sliding_window=(-1,-1);impl.logits_soft_cap=0;impl.scale=.25
                layer=NS(impl=impl,_k_scale=1.,_v_scale=1.)
                sparse=NS(active=True,position_attention_kernel='_ragged_positions_fa2_kernel',audit=False,
                    fusion_positions=torch.tensor(positions),fusion_slots=torch.arange(7),block_size=8,
                    fusion_zero=torch.tensor([0],dtype=torch.int32),fusion_q_len=torch.tensor([3]),
                    fusion_k_len=torch.tensor([7]),connector=NS())
                kernel=Mock();reshape=Mock();attention.install(layer,sparse)
                mods={'ucm.sparse.prophetkv.causal_fa2_kernel':NS(ragged_positions_attention_fwd=kernel),
                      'vllm.v1.attention.backends.flash_attn':NS(reshape_and_cache_flash=reshape)}
                q=torch.zeros(3,4,16);kv=torch.zeros(2,1,8,2,16);out=torch.empty_like(q)
                with patch.dict(sys.modules,mods):
                    impl.forward(layer,q,q,q,kv,NS(use_cascade=False,slot_mapping=torch.tensor(positions)),out)
                    native.assert_not_called();reshape.assert_called_once();kernel.assert_called_once()
                    self.assertEqual(kernel.call_args.args[8].tolist(),positions)
                    sparse.active=False
                    impl.forward(layer,q,q,q,kv,None,out)
                    native.assert_called_once();kernel.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec('triton'), 'CPU interpreter requires Triton')
    def test_supplied_kernel_cpu_interpreter_causal_gqa_and_partial_tiles(self):
        import numpy as np
        import triton
        if triton.__version__ == '3.3.0' and np.lib.NumpyVersion(np.__version__) >= '2.4.0':
            self.skipTest('Triton 3.3 interpreter scalar conversion requires NumPy < 2.4')
        script = '''
import torch
from ucm.sparse.prophetkv.causal_fa2_kernel import ragged_positions_attention_fwd
from ucm.sparse.prophetkv.attention import dense_reference
torch.manual_seed(7)
for n,m,d in [(7,3,16),(129,65,32),(145,63,80)]:
    for suffix in (False,True):
        positions=torch.arange(n-m,n) if suffix else torch.linspace(0,n-1,m).long()
        q=torch.randn(m,4,d);k=torch.randn(n,2,d);v=torch.randn_like(k);out=torch.empty_like(q)
        zero=torch.tensor([0],dtype=torch.int32)
        ragged_positions_attention_fwd(q,k,v,out,zero,torch.tensor([m],dtype=torch.int32),
            zero,torch.tensor([n],dtype=torch.int32),positions,m)
        torch.testing.assert_close(out,dense_reference(q,k,v,positions),atol=2e-5,rtol=2e-4)
print('six FP32 causal/GQA/partial-tile interpreter cases passed; CPU only')
'''
        result=subprocess.run([sys.executable,'-c',script],cwd=ROOT,capture_output=True,text=True,timeout=60,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TRITON_INTERPRET='1',PYTHONPATH=str(ROOT)))
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__': unittest.main()
