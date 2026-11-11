"""Kernel routing and optional CPU Triton-interpreter numerical checks."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest

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
