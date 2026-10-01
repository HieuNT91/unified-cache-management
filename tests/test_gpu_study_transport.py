"""Use the actual installed vLLM wire codec, without CUDA or inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
import unittest
import numpy as np
from runner.gpu_study_transport import pack,unpack
class TransportTests(unittest.TestCase):
    def test_actual_untyped_vllm_rpc_codec(self):
        from vllm.v1.serial_utils import MsgpackEncoder,MsgpackDecoder
        source=[dict(rank=r,layers=np.arange(6400,dtype=np.float32).reshape(64,100),references=[(7,np.ones((2,3,4),np.float32))]) for r in range(4)]
        result=unpack(MsgpackDecoder().decode(MsgpackEncoder().encode(pack(source))))
        for a,b in zip(source,result):
            np.testing.assert_array_equal(a['layers'],b['layers']);np.testing.assert_array_equal(a['references'][0][1],b['references'][0][1])
            self.assertEqual(a['rank'],b['rank'])
    def test_truncation_and_type_rejection(self):
        packet=pack(np.ones(100,np.float32));packet['data']=packet['data'][:-1]
        with self.assertRaises(ValueError):unpack(packet)
        with self.assertRaises(ValueError):pack(np.ones(100,np.float64))
if __name__=='__main__':unittest.main()
