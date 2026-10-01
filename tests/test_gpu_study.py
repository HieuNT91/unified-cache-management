"""Formula, isolation, publication and bounded-statistic gates for the live study."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from runner.gpu_study_features import *
from runner.gpu_study_policy import Handoff,make,decide
from runner.gpu_study_worker import distributions,statistic_arrays,confidence_values,LIMIT
from runner.gpu_study import sealed
from runner.gpu_study_search import cost_estimate

class GPUStudyTests(unittest.TestCase):
    def test_inventory_and_sampling(self):
        self.assertEqual(len(ADDITIONS),76);self.assertEqual(len(set(REGISTERED)),81)
        self.assertEqual(len(HEAD),12);self.assertEqual(len(QUERY),12);self.assertEqual(len(ENTROPY),6);self.assertEqual(len(CONFIDENCE),6)
        for n in range(1,300):
            q=query_indices(n);self.assertEqual(len(q),min(16,n));self.assertEqual(q,sorted(set(q)))
            self.assertEqual(q[0],0);self.assertEqual(q[-1],n-1)
        with self.assertRaises(ValueError):query_indices(0)
    def test_full_context_gqa_reference(self):
        torch.manual_seed(7);q=torch.randn(19,4,8);k=torch.randn(101,2,8)
        h,z,bound=distributions(q,k,4)
        ref=torch.einsum('qhd,khd->qhk',q,k.repeat_interleave(2,dim=1))/8**.5
        weights=ref.softmax(-1)
        torch.testing.assert_close(h,weights[:,:,4:].mean(0),rtol=2e-6,atol=1e-7)
        torch.testing.assert_close(z,weights[query_indices(19),:,4:].mean(1),rtol=2e-6,atol=1e-7)
        self.assertLessEqual(bound,LIMIT)
        for head,query in ((True,False),(False,True)):
            a,b,_=distributions(q,k,4,head,query)
            if head:torch.testing.assert_close(a,h)
            if query:torch.testing.assert_close(b,z)
        with self.assertRaises(MemoryError):distributions(q,k,4,retained_bytes=LIMIT)
    def test_stats_ties_zero_and_entropy(self):
        p=torch.ones((8,1000));scores=torch.ones(1000)
        r=statistic_arrays(p,scores,GPU_FEATURES,'head')
        for percent in (1,5,10):np.testing.assert_allclose(r[str(percent)],percent/100,rtol=1e-6)
        np.testing.assert_allclose(r['entropy'],1,rtol=1e-6)
        z=statistic_arrays(torch.zeros_like(p),scores,GPU_FEATURES,'head')
        self.assertTrue(all(v is None for v in z['entropy']))
    def test_all_rank_summaries(self):
        ranks=[dict(rank=r,statistics={'head':{'1':[r+1,r+2],'5':[r+3],'10':[r+4],'entropy':[r/4]},
            'query':{'1':[.1,.2],'5':[.3],'10':[.4],'entropy':[.5]}},confidence=dict(zip(CONFIDENCE,[.5]*6))) for r in range(4)]
        full=reduce_statistics(ranks)
        self.assertEqual(full['head_coverage1_min'],1)
        self.assertEqual(full['head_coverage1_median'],3)
        self.assertAlmostEqual(full['query_coverage1_median'],.15)
        for feature in GPU_FEATURES:self.assertEqual(reduce_statistics(ranks,[feature]),{feature:full[feature]})
        with self.assertRaises(ValueError):reduce_statistics(ranks[:3])
        ranks[0]['statistics']['head']['1']=[None]
        self.assertIsNone(reduce_statistics(ranks)['head_coverage1_min'])
    def test_vocabulary_uniform_concentrated(self):
        r=confidence_values(torch.zeros(1,100))
        for k,v in dict(entropy=1,effective_support=1,top1=.01,margin=0,top5=.05,top20=.2).items():self.assertAlmostEqual(r['confidence_'+k],v,places=6)
        logits=torch.full((1,100),-1000.);logits[0,0]=0
        r=confidence_values(logits);self.assertEqual(r['confidence_entropy'],0)
        self.assertAlmostEqual(r['confidence_effective_support'],.01,places=7)
        self.assertEqual(r['confidence_top1'],1)
    def test_registered_policy_boundaries_fallback(self):
        f='head_coverage1_min';tree=dict(feature=f,threshold=.5,le=dict(action='nocache'),gt=dict(action='prophetkv-1'))
        p=make(tree,['nocache','prophetkv-1'],{'test':True})
        self.assertEqual(decide(p,{f:.5}),'nocache');self.assertEqual(decide(p,{f:float(np.nextafter(.5,np.inf))}),'prophetkv-1')
        for v in (None,True,float('inf'),float('nan')):self.assertEqual(decide(p,{f:v}),'nocache')
        self.assertEqual(decide(p,{}),'nocache')
        q=copy.deepcopy(p);q['tree']['threshold']=.6
        with self.assertRaises(ValueError):decide(q,{f:.7})
        with self.assertRaises(ValueError):make(dict(feature='task',threshold=1,le={'action':'nocache'},gt={'action':'nocache'}),['nocache'],{})
    def test_handoff_isolation_consumption(self):
        binding=dict(input_sha256='input',cache_identity='abc',runtime_sha256='runtime',gpu_uuids=['one','two'],source_request='abc:probe')
        a=np.arange(10,dtype=np.float32);h=Handoff(binding,a,{'probe_layers':64});a[:]=99
        for key in binding:
            bad=dict(binding,**{key:'changed'})
            with self.assertRaises(ValueError):h.arm(bad,'abc:answer')
        with self.assertRaises(ValueError):h.arm(binding,'def:answer')
        h.arm(binding,'abc:answer')
        with self.assertRaises(ValueError):h.consume(binding,'abc:another')
        scores,event=h.consume(binding,'abc:answer');np.testing.assert_array_equal(scores,np.arange(10))
        with self.assertRaises(ValueError):h.consume(binding,'abc:answer')
        with self.assertRaises(ValueError):h.arm(binding,'abc:answer2')
    def test_atomic_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'unit.json';sealed(path,{'features':[1,2]});before=(path.stat().st_mtime_ns,path.read_bytes())
            sealed(path,{'features':[1,2]});self.assertEqual(before,(path.stat().st_mtime_ns,path.read_bytes()))
            with self.assertRaises(FileExistsError):sealed(path,{'features':[3]})
            path.write_text(path.read_text().replace('[1,2]','[1,3]'))
            with self.assertRaises(ValueError):sealed(path)
    def test_required_base_arithmetic_exact(self):
        from runner.router_policy import features_from_arrays
        rng=np.random.default_rng(9);a=rng.random((64,1004));scores=rng.random(1000).astype(np.float32)
        full=features_from_arrays(a,scores,4,1004)
        for f in FEATURES:self.assertEqual(base_features(a,scores,4,1004,[f]),{f:full[f]})
        self.assertEqual(base_features(a,scores,4,1004,list(FEATURES)),full)
    def test_independent_boundaries(self):
        from runner.gpu_study_policy import check_boundaries
        p=make(dict(feature='head_coverage1_min',threshold=.5,le={'action':'nocache'},gt={'action':'prophetkv-1'}),['nocache','prophetkv-1'],{})
        self.assertGreater(check_boundaries(p,[{'head_coverage1_min':.2},{'head_coverage1_min':.8}]),2)
    def test_cost_shared_dependency_once(self):
        c=dict(native_transfer_rpc=1,aggregation=2,base_cpu=3,attention_statistics=4,statistics_reduce_transfer_sync=5,gpu_scalar_reduction=6,confidence=7,cpu_subsets={'top5_mass':8})
        self.assertEqual(cost_estimate(['head_coverage1_min'],c),18)
        self.assertEqual(cost_estimate(['head_coverage1_min','query_entropy_min'],c),18)
        self.assertEqual(cost_estimate(['head_coverage1_min','confidence_top1'],c),25)
        self.assertEqual(cost_estimate(['top5_mass'],c),11)

if __name__=='__main__':unittest.main()
