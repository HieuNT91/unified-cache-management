"""Depth expansion, structural bounds and original-search compatibility."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import unittest
import numpy as np
from runner.deeper_router_study import GRID, OLD_GRID, key, tree_shape
from runner.attention_study import fit_ordered, compile_tree, check_export, SCHEMA
from runner.corpus_train import fit_tree


class DeeperRouterTests(unittest.TestCase):
    def test_grid_is_exact_depth_extension(self):
        self.assertEqual(len(GRID),4320)
        self.assertEqual([h for h in GRID if h['depth']<=3],OLD_GRID)
        self.assertEqual(len({key(h) for h in GRID}),4320)
        for d in range(1,6):self.assertEqual(sum(h['depth']==d for h in GRID),864)

    def test_deep_fitter_matches_independent_original(self):
        rng=np.random.default_rng(93)
        x=rng.integers(0,20,(260,5)).astype(float);y=rng.random((260,5));cost=rng.random((260,6))
        for depth in (4,5):
            for minimum in (5,10,20):
                for penalty in (0,.005,.02):
                    h=dict(depth=depth,min_leaf=minimum,penalty=penalty)
                    raw=fit_ordered(x,y,h,cost)
                    self.assertEqual(raw,fit_tree(x,y,h,cost))
                    tree_shape(raw,minimum,depth)

    def test_full_depth_maximum_leaves_and_boundaries(self):
        # A separate action per interval gives a strict improvement at every split.
        # Symmetric binary feature columns force a balanced depth-d tree.
        for depth in (4,5):
            n=2**depth;indices=np.repeat(np.arange(n),5)
            x=np.array([[(i>>b)&1 for b in range(depth)] for i in indices],float)
            cost=np.ones((len(x),n));cost[np.arange(len(x)),indices]=0
            h=dict(depth=depth,min_leaf=5,penalty=0)
            raw=fit_ordered(x,cost,h,cost)
            self.assertEqual(tree_shape(raw,5,depth),(depth,n))
            names=['nocache']+[f'action-{i}' for i in range(1,n)];columns=[f'f{i}' for i in range(depth)]
            rows=[dict(zip(columns,r)) for r in x]
            policy=dict(schema=SCHEMA,offline_only=True,feature_names=columns,tree=compile_tree(raw,columns,names))
            self.assertGreater(check_export(policy,raw,rows,names),len(rows)+3*(n-1))
            with self.assertRaises(ValueError):tree_shape(raw,6,depth)
            with self.assertRaises(ValueError):tree_shape(raw,5,depth-1)

if __name__=='__main__':unittest.main()
