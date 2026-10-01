"""Numerical formulas, ordered CART equivalence, export and immutable resume."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from runner.attention_features import ADDITIONS, additional_features
from runner.attention_study import fit_ordered, compile_tree, decide, check_export, sealed, SCHEMA
from runner.corpus_train import fit_tree
from runner.extension_train import GRID


class AttentionStudyTests(unittest.TestCase):
    def uniform(self,n=1000):return np.ones((64,n+4),np.float64),np.ones(n,np.float32)

    def test_frozen_feature_inventory(self):
        self.assertEqual(len(ADDITIONS),40);self.assertEqual(len(set(ADDITIONS)),40)
        self.assertEqual(len(GRID),2592)

    def test_uniform_and_tied_position_boundaries(self):
        layers,scores=self.uniform();r=additional_features(layers,scores,4,1004)
        self.assertEqual(set(r),set(ADDITIONS))
        for name,expected in {'top5_mass':.05,'top10_mass':.1,'mass_1_to_5':.04,'mass_5_to_10':.05,'mass_10_to_20':.1,
                              'normalized_entropy':1,'effective_support':1,'mass90_fraction':.9,'mass95_fraction':.95,
                              'gap1':0,'gap5':0,'gap10':0,'band0_1_jsd':0,'band0_1_jaccard1':1}.items():
            self.assertAlmostEqual(r[name],expected,places=12,msg=name)
        for p in (1,5,10):
            for summary in ('min','p10','median'):self.assertAlmostEqual(r[f'coverage{p}_{summary}'],p/100)
            self.assertAlmostEqual(r[f'coverage{p}_std'],0)
            self.assertAlmostEqual(r[f'geometry{p}_adjacent_fraction'],1)
            self.assertAlmostEqual(r[f'geometry{p}_position_std'],np.arange(p*10).std()/999)
        # Excluded prefix mass must not alter any features.
        layers[:,:4]=1e9
        self.assertEqual(r,additional_features(layers,scores,4,1004))

    def test_concentrated_and_zero_mass(self):
        layers,scores=self.uniform();scores[:]=0;scores[-1]=1;layers[:]=0;layers[:,-1]=1
        r=additional_features(layers,scores,4,1004)
        self.assertEqual(r['normalized_entropy'],0);self.assertEqual(r['effective_support'],.001)
        self.assertEqual(r['mass90_fraction'],.001);self.assertEqual(r['mass95_fraction'],.001)
        self.assertEqual(r['top5_mass'],1);self.assertEqual(r['coverage1_min'],1)
        self.assertEqual(r['gap1'],0)  # zero boundary denominator is defined as zero
        # Stable zero ties select positions 0..8 and the final position.
        self.assertAlmostEqual(r['geometry1_position_std'],np.array([*range(9),999]).std()/999)
        self.assertAlmostEqual(r['geometry1_adjacent_fraction'],8/9)
        scores[:]=0
        self.assertTrue(all(v is None for v in additional_features(layers,scores,4,1004).values()))

    def test_layer_coverage_percentiles_and_disagreement(self):
        layers,scores=self.uniform();scores[:]=1;scores[:10]=2
        for i in range(64):layers[i,4:14]=i+1
        r=additional_features(layers,scores,4,1004)
        expected=np.arange(1,65)*10/(np.arange(1,65)*10+990)
        self.assertAlmostEqual(r['coverage1_p10'],np.percentile(expected,10,method='linear'))
        self.assertAlmostEqual(r['coverage1_median'],np.median(expected))
        self.assertAlmostEqual(r['coverage1_std'],np.std(expected,ddof=0))
        layers[:]=0;layers[:16,4]=1;layers[16:32,1003]=1;layers[32:,4]=1
        r=additional_features(layers,scores,4,1004)
        self.assertAlmostEqual(r['band0_1_jsd'],np.log(2))
        self.assertAlmostEqual(r['band0_1_jaccard1'],9/11)
        self.assertEqual(r['band0_coverage1'],1);self.assertEqual(r['band1_coverage1'],0)

    def test_nonzero_gap_and_undefined(self):
        layers,scores=self.uniform();scores[:10]=3;scores[10:]=1
        r=additional_features(layers,scores,4,1004)
        self.assertAlmostEqual(r['gap1'],2/3)
        layers[3]=0;r=additional_features(layers,scores,4,1004)
        self.assertIsNone(r['coverage1_min']);self.assertIsNone(r['coverage1_median'])
        layers,scores=self.uniform(10);r=additional_features(layers,scores,4,14)
        self.assertIsNone(r['gap1']);self.assertIsNone(r['geometry1_adjacent_fraction'])
        scores[0]=-1
        with self.assertRaises(ValueError):additional_features(layers,scores,4,14)

    def test_required_only_agrees(self):
        rng=np.random.default_rng(2);layers=rng.random((64,1004));scores=rng.random(1000).astype(np.float32)
        full=additional_features(layers,scores,4,1004)
        for name in ADDITIONS:self.assertEqual(additional_features(layers,scores,4,1004,[name]),{name:full[name]})

    def test_ordered_cart_original_equivalence(self):
        rng=np.random.default_rng(7);x=rng.integers(0,9,(70,5)).astype(float);y=rng.random((70,5));cost=rng.random((70,6))
        for depth in (1,2,3):
            for minimum in (5,10,20):
                for penalty in (0,.005,.02):
                    h=dict(depth=depth,min_leaf=minimum,penalty=penalty)
                    self.assertEqual(fit_ordered(x,y,h,cost),fit_tree(x,y,h,cost))

    def test_variable_columns_boundary_and_missing(self):
        x=np.zeros((40,6));x[:,5]=np.arange(40);y=np.ones((40,2));cost=np.ones((40,2));cost[:20,0]=0;cost[20:,1]=0
        raw=fit_ordered(x,y,dict(depth=3,min_leaf=5,penalty=0),cost)
        self.assertEqual(raw['feature'],5)
        columns=['a','b','c','d','e','new'];names=['nocache','one'];rows=[dict(zip(columns,r)) for r in x]
        tree=dict(schema=SCHEMA,offline_only=True,feature_names=columns,tree=compile_tree(raw,columns,names))
        self.assertGreater(check_export(tree,raw,rows,names),40)
        self.assertEqual(decide(tree,dict(rows[0],new=19.5)),'nocache')
        self.assertEqual(decide(tree,dict(rows[0],new=float(np.nextafter(19.5,np.inf)))),'one')
        with self.assertRaises(ValueError):decide(dict(tree,schema='runtime'),rows[0])

    def test_resume_preserves_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'unit.json';sealed(p,{'features':[1,2],'result':3});before=(p.stat().st_mtime_ns,p.read_bytes())
            sealed(p);sealed(p,{'features':[1,2],'result':3});self.assertEqual(before,(p.stat().st_mtime_ns,p.read_bytes()))
            with self.assertRaises(ValueError):sealed(p,{'features':[1,2],'result':4})
            v=json.loads(p.read_text());v['data']['result']=4;p.write_text(json.dumps(v))
            with self.assertRaises(ValueError):sealed(p)

if __name__=='__main__':unittest.main()
