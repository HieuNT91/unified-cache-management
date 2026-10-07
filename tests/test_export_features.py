"""CPU-only arithmetic, integrity and standalone export tests using synthetic data."""
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import numpy as np
from scripts import export_features as ef
from runner.longbench_features import features as legacy_features


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ef.canonical(value)+b'\n')


def fixture(root, dataset='longbench-v2', count=200, records=True, tp=2, thinking=False):
    settings = dict(schema='tp2-data-config-v2', dataset=dataset, tp=tp, seed=42,
                    samples_per_task=count, model='/missing/model', prepared='/missing/prepared', cache_root='/missing/cache')
    if thinking:
        settings.update(execution_profile='ruler-thinking', evaluation_protocol='synthetic-thinking')
    rows = []
    pairs = [('longbench', i) for i in range(503)] if dataset == 'longbench-v2' else [(t, i) for t in ef.TASKS for i in range(count)]
    for task, i in pairs:
        pid = f'{task}-{i}'
        row = dict(id=pid, ordinal=i, dataset=dataset, subtask=task, sha256=hashlib.sha256(pid.encode()).hexdigest(),
                   input_tokens=576, evaluation_protocol='synthetic', source_id=pid, length=('short','medium','long')[i%3])
        rows.append(row)
    write(root/'settings.json', settings); write(root/'rows.json', rows)
    schedule = {'primary':ef.PRIMARY,'extra':ef.EXTRA,'features':['probe']} if dataset=='longbench-v2' else {'ruler':list(ef.ACTION_NAMES),'features':['probe']}
    plan = dict(schema='tp2-data-plan-v2', settings_sha256=ef.file_hash(root/'settings.json'), rows_sha256=ef.file_hash(root/'rows.json'),
                actions=ef.ACTIONS,schedule=schedule,feature_profile=ef.LEGACY_DEFINITIONS,samples=len(rows),answers=len(rows)*12,probes=len(rows))
    write(root/'plan.json',plan)
    groups = {}
    for k, role in enumerate(r for r in schedule if r!='features'):
        group_count = 2 if dataset=='longbench-v2' else 4 if thinking else 5
        groups[role]=[[f'GPU-{k:08x}-0000-0000-0000-{g*tp+j:012x}' for j in range(tp)] for g in range(group_count)]
        write(root/role/'devices.json',dict(groups=groups[role],tp=tp))
    groups['features']=groups['ruler' if dataset=='ruler' else 'primary']
    protocols={}
    for role,cases in schedule.items():
        proto=dict(schema='tp2-data-stage-v2',dataset=dataset,tp=tp,plan_sha256=ef.file_hash(root/'plan.json'),
                   actions=ef.ACTIONS,scheduled_actions=cases,groups=groups[role],answer_validation=ef.PROBE if role=='features' else ef.NATIVE,
                   prepared='/missing/prepared',model='/missing/model',cache_root='/missing/cache')
        if role=='features':proto['feature_profile']=ef.LEGACY_DEFINITIONS
        if thinking:proto.update(execution_profile='ruler-thinking',evaluation_protocol='synthetic-thinking')
        protocols[role]=proto;write(root/role/'protocol.json',proto)
        completion=dict(protocol_sha256=ef.file_hash(root/role/'protocol.json'),owned_engines_exited=True)
        completion['probes' if role=='features' else 'answers']=len(rows)*len(cases)
        write(root/role/('complete.json' if role=='features' else 'controls-complete.json'),completion)
        if role!='features':write(root/role/'complete.json',dict(complete=True,owned_engines_exited=True,plan_sha256=ef.file_hash(root/'plan.json')))
        write(root/role/'session/initialization.json',dict(validated=True,engine_config={}))
    if not records:return rows
    shared=root/'synthetic-shards';shared.mkdir()
    end,prefix=320,64
    local=np.full((64,end),1/end,np.float32)
    mean=np.zeros(end,np.float32)
    for v in local:mean+=v
    mean/=np.float32(64)
    score=mean[prefix:].copy()
    array_heads=np.full((tp,8,64//tp,end),1/end,np.float32)
    att=dict(layers=np.stack([local]*tp),scores=score,heads=array_heads,head_layers=ef.HEAD_LAYERS)
    features=legacy_features(att,prefix,end)
    archives=[]
    for rank in range(tp):
        file=shared/f'attention.rank{rank}.npz'
        arrays=dict(layers=local,local_mean=mean,scores=score,heads=array_heads[rank],head_layers=np.array(ef.HEAD_LAYERS,np.int64),
                    boundaries=np.array([0,prefix,end,576],np.int64),question_positions=np.array([end],np.int64),
                    context_positions=np.arange(end,dtype=np.int64),original_to_formatted=np.arange(576,dtype=np.int64))
        arrays.update({f'prophetkv-{b}':ef.mask(ef.ranked(score),b)+prefix for b in ef.RATIOS})
        np.savez_compressed(file,**arrays)
        archives.append(dict(rank=rank,path=file.name,sha256=ef.file_hash(file)))
    for row in rows:
        for role,cases in schedule.items():
            proto=protocols[role];g=row['ordinal']%len(proto['groups'])
            for case in cases:
                folder=root/role/'records'/case/row['id'];folder.mkdir(parents=True)
                result=dict(prompt_id=row['id'],method=case,input_sha256=row['sha256'],group=g,gpu_uuids=proto['groups'][g],cache_immutable=True,
                            prompt_protocol=ef.PROMPT_PROTOCOL,token_sha256=row['sha256'],answer_validation=proto['answer_validation'],evaluation_protocol='synthetic',
                            retirement=[dict(rank=r,quiescent=True,transfers=dict(pending=0),request_bookkeeping=0) for r in range(tp)],
                            initialization='session/initialization.json',initialization_sha256=ef.file_hash(root/role/'session/initialization.json'))
                if case=='probe':
                    result.update(internal_tokens=1,features=features,artifacts=archives,timings=dict(routing_overhead_seconds=.25))
                    for a in archives:os.link(shared/a['path'],folder/a['path'])
                else:
                    j=ef.ACTION_NAMES.index(case)
                    result.update(executed_action=case,accuracy=(j%3)/2,timings=dict(ttft_seconds=10+j),num_cached_tokens=0,evaluation_protocol='synthetic')
                write(folder/'result.json',result);write(folder/'diagnostics.json',[])
                files={p.name:ef.file_hash(p) for p in folder.iterdir()}
                write(folder/'validated.json',dict(complete=True,protocol_sha256=ef.identity(proto),files=files))
    return rows


def vary_probe(root, row, peak):
    """Different scalar features expose reordered-row bugs in parallel exports."""
    folder=root/'features/records/probe'/row['id']
    record=json.loads((folder/'result.json').read_text())
    local=np.full((64,320),.001,np.float32);local[:,peak]=1
    mean=np.zeros(320,np.float32)
    for v in local:mean+=v
    mean/=np.float32(64)
    scores=mean[64:].copy()
    captured=[]
    for artifact in record['artifacts']:
        path=folder/artifact['path']
        with np.load(path,allow_pickle=False) as saved:
            arrays={k:saved[k] for k in saved.files}
        heads=np.broadcast_to(local[0],arrays['heads'].shape).copy()
        arrays.update(layers=local,local_mean=mean,scores=scores,heads=heads)
        arrays.update({f'prophetkv-{b}':ef.mask(ef.ranked(scores),b)+64 for b in ef.RATIOS})
        path.unlink()  # Other synthetic prompts share the original immutable inode.
        np.savez_compressed(path,**arrays)
        artifact['sha256']=ef.file_hash(path);captured.append(heads)
    record['features']=legacy_features(dict(layers=np.stack([local]*2),scores=scores,
        heads=np.stack(captured),head_layers=ef.HEAD_LAYERS),64,320)
    write(folder/'result.json',record)
    receipt=json.loads((folder/'validated.json').read_text())
    receipt['files']={p.name:ef.file_hash(p) for p in folder.iterdir() if p.name!='validated.json'}
    write(folder/'validated.json',receipt)


class FeatureMathTests(unittest.TestCase):
    def uniform(self):
        return np.full((64,320),1/320),np.full((8,64,320),1/320,np.float32),np.full(256,1/320,np.float32),64

    def test_schema_matches_document(self):
        doc=(Path(__file__).resolve().parents[1]/'features_list.md').read_text()
        names=re.findall(r'^\| \d{3} \| `([^`]+)`',doc,re.M)
        self.assertEqual(names,list(ef.FEATURE_NAMES));self.assertEqual(len(set(names)),100)
        self.assertEqual(len({s[1] for s in ef.FEATURE_SPEC}),18)

    def test_uniform_and_floor_ties(self):
        f=dict(zip(ef.FEATURE_NAMES,ef.compute_features(*self.uniform())))
        self.assertAlmostEqual(f['top1_mass'],2/256)
        self.assertAlmostEqual(f['top20_mass'],51/256)
        self.assertEqual(f['attention_entropy_norm'],1)
        self.assertEqual(f['attention_gini'],0)
        self.assertAlmostEqual(f['group_agreement'],1)
        self.assertEqual(f['mass80_token_fraction'],205/256)
        self.assertAlmostEqual(f['first_chunk_mass_global'],.2)
        self.assertEqual(f['within_layer_head_top5_jaccard_mean'],1)
        self.assertEqual(f['head_top5_consensus90_fraction'],1)
        self.assertEqual(f['head_top5_union_ratio'],12/256)
        self.assertTrue(np.isnan(f['attention_lag1_autocorrelation']))
        self.assertTrue(np.isnan(f['largest_adjacent_layer_shift_position']))

    def test_one_hot_and_zero_mass(self):
        layers,heads,scores,prefix=self.uniform()
        layers[:]=0;heads[:]=0;scores[:]=0
        for a in (layers,heads):a[...,128]=1
        scores[64]=1
        f=dict(zip(ef.FEATURE_NAMES,ef.compute_features(layers,heads,scores,prefix)))
        self.assertEqual(f['top1_mass'],1);self.assertEqual(f['head_coverage1_p10'],1)
        self.assertEqual(f['mass99_token_fraction'],1/256)
        self.assertEqual(f['attention_entropy_norm'],0)
        self.assertEqual(f['effective_support_ratio'],1/256)
        self.assertEqual(f['head_peak_position_std'],0)
        self.assertTrue(np.isnan(f['first_chunk_entropy_norm']))
        layers[:]=0;heads[:]=0;scores[:]=0
        f=dict(zip(ef.FEATURE_NAMES,ef.compute_features(layers,heads,scores,prefix)))
        self.assertTrue(np.isnan(f['coverage5_median']))
        self.assertTrue(np.isnan(f['head_coverage1_p10']))
        self.assertTrue(np.isnan(f['group_agreement']))
        with self.assertRaisesRegex(ef.ExportError,'Malformed'):
            layers[0,0]=-1;ef.compute_features(layers,heads,scores,prefix)

    def test_legacy_math_and_head_masks_independently(self):
        rng=np.random.default_rng(18);prefix,end=64,320
        layers=rng.random((2,64,end),dtype=np.float32)
        heads=rng.random((2,8,32,end),dtype=np.float32)
        score=rng.random(end-prefix,dtype=np.float32)
        # Native arrays are validated by the loader; this tests extraction independently.
        att=dict(layers=layers,heads=heads,scores=score,head_layers=ef.HEAD_LAYERS)
        expected=legacy_features(att,prefix,end)
        merged=np.concatenate(list(heads),axis=1)
        got=dict(zip(ef.FEATURE_NAMES,ef.compute_features(layers.astype(np.float64).mean(0),merged,score,prefix)))
        for name,value in expected.items():self.assertAlmostEqual(got[name],value,places=12,msg=name)
        # Check pairwise mask similarity against a straightforward Python-set oracle.
        values=[]
        for group in merged:
            masks=[set(np.argsort(-h[prefix:],kind='stable')[:12].tolist()) for h in group]
            for i,a in enumerate(masks):
                for b in masks[i+1:]:values.append(len(a&b)/len(a|b))
        self.assertAlmostEqual(got['within_layer_head_top5_jaccard_mean'],np.mean(values),places=12)
        self.assertTrue(all(np.isfinite(v) for v in got.values()))

    def test_missing_head_propagates_without_dropping_it(self):
        layers,heads,scores,prefix=self.uniform();heads[0,0]=0
        f=dict(zip(ef.FEATURE_NAMES,ef.compute_features(layers,heads,scores,prefix)))
        self.assertTrue(np.isnan(f['head_coverage1_p10']))
        self.assertTrue(np.isnan(f['head_entropy_mean']))
        self.assertTrue(np.isnan(f['head_top5_union_ratio']))
        self.assertTrue(np.isfinite(f['coverage5_median']))


class WorkerConfigurationTests(unittest.TestCase):
    def test_default_64_and_explicit_workers(self):
        for args,expected in [([],64),(['--workers','2'],2),(['--workers','1'],1)]:
            with self.subTest(args=args),patch.object(ef,'export') as export:
                self.assertEqual(ef.main(['/unused','--quiet',*args]),0)
                self.assertEqual(export.call_args.kwargs['workers'],expected)
        for value in ('0','-1','1.5','bad'):
            with self.subTest(value=value),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:
                ef.main(['/unused','--workers',value])
            self.assertEqual(error.exception.code,2)

    def test_worker_environment_limits_and_restoration(self):
        with patch.dict(os.environ,{'OPENBLAS_NUM_THREADS':'64','MKL_NUM_THREADS':'32'},clear=True):
            before=dict(os.environ)
            with self.assertRaisesRegex(RuntimeError,'injected'):
                with ef.worker_environment():
                    self.assertTrue(all(os.environ[v]=='1' for v in ef.BLAS_THREAD_VARIABLES))
                    raise RuntimeError('injected')
            self.assertEqual(dict(os.environ),before)


class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='compact export test ')
        cls.root=Path(cls.temp.name)/'source';cls.root.mkdir()
        cls.rows=fixture(cls.root)
        vary_probe(cls.root,cls.rows[1],100)
        vary_probe(cls.root,cls.rows[2],260)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_load_rank_replay_and_corruption(self):
        c=ef.Collection(self.root);row=c.rows[0]
        probe,folder=c.record('features','probe',row)
        data=ef.load_attention(folder,probe,row,2)
        self.assertEqual(data[1].shape,(8,64,320))
        bad=dict(probe,artifacts=[probe['artifacts'][0]]*2)
        with self.assertRaises(ef.ExportError):
            # Duplicate ranks are rejected before archive loading.
            payload=json.loads((folder/'result.json').read_text());payload['artifacts']=bad['artifacts']
            with patch.object(c,'read',side_effect=lambda name,expected=None: payload if name.endswith('result.json') else json.loads((self.root/name).read_text())):
                c.record('features','probe',row)
        bad=dict(probe,artifacts=[dict(a,sha256='0'*64) for a in probe['artifacts']])
        with self.assertRaisesRegex(ef.ExportError,'checksum'):ef.load_attention(folder,bad,row,2)

    def test_incomplete_metadata_and_membership(self):
        for dataset,count,thinking,tp in [('ruler',100,False,2),('ruler',200,False,2),('ruler',30,True,4)]:
            with self.subTest(count=count),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);fixture(root,dataset,count,False,tp,thinking)
                c=ef.Collection(root);self.assertEqual(len(c.rows),13*count)
                rowpath=root/'rows.json';rows=json.loads(rowpath.read_text());rows[-1]['ordinal']=0
                write(rowpath,rows)
                plan=json.loads((root/'plan.json').read_text());plan['rows_sha256']=ef.file_hash(rowpath);write(root/'plan.json',plan)
                with self.assertRaisesRegex(ef.ExportError,'unique ordinals'):ef.Collection(root)
        path=self.root/'primary/controls-complete.json'
        original=path.read_bytes()
        try:
            marker=json.loads(original);marker['owned_engines_exited']=False;write(path,marker)
            with self.assertRaisesRegex(ef.ExportError,'engine-exit'):ef.Collection(self.root)
        finally:path.write_bytes(original)

    def test_missing_control_and_mismatched_prompt(self):
        c=ef.Collection(self.root);row=c.rows[0]
        path=self.root/'extra/records/prophetkv-10'/row['id']/'validated.json'
        original=path.read_bytes()
        try:
            path.unlink()
            with self.assertRaisesRegex(ef.ExportError,'Missing required'):c.record('extra','prophetkv-10',row)
        finally:path.write_bytes(original)
        bad=dict(row,sha256='f'*64)
        with self.assertRaisesRegex(ef.ExportError,'identity mismatch'):c.record('extra','prophetkv-10',bad)

    def test_relocated_standalone_cli_full_cohort(self):
        # Copy only the script, never import the repo in the child process.
        work=Path(self.temp.name)/'standalone';work.mkdir(exist_ok=True)
        script=work/'export_features.py';shutil.copyfile(ef.__file__,script)
        output=work/'longbench-v2-features.npz'
        # The CLI must override inherited thread fan-out before NumPy loads.
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',OPENBLAS_NUM_THREADS='64',OMP_NUM_THREADS='64')
        result=subprocess.run([sys.executable,str(script),str(self.root),'--output',str(output),'--workers','2','--quiet'],
                              cwd=work,env=env,text=True,capture_output=True,timeout=240)
        self.assertEqual(result.returncode,0,result.stderr)
        arrays,meta=ef.verify_npz(output)
        self.assertEqual(arrays['X'].shape,(503,100));self.assertEqual(arrays['y_ttft'].shape,(503,12))
        self.assertEqual(arrays['X'].dtype,np.float32)
        np.testing.assert_array_equal(arrays['y_accuracy'][0],[(j%3)/2 for j in range(12)])
        np.testing.assert_array_equal(arrays['y_ttft'][0],np.arange(10,22))
        np.testing.assert_array_equal(arrays['X_missing'],np.isnan(arrays['X']))
        self.assertTrue(all(not v.dtype.hasobject for v in arrays.values()))
        self.assertEqual(meta['feature_count'],100);self.assertEqual(len(meta['feature_definitions']),100)
        self.assertEqual(len(set(arrays['length_group'])),3)
        self.assertTrue((arrays['probe_overhead_seconds']==.25).all())
        self.assertIn('not online',meta['timing'])
        self.assertEqual(meta['export_execution']['workers_used'],2)
        self.assertEqual(meta['export_execution']['blas_threads_per_worker'],1)
        self.assertEqual(meta['export_execution']['start_method'],'spawn')
        serial=ef.export(self.root,work/'serial.npz',quiet=True,workers=1)
        serial_arrays,serial_meta=ef.verify_npz(serial)
        for name in arrays:
            if name!='offline_extraction_seconds':
                np.testing.assert_array_equal(arrays[name],serial_arrays[name],err_msg=name)
        self.assertFalse(np.array_equal(arrays['X'][0],arrays['X'][1],equal_nan=True))
        self.assertFalse(np.array_equal(arrays['X'][1],arrays['X'][2],equal_nan=True))
        self.assertEqual(meta['source_record_receipts_sha256'],serial_meta['source_record_receipts_sha256'])
        self.assertEqual(meta['source_control_files'],serial_meta['source_control_files'])
        prior=ef.file_hash(output)
        with self.assertRaisesRegex(ef.ExportError,'exists'):ef.export(self.root,output,quiet=True)
        self.assertEqual(ef.file_hash(output),prior)
        # Data tampering must invalidate the portable payload checksum.
        arrays['X'][0,0]=.123
        tampered=work/'tampered.npz'
        np.savez_compressed(tampered,**arrays,metadata_json=np.array(json.dumps(meta)))
        with self.assertRaisesRegex(ef.ExportError,'checksum'):ef.verify_npz(tampered)

    def test_source_read_lock_and_path_escape(self):
        with self.assertRaisesRegex(ef.ExportError,'Unsafe'):ef.safe_path(self.root,'../outside')
        import fcntl
        path=self.root/'primary/run.lock'
        with path.open('w') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ef.ExportError,'still running'):
                with ef.source_read_lock(self.root):pass
        path.unlink()

    def test_ruler_export_default_name_full_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root,'ruler',100)
            c=ef.Collection(root);probe,folder=c.record('features','probe',c.rows[0])
            values=ef.compute_features(*ef.load_attention(folder,probe,c.rows[0],2))
            # Math is exercised above and in the real standalone CLI test. Reuse
            # its result for identical synthetic attention, still validate every
            # archive/record and the RULER-specific role/ordinal joins.
            with patch.object(ef,'compute_features',return_value=values) as compute:
                output=ef.export(root,quiet=True,workers=1)
            self.assertEqual(compute.call_count,1300)
            self.assertEqual(output.name,'ruler-features.npz')
            a,m=ef.verify_npz(output)
            self.assertEqual(a['X'].shape,(1300,100))
            self.assertEqual(a['y_accuracy'].shape,(1300,12))
            self.assertEqual(set(a['task']),set(ef.TASKS))
            for i in range(1300):
                self.assertEqual(len({tuple(g) for g in a['control_gpu_uuids'][i]}),1)
            self.assertEqual(m['source_settings']['samples_per_task'],100)

    def test_worker_failure_cleans_up_and_publishes_nothing(self):
        output=Path(self.temp.name)/'worker-failure.npz'
        original=ef.extracted_prompts
        before={p.pid for p in multiprocessing.active_children()}
        def fail_one(jobs,workers):
            def changed():
                for index,folder,probe,row,tp in jobs:
                    if index==0:
                        probe=json.loads(json.dumps(probe))
                        probe['artifacts'][0]['sha256']='0'*64
                    yield index,folder,probe,row,tp
            return original(changed(),workers)
        with patch.object(ef,'extracted_prompts',side_effect=fail_one):
            with self.assertRaisesRegex(ef.ExportError,'Prompt longbench-0: Attention checksum mismatch'):
                ef.export(self.root,output,quiet=True,workers=2)
        self.assertFalse(output.exists())
        self.assertEqual({p.pid for p in multiprocessing.active_children()},before)

    def test_tp4_archive_head_coverage_and_shapes(self):
        c=ef.Collection(self.root);row=c.rows[0]
        probe,folder=c.record('features','probe',row)
        with np.load(folder/'attention.rank0.npz',allow_pickle=False) as d:
            arrays={k:d[k] for k in d.files}
        arrays['heads']=arrays['heads'][:,:16]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);artifacts=[]
            for rank in range(4):
                file=root/f'attention.rank{rank}.npz'
                np.savez_compressed(file,**arrays)
                artifacts.append(dict(rank=rank,path=file.name,sha256=ef.file_hash(file)))
            data=ef.load_attention(root,dict(artifacts=artifacts),row,4)
            self.assertEqual(data[1].shape,(8,64,320))
            values=dict(zip(ef.FEATURE_NAMES,ef.compute_features(*data)))
            self.assertAlmostEqual(values['head_coverage1_p10'],2/256)
            arrays['heads']=arrays['heads'][:,:15]
            file=root/'attention.rank3.npz';np.savez_compressed(file,**arrays)
            artifacts[-1]['sha256']=ef.file_hash(file)
            with self.assertRaisesRegex(ef.ExportError,'all-64-Q-head'):
                ef.load_attention(root,dict(artifacts=artifacts),row,4)

    def test_publication_failure_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'out.npz'
            with patch.object(ef,'verify_npz',side_effect=ef.ExportError('injected failure')):
                with self.assertRaisesRegex(ef.ExportError,'injected'):
                    ef.publish_npz(output,{'X':np.zeros((2,100),np.float32)},{'schema':'attention-router-compact-v1'})
            self.assertFalse(output.exists())
            self.assertEqual(list(root.iterdir()),[])


if __name__=='__main__':unittest.main()
