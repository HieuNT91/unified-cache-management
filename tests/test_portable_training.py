"""Synthetic portable data: full cohorts, checksum gates and CPU tree pipeline."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from scripts.longbench_a800_data import ACTIONS, SCHEDULE, assign_folds
from scripts.router_dataset import save, load, validate
from scripts import train_router as training
from runner.longbench_features import FEATURES
from runner.setups import fingerprint, file_hash
from runner.corpus_train import search
from runner.tree_policy import load as load_tree, decide

ROOT=Path(__file__).resolve().parents[1]
SMALL_GRID=[dict(family='cost',depth=d,min_leaf=2,lam=w,penalty=0.) for d,w in ((1,1),(2,5),(3,20))]


def synthetic_rows(dataset, count=None):
    from scripts.ruler import TASKS,EVALUATION_PROTOCOL
    count=count or (1300 if dataset=='ruler' else 503)
    samples=count//13 if dataset=='ruler' else None
    rows=[]
    uuids=[f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(10)]
    for i in range(count):
        task=TASKS[i//samples] if dataset=='ruler' else 'domain'+str(i%7)
        baseline=10. if dataset=='ruler' else 100.
        fraction=(i%10)/10
        group=(i%samples)%5 if dataset=='ruler' else i%2
        row=dict(id=f'{task}-{i:04d}',dataset=dataset,subtask=task,input_sha256=fingerprint([dataset,i]),
                 source_pins={'record':fingerprint(i)},features=dict.fromkeys(FEATURES,fraction),
                 evaluation_protocol=EVALUATION_PROTOCOL if dataset=='ruler' else None,
                 probe_overhead_seconds=baseline*.01,outcomes={})
        for index,a in enumerate(ACTIONS):
            sparse=a['id']!='nocache'
            extra=dataset=='longbench-v2' and a['id'] in SCHEDULE['extra']
            start=2*group+4*extra
            row['outcomes'][a['id']]=dict(accuracy=(.7 if sparse and fraction<.3 and a['ratio']<.2 else 1.),
                ttft_seconds=baseline*(.1+index*.02) if sparse else baseline,
                gpu_uuids=uuids[start:start+2],thinking_tokens=0,answer_tokens=2,control_tokens=0,
                output_tokens=2,output_cap_reached=False)
        if dataset=='longbench-v2':row.update(length=('short','medium','long')[i%3],source_id=f'source-{i}')
        rows.append(row)
    return rows


def bundle(root,dataset):
    path=root/(dataset+'.json')
    save(path,dataset,ACTIONS,synthetic_rows(dataset),dict(tp=2,seed=42))
    return path


class PortableTests(unittest.TestCase):
    def test_full_cohorts_three_modes_no_silent_action_intersection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ruler=bundle(root,'ruler');lb=bundle(root,'longbench-v2')
            for mode,count in (('ruler',1300),('longbench-v2',503),('both',1803)):
                rows,inventory,meta=training.select_data(mode,ruler,lb)
                self.assertEqual(len(rows),count);self.assertEqual(inventory,ACTIONS)
                self.assertEqual(meta['weighting'],'equal-prompt');self.assertTrue(meta['training_overlap'])
                folds=assign_folds(rows)
                self.assertEqual(set(folds.values()),set(range(5)))
                self.assertEqual(len(folds),count)
            with self.assertRaisesRegex(ValueError,'12-action'):
                training.select_data('both',ruler,lb,['nocache','prophetkv-1'])

    def test_ruler200_export_import_and_combined_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ruler=root/'ruler200.json'; lb=bundle(root,'longbench-v2')
            value=save(ruler,'ruler',ACTIONS,synthetic_rows('ruler',2600),dict(tp=2,seed=42,samples_per_task=200))
            self.assertEqual(len(load(ruler)['rows']),2600)
            for mutate in (lambda d:d['provenance'].update(samples_per_task=100),
                           lambda d:d['rows'].pop(),lambda d:d['rows'][0].update(subtask='invalid')):
                bad=copy.deepcopy(value);mutate(bad)
                bad['payload_sha256']=fingerprint({k:v for k,v in bad.items() if k!='payload_sha256'})
                with self.assertRaises(ValueError):validate(bad)
            for mode,count in (('ruler',2600),('both',3103)):
                rows,inventory,meta=training.select_data(mode,ruler,lb)
                self.assertEqual(len(rows),count)
                self.assertEqual(meta['weighting'],'equal-prompt')
                self.assertEqual(len(assign_folds(rows)),count)
                if mode=='both':
                    with patch('runner.corpus_train.GRID',SMALL_GRID),patch.object(training,'GRID',SMALL_GRID):
                        report=training.train(rows,inventory,meta,root/'fit200')
                    self.assertEqual(report['samples'],3103)
                    self.assertEqual(len(report['policies']),3)
                    for item in report['policies'].values():
                        self.assertEqual(item['overall']['training_overlap'],3103)
                        self.assertEqual(item['datasets']['ruler']['overall']['samples'],2600)
                        self.assertEqual(item['datasets']['longbench-v2']['overall']['samples'],503)

    def test_checksum_incomplete_action_features_and_gpu_cohorts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=bundle(Path(tmp),'ruler');original=load(path)
            for mutate in (lambda d:d['rows'].pop(),lambda d:d['actions'].pop(),
                           lambda d:d['rows'][0]['features'].pop(FEATURES[0]),
                           lambda d:d['rows'][0]['outcomes'].pop('prophetkv-90'),
                           lambda d:d['rows'][0]['outcomes']['prophetkv-90'].update(gpu_uuids=d['rows'][1]['outcomes']['nocache']['gpu_uuids'])):
                bad=copy.deepcopy(original);mutate(bad)
                bad['payload_sha256']=fingerprint({k:v for k,v in bad.items() if k!='payload_sha256'})
                with self.assertRaises(ValueError):validate(bad)
            path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'checksum'):load(path)

    def test_completed_collection_exports_records_only_after_exit_gates(self):
        from test_a800_longbench import fixture
        from scripts.router_dataset import export_collection
        from scripts.longbench_a800_data import read
        from runner.setups import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            path=root/'longbench-data.json'
            with self.assertRaisesRegex(ValueError,'Finish all controls'):
                export_collection(root,path)
            for role in ('primary','extra','features'):
                atomic_json(root/role/('complete.json' if role=='features' else 'controls-complete.json'),
                            dict(owned_engines_exited=True))
            rows=read(root/'rows.json');fake=synthetic_rows('longbench-v2')
            for row in rows:
                for role in ('primary','extra','features'):
                    for case in SCHEDULE[role]:
                        atomic_json(root/role/'records'/case/row['id']/'validated.json',dict(complete=True))
            def accepted(base,case,row,protocol):
                source=fake[row['ordinal']]
                if case=='probe':
                    return dict(features=source['features'],timings=dict(routing_overhead_seconds=source['probe_overhead_seconds']))
                value=source['outcomes'][case]
                return dict(value,timings=dict(ttft_seconds=value['ttft_seconds']))
            with patch('scripts.router_dataset.accepted',side_effect=accepted), \
                 patch('scripts.longbench_a800_control.engines_idle') as idle:
                result=export_collection(root,path)
            self.assertEqual(len(result['rows']),503)
            self.assertEqual(result['actions'],ACTIONS)
            self.assertEqual({c.args[0].name for c in idle.call_args_list},{'primary','extra','features'})
            self.assertEqual(load(path),result)
            self.assertEqual(sum(len(r['outcomes']) for r in result['rows']),6036)

    def test_export_missing_checksum_recovers_without_rewriting_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=bundle(Path(tmp),'longbench-v2');prior=load(path)
            pin=(file_hash(path),path.stat().st_mtime_ns)
            Path(str(path)+'.sha256').unlink()
            save(path,'longbench-v2',ACTIONS,prior['rows'],prior['provenance'])
            self.assertEqual((file_hash(path),path.stat().st_mtime_ns),pin)
            self.assertEqual(load(path),prior)

    def test_train_all_modes_compiles_three_depth3_trees_and_overlapping_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ruler=bundle(root,'ruler');lb=bundle(root,'longbench-v2')
            for mode,count in (('ruler',1300),('longbench-v2',503),('both',1803)):
                rows,inventory,meta=training.select_data(mode,ruler,lb)
                output=root/mode
                # Exercise real fold fitting/compiler/export/evaluation with a small grid.
                with patch('runner.corpus_train.GRID',SMALL_GRID),patch.object(training,'GRID',SMALL_GRID):
                    report=training.train(rows,inventory,meta,output)
                self.assertEqual(report['samples'],count);self.assertEqual(report['primary'],'router1')
                self.assertEqual(len(report['policies']),3)
                for name in report['policies']:
                    tree=load_tree(output/(name+'.json'))
                    self.assertEqual(len(tree['training_ids']),count)
                    self.assertEqual(tree['heldout_ids'],[])
                    self.assertEqual(report['policies'][name]['overall']['training_overlap'],count)
                    with (output/(name+'-decisions.csv')).open() as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))),count)
                    def depth(node):return 0 if 'action' in node else 1+max(depth(node['le']),depth(node['gt']))
                    self.assertLessEqual(depth(tree['tree']),3)
                    self.assertEqual(decide(tree,{})['action'],'nocache')
                if mode=='both':
                    self.assertEqual(set(report['policies']['router1']['datasets']),{'ruler','longbench-v2'})
                    self.assertEqual(set(report['policies']['router1']['datasets']['longbench-v2']['lengths']),{'short','medium','long'})
                # An arbitrary server scale cancels before fit and OOF selection.
                if mode=='both':
                    scaled=copy.deepcopy(rows)
                    for row in scaled:
                        if row['dataset']=='longbench-v2':
                            row['probe_overhead_seconds']*=100
                            for outcome in row['outcomes'].values():outcome['ttft_seconds']*=100
                    with patch('runner.corpus_train.GRID',SMALL_GRID),patch.object(training,'GRID',SMALL_GRID):
                        training.train(scaled,inventory,meta,root/'scaled')
                    for name in report['policies']:
                        self.assertEqual(load_tree(output/(name+'.json'))['tree'],load_tree(root/'scaled'/(name+'.json'))['tree'])

    def test_prompt_weighting_not_task_or_dataset_macro(self):
        samples=[dict(id=str(i),task='large' if i<9 else 'small') for i in range(10)]
        features=[dict.fromkeys(FEATURES,.5) for _ in samples]
        scores=[[1.,1.] for _ in range(9)]+[[1.,0.]]
        inventory=[ACTIONS[0],ACTIONS[1]]
        grid=[dict(family='loss',depth=1,min_leaf=1,shrinkage=0,threshold=.5)]
        with patch('runner.corpus_train.GRID',grid):
            _,table,_=search(samples,features,scores,[[1.,.01]]*10,[0.]*10,inventory,count=1,
                              feature_names=FEATURES,folds=[i%5 for i in range(10)],score_weighting='prompt')
        self.assertAlmostEqual(table[0]['macro_score'],.9)
        self.assertEqual(table[0]['score_weighting'],'prompt')
        self.assertAlmostEqual(table[0]['macro_loss'],.1)

    def test_evaluate_cli_never_refits_and_writes_csv_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=bundle(root,'longbench-v2')
            rows,inv,meta=training.select_data('longbench-v2',longbench=path)
            with patch('runner.corpus_train.GRID',SMALL_GRID),patch.object(training,'GRID',SMALL_GRID):
                training.train(rows,inv,meta,root/'fit')
            args=['train_router.py','--dataset','longbench-v2','--longbench-data',str(path),
                  '--tree',str(root/'fit/router1.json'),'--output',str(root/'evaluation')]
            with patch.object(sys,'argv',args),patch.object(training,'search',side_effect=AssertionError('refit')),patch('builtins.print'):
                training.main()
            report=json.loads((root/'evaluation/evaluation.json').read_text())
            self.assertFalse(report['refit']);self.assertEqual(report['overall']['training_overlap'],503)
            other=synthetic_rows('ruler')
            other=[dict(r,sample_id='ruler:'+r['id']) for r in other]
            transfer,_=training.evaluate(load_tree(root/'fit/router1.json'),other)
            self.assertEqual(transfer['overall']['training_overlap'],0)
            self.assertTrue((root/'evaluation/decisions.csv').exists());self.assertTrue((root/'evaluation/evaluation.txt').exists())

    def test_launcher_loads_a800_config_and_forces_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fake=root/'python';envfile=root/'a800.env'
            fake.write_text(f'#!{sys.executable}\nimport os,sys,json\nprint(json.dumps([sys.argv[1:],os.environ["CUDA_VISIBLE_DEVICES"]]))\n');fake.chmod(0o755)
            envfile.write_text(f'PYTHON_BIN="{fake}"\nDATA_IMPORT_DIR="/import files"\nEXPERIMENT_DIR="/results"\nTRAINING_OUTPUT_DIR="/train"\n')
            env=dict(PATH=os.environ['PATH'],UCM_ENV_FILE=str(envfile))
            for mode in ('ruler','longbench-v2','both'):
                result=subprocess.run(['bash',str(ROOT/'scripts/launcher/router_training.sh'),'train',mode],env=env,cwd='/tmp',capture_output=True,text=True,check=True)
                args,cuda=json.loads(result.stdout);self.assertEqual(cuda,'')
                self.assertEqual(args[args.index('--dataset')+1],mode)
                self.assertEqual(args[args.index('--ruler-data')+1],'/import files/ruler-data.json')
            result=subprocess.run(['bash',str(ROOT/'scripts/launcher/router_training.sh'),'evaluate','both'],env=env,capture_output=True)
            self.assertEqual(result.returncode,2)

if __name__=='__main__':unittest.main()
