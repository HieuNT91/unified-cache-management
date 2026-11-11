"""CPU checks for reproducible input adoption, both chunk routers and launcher."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from runner.cache.setups import file_hash
from runner.tree.collection import collection_actions, runtime_sample
from scripts.collection import run_controller as collection
from scripts import run

from run_fixtures import GPUS, fixture

ROOT = Path(__file__).resolve().parents[1]


class RunControllerTests(unittest.TestCase):
    def test_both_datasets_and_chunks_keep_prepared_bytes_and_share_policy(self):
        for dataset in ('ruler', 'longbench-v2'):
            for chunk in (4096, 2048):
                with self.subTest(dataset=dataset,chunk=chunk), tempfile.TemporaryDirectory() as tmp:
                    args=fixture(Path(tmp),dataset,chunk)
                    pins={p:(file_hash(p),p.stat().st_mtime_ns) for p in args.prepared.rglob('*') if p.is_file()}
                    collection.adopt(args)
                    protocol,rows=collection.stage(args.root,'tree')
                    self.assertEqual(protocol['scheduled_actions'], ['adatree-v1','nocache'])
                    self.assertEqual(protocol['phase_order'], ['cached','baseline'])
                    self.assertEqual(protocol['answer_generation']['thinking_cap'],8192)
                    self.assertEqual(collection.load(args.root)[1]['answers'],4)
                    for row in rows:
                        sample=json.loads((args.prepared/row['prepared']).read_text())
                        before=copy.deepcopy(sample)
                        effective=runtime_sample(sample,protocol)
                        self.assertEqual(effective['token_ids'],before['token_ids'])
                        self.assertEqual(sample,before)
                        self.assertTrue(all(b-a<=chunk for a,b in zip(effective['boundaries'][:-2],effective['boundaries'][1:-1])))
                    collection.adopt(args)
                    self.assertEqual(pins,{p:(file_hash(p),p.stat().st_mtime_ns) for p in pins})
                    changed=dict(protocol,followup_nocache=False)
                    with self.assertRaises(ValueError):collection_actions(changed)
                    args.kv_chunk_size=2048 if chunk==4096 else 4096
                    args.percentages=[5,20,60] if args.kv_chunk_size==2048 else [5,20,40]
                    with self.assertRaisesRegex(ValueError,'parameters differ'):collection.adopt(args)

    def test_rejects_device_counts_before_writing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp),'ruler',4096)
            for count in (0,3,8):
                args.number_gpus=count
                with self.assertRaises(ValueError):collection.adopt(args)
                self.assertFalse(args.root.exists())

    def test_download_retains_existing_sources_and_other_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.json'; path.write_text('different')
            with patch.object(run.urllib.request,'urlopen',side_effect=AssertionError('network')):
                with self.assertRaises(ValueError):run.download_longbench(path)
                self.assertEqual(path.read_text(),'different')
                path.unlink(); temporary=path.with_name('data.json.download'); temporary.write_text('owned elsewhere')
                with self.assertRaises(FileExistsError):run.download_longbench(path)
                self.assertEqual(temporary.read_text(),'owned elsewhere')

    def test_launcher_env_paths_hyperparameters_and_explicit_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp=Path(tmp); fake=temp/'python'
            fake.write_text(f'#!{sys.executable}\nimport json,sys,os\nprint(json.dumps([sys.argv[1:],os.environ["CUDA_VISIBLE_DEVICES"]]))\n')
            fake.chmod(0o755)
            env_file=temp/'config.env'
            env_file.write_text('\n'.join(f'{key}="{value}"' for key,value in dict(
                PYTHON_BIN=fake,MODEL_PATH=temp/'Qwen3-32B',RULER_PATH=temp/'RULER',LONGBENCH_DATA=temp/'data.json',
                RULER_PREPARED=temp/'ruler',LONGBENCH_PREPARED=temp/'longbench',CACHE_ROOT=temp/'cache',
                RESULTS_DIR=temp/'results',GPU_UUIDS=','.join(GPUS)).items())+'\n')
            for command,dataset,extra in [('prepare','ruler',[]),('configure','longbench-v2',[]),('smoke','ruler',['input-0'])]:
                proc=subprocess.run(['bash',str(ROOT/'run.sh'),command,dataset,*extra],cwd='/tmp',
                    env=dict(PATH=os.environ['PATH'],UCM_ENV_FILE=str(env_file)),text=True,capture_output=True)
                self.assertEqual(proc.returncode,0,proc.stderr)
                argv,cuda=json.loads(proc.stdout);self.assertEqual(cuda,'')
                if command=='prepare':self.assertIn('--prepared-answer-seeds',argv)
                else:
                    self.assertIn('--followup-nocache',argv)
                    self.assertEqual(argv[argv.index('--number-gpus')+1],'4')
                    self.assertIn(str(temp/'results'),argv[argv.index('--root')+1])
                    if command=='smoke':self.assertEqual(argv[argv.index('--tp')+1],'4');self.assertIn('--local-validation',argv)

    def test_stop_uses_process_receipts_when_prepared_inputs_are_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp),'ruler',4096)
            collection.adopt(args)
            args.prepared.rename(args.prepared.with_name('inputs-unavailable'))
            with patch.object(sys,'argv',['run_controller.py','stop','--root',str(args.root)]), \
                    patch.object(collection,'stage',side_effect=AssertionError('Stop must not open prepared data')):
                collection.main()
            receipt=json.loads((args.root/'tree/stop-receipt.json').read_text())
            self.assertTrue(receipt['owned_engines_exited'])
            self.assertEqual(receipt['processes'],[])

    def test_smoke_stop_and_status_need_only_python_and_results_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp=Path(tmp); fake=temp/'python'
            fake.write_text(f'#!{sys.executable}\nimport json,sys,os\nprint(json.dumps([sys.argv[1:],os.environ["CUDA_VISIBLE_DEVICES"]]))\n')
            fake.chmod(0o755)
            env_file=temp/'config.env'
            env_file.write_text(f'PYTHON_BIN="{fake}"\nRESULTS_DIR="{temp}/results"\n')
            for command in ('stop','status','smoke-stop','smoke-status'):
                proc=subprocess.run(['bash',str(ROOT/'run.sh'),command,'ruler'],cwd='/tmp',
                    env=dict(PATH=os.environ['PATH'],UCM_ENV_FILE=str(env_file)),text=True,capture_output=True)
                self.assertEqual(proc.returncode,0,proc.stderr)
                argv,cuda=json.loads(proc.stdout);self.assertEqual(cuda,'')
                target=temp/'results/adatree-v1.1'
                if command.startswith('smoke-'):target=target/'smoke'
                target=target/'ruler/chunk4096'
                self.assertEqual(argv[-1],str(target))
                if command.endswith('stop'):
                    self.assertEqual(argv[1:],['stop','--root',str(target)])
                else:self.assertEqual(Path(argv[0]).name,'status.py')


if __name__=='__main__':unittest.main()
