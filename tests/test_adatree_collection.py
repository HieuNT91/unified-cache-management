"""AdaTree A800 adoption, read-only baseline pairing and seeded dispatch (CPU)."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from runner.cache.setups import atomic_json, file_hash
from runner.corpus.runtime import Engine
from runner.router.adatree import SCHEMA, baseline_receipt, validate_record
from runner.tree.collection import collection_actions, inventory
from scripts.collection import tree_prophetkv_data as collection
from scripts.reporting import experiment_task_breakdown as reporting
from test_tree_collection import fixture, ROOT
from test_tree_chunk_size import enlarged
from test_baseline_breakdown import fixture as baseline_fixture, commit


class AdaTreeCollectionTests(unittest.TestCase):
    def test_adoption_keeps_exact_inputs_schedules_only_router_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp), 'longbench-v2')
            args.adatree_v1 = True; args.percentages = [5, 20, 40]
            args.answer_seeds = [0, 1]; args.baseline_source = Path(temp)/'baseline'
            receipt = collection.read(args.prepared/'preparation.json')
            for path in (args.prepared/'samples').glob('lb-*.json'):
                atomic_json(path, enlarged(collection.read(path)))
                receipt['files'][str(path.relative_to(args.prepared))] = file_hash(path)
            atomic_json(args.prepared/'preparation.json', receipt)
            pins = {p:(file_hash(p),p.stat().st_mtime_ns) for p in args.prepared.rglob('*') if p.is_file()}
            adopted = dict(source=str(args.baseline_source), answers=1006, answer_seeds=[0,1], metadata={})
            with patch('runner.router.adatree.baseline_receipt', return_value=adopted) as baseline, \
                    patch('scripts.router.router_control.code_hashes', return_value={}):
                collection.adopt(args)
                protocol, rows = collection.stage(args.root, 'tree')
                self.assertEqual(protocol['scheduled_actions'], ['adatree-v1'])
                self.assertEqual(protocol['answer_seeds'], [0,1])
                self.assertEqual(len(protocol['groups']), 2)
                plan = collection.load(args.root)[1]
                self.assertEqual((plan['answers'],plan['baseline_answers'],plan['probes']), (1006,0,0))
                self.assertEqual(len(rows),503)
                collection.adopt(args)
                baseline.assert_called_once()
                args.adatree_v1=False; args.baseline_source=None
                with self.assertRaisesRegex(ValueError,'parameters differ'): collection.adopt(args)
            self.assertEqual(pins, {p:(file_hash(p),p.stat().st_mtime_ns) for p in pins})

    def test_schedule_rejects_changed_router_seeds_or_layout(self):
        protocol = dict(adaptive_routing=SCHEMA, percentages=[5,20,40], dataset='longbench-v2',
                        answer_seeds=[0,1], answers_per_question=2)
        self.assertEqual(collection_actions(protocol), [dict(id='adatree-v1',method='adatree-v1',ratio=None)])
        for key,value in [('kv_chunk_size',4096),('answer_seeds',[2,3]),('percentages',[5,20,40,60]),
                          ('dataset','ruler'),('adaptive_routing','unknown')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                collection_actions(dict(protocol, **{key:value}))

    def test_baseline_reuses_original_receipts_exact_tokens_and_seed_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp); target,baseline=baseline_fixture(base,'longbench-v2')
            settings=dict(target[1],baseline_source=str(base/'baseline-source'))
            rows=[dict(r,token_sha256='a'*64) for r in target[2]]
            for i in range(3):
                for seed in (0,1): commit(baseline,'nocache',i=i,seed=seed)
            pins={p:(file_hash(p),p.stat().st_mtime_ns) for p in (base/'baseline-source').rglob('*') if p.is_file()}
            receipt=baseline_receipt(base,settings,rows)
            self.assertEqual(receipt['answers'],6)
            self.assertEqual(len(receipt['metadata']),4)
            self.assertEqual(pins,{p:(file_hash(p),p.stat().st_mtime_ns) for p in pins})
            rows[0]['token_sha256']='b'*64
            with self.assertRaisesRegex(ValueError,'token IDs'): baseline_receipt(base,settings,rows)
            rows[0]['token_sha256']='a'*64
            path=baseline[0]/'records/nocache'/rows[0]['id']/'seed-1/validated.json'
            path.unlink()
            with self.assertRaisesRegex(ValueError,'all prepared inputs'): baseline_receipt(base,settings,rows)

    def test_adaptive_worker_dispatch_and_resume_decision_validation(self):
        from runner.tree.collection import SCHEMA as TREE_SCHEMA
        from runner.tree.prophetkv import PROTOCOL
        protocol=dict(schema=TREE_SCHEMA,tree_protocol=PROTOCOL,adaptive_routing=SCHEMA,
            percentages=[5,20,40],dataset='longbench-v2',answer_seeds=[0,1],answers_per_question=2,
            phase_order=['cached'],answer_validation='native-answer-diagnostics-v1')
        definition=collection_actions(protocol)[0]
        protocol.update(actions=[definition],scheduled_actions=['adatree-v1'])
        engine=Engine.__new__(Engine);engine.protocol=protocol;engine.cached=True
        with patch('runner.router.adatree.answer',return_value=('record','diagnostics')) as answer:
            self.assertEqual(engine.answer(definition,'adatree-v1',seed=1),('record','diagnostics'))
            answer.assert_called_once_with(engine,1)
            with self.assertRaises(ValueError): engine.answer(definition,'adatree-v1',attention={})
        result=dict(executed_action='tree-prophetkv-20',independent_probes=0,
            routing=dict(schema=SCHEMA,features={'S60_p90':.8,'G5_mean':.2},action='tree-prophetkv-20',
                ratio=.2,independent_probes=0,feature_mask_only=True,included_in_native_probe=True,request_id='request'))
        validate_record(protocol,result)
        result['executed_action']='tree-prophetkv-40'
        with self.assertRaisesRegex(ValueError,'decision changed'): validate_record(protocol,result)

    def test_launcher_needs_no_configure_and_passes_exact_fixed_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            fake=Path(temp)/'python'
            fake.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n');fake.chmod(0o755)
            env=dict(PATH=os.environ['PATH'],PYTHON_BIN=str(fake))
            launcher=ROOT/'scripts/launcher/a800_longbench_adatree_v1.sh'
            for command in ('start','resume','status','report'):
                out=subprocess.run(['bash',str(launcher),command],cwd='/tmp',env=env,text=True,capture_output=True)
                self.assertEqual(out.returncode,0,out.stderr)
                argv=json.loads(out.stdout)
                self.assertIn('--adatree-v1',argv);self.assertNotIn('--kv-chunk-size',argv)
                self.assertEqual(argv[argv.index('--answer-seeds')+1:argv.index('--percentages')],['0','1'])
                self.assertTrue(argv[argv.index('--prepared')+1].endswith('tree-prophetkv-2seed-v1/prepared'))
                self.assertTrue(argv[argv.index('--baseline-source')+1].endswith('tree-prophetkv-2seed-v1/baseline-source'))
            out=subprocess.run(['bash',str(launcher),'start'],env=dict(env,KV_CHUNK_SIZE='1024'),capture_output=True)
            self.assertNotEqual(out.returncode,0)

    def test_report_pairs_adaptive_answers_with_external_baseline(self):
        from runner.cache.setups import fingerprint
        from runner.tree.prophetkv import PROTOCOL
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)/'run'; target,baseline=baseline_fixture(base,'longbench-v2')
            commit(baseline,'nocache',ttft=4.)
            external=Path(temp)/'old-baseline'; (base/'baseline-source').rename(external)
            settings=reporting.read(base/'settings.json')
            settings.update(adaptive_routing=SCHEMA,percentages=[5,20,40],tree_protocol=PROTOCOL,
                            phase_order=['cached'],baseline_source=str(external))
            atomic_json(base/'settings.json',settings)
            plan=reporting.read(base/'plan.json');plan['settings_sha256']=file_hash(base/'settings.json')
            atomic_json(base/'plan.json',plan)
            protocol=dict(target[1],**settings)
            protocol.update(actions=collection_actions(settings),scheduled_actions=['adatree-v1'],plan_sha256=fingerprint(plan))
            atomic_json(base/'tree/protocol.json',protocol)
            commit((target[0],protocol,target[2]),'adatree-v1',ttft=2.)
            folder=target[0]/'records/adatree-v1'/target[2][0]['id']/'seed-0'
            result=reporting.read(folder/'result.json')
            result.update(executed_action='tree-prophetkv-5',independent_probes=0,
                routing=dict(schema=SCHEMA,features={'S60_p90':.7,'G5_mean':.3},action='tree-prophetkv-5',
                    ratio=.05,independent_probes=0,feature_mask_only=True,included_in_native_probe=True,request_id='request'))
            atomic_json(folder/'result.json',result)
            receipt=reporting.read(folder/'validated.json');receipt['files']['result.json']=file_hash(folder/'result.json')
            atomic_json(folder/'validated.json',receipt)
            text,tables,matching=reporting.report(base)
            self.assertEqual(matching['matched_samples'],1)
            self.assertEqual(tables['overall']['nocache']['mean_ttft_seconds'],4.)
            self.assertEqual(tables['overall']['adatree-v1']['mean_ttft_seconds'],2.)
            self.assertIn(str(external),text)
            external.rename(Path(temp)/'missing-baseline')
            with self.assertRaises(FileNotFoundError): reporting.report(base)


if __name__=='__main__': unittest.main()
