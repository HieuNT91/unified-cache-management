"""Matched status must compare identical accepted prompts without running inference."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from runner.matched_status import committed_records, match_records
from runner.reporting import aggregate
from runner.setups import atomic_json, file_hash, fingerprint
from runner.sweep import configurations
from scripts.sweep_report import collect

ROOT = Path(__file__).resolve().parents[1]


def record(pid, task='a', score=1., ttft=2.):
    return dict(prompt_id=pid, subtask=task, accuracy=score, thinking_tokens=8,
        answer_tokens=2, control_tokens=1, output_tokens=11,
        output_cap_reached=False, unfinished_thinking=False,
        timings=dict(ttft_seconds=ttft,answer_engine_ttft_seconds=ttft,routing_overhead_seconds=0.))


def commit(root, case, row, protocol_hash):
    folder=root/'records'/case/row['prompt_id']
    atomic_json(folder/'result.json',dict(row,method=case))
    atomic_json(folder/'validated.json',dict(complete=True,protocol_sha256=protocol_hash,
        files={'result.json':file_hash(folder/'result.json')}))


class MatchedStatusTests(unittest.TestCase):
    def test_intersection_not_minimum_count_or_order(self):
        rows=[record('p'),record('q'),record('r','b')]
        selected,metadata=match_records({'a':rows[:2], 'b':rows[1:]},rows)
        self.assertEqual(metadata['prompt_ids'],['q'])
        self.assertEqual(metadata['per_subtask'],{'a':1,'b':0})
        self.assertEqual(selected['a'],selected['b'])
        self.assertEqual(match_records({'a':rows,'b':[]},rows)[1]['matched_samples'],0)
        for bad in ([rows[0],rows[0]],[record('unknown')],[record('p','wrong')]):
            with self.assertRaises(ValueError):match_records({'a':bad},rows)

    def test_sweep_reaggregates_metrics_and_preserves_existing_reports(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest=root/'manifest.jsonl'
            rows=[record('p',score=0.,ttft=100.),record('q'),record('r','b')]
            manifest.write_text(''.join(json.dumps(dict(id=r['prompt_id'],subtask=r['subtask'],prepared='unused'))+'\n' for r in rows))
            configs=configurations([1]);before={}
            for config,rs in zip(configs,[rows,rows[1:],rows[1:2]]):
                dest=root/config['name']
                atomic_json(dest/'aggregation_state.json',dict(records=rs))
                atomic_json(dest/'live_aggregation.json',aggregate(rs,rows,
                    dict(method=config['method'],ratio=config['ratio'],sweep_fingerprint='fixture'),'running'))
                before[dest/'live_aggregation.json']=file_hash(dest/'live_aggregation.json')
            ordinary=collect(root,manifest,[1])
            before[root/'live_summary.json']=file_hash(root/'live_summary.json')
            matched=collect(root,manifest,[1],same_count=True)
            self.assertEqual(ordinary['completed'],6)
            self.assertEqual(matched['completed'],3)
            self.assertEqual(matched['matching']['prompt_ids'],['q'])
            for method in matched['methods'].values():
                self.assertEqual(method['overall']['mean_ttft_seconds'],2.)
                self.assertEqual(method['overall']['accuracy_percent'],100.)
                self.assertIsNone(method['subtasks']['b']['accuracy_percent'])
            self.assertTrue(all(file_hash(p)==h for p,h in before.items()))
            # A continuation compares only baseline and the continuing vanilla method.
            scope=dict(attempt='continuation/attempt-test',configurations=configurations([1],skip_selective=True),
                discontinued=['selective-1'],fingerprint='fixture')
            atomic_json(root/'continuation.json',scope)
            for config in scope['configurations']:
                dest=root/scope['attempt']/'reports'/config['name']
                for name in ('aggregation_state.json','live_aggregation.json'):
                    atomic_json(dest/name,json.loads((root/config['name']/name).read_text()))
            continued=collect(root,manifest,[1],same_count=True)
            self.assertEqual(continued['matching']['prompt_ids'],['q','r'])
            self.assertNotIn('selective-1',continued['methods'])

    def test_committed_reader_excludes_inflight_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            commit(root,'baseline',record('p'),'pin')
            atomic_json(root/'records'/'baseline'/'q'/'result.json',record('q'))
            self.assertEqual(len(committed_records(root,['baseline'],'pin')['baseline']),1)
            with self.assertRaises(ValueError):committed_records(root,['baseline'],'other')
            atomic_json(root/'records'/'baseline'/'p'/'result.json',record('p',score=0.))
            with self.assertRaises(ValueError):committed_records(root,['baseline'],'pin')

    def test_native_router_matches_counts_and_metrics_including_empty(self):
        from scripts.router_control import status
        from runner.router_sweep import CASES
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);protocol=dict(prepared=td)
            atomic_json(root/'protocol.json',protocol)
            (root/'manifest.jsonl').write_text(json.dumps(dict(id='p',subtask='a'))+'\n')
            for case in CASES[:-1]:commit(root,case,record('p'),fingerprint(protocol))
            with contextlib.redirect_stdout(io.StringIO()):empty=status(root,True)
            self.assertEqual(set(empty['answers'].values()),{0})
            self.assertIsNone(empty['methods']['baseline']['overall']['mean_ttft_seconds'])
            commit(root,CASES[-1],record('p'),fingerprint(protocol))
            with contextlib.redirect_stdout(io.StringIO()):matched=status(root,True)
            self.assertEqual(set(matched['answers'].values()),{1})
            self.assertEqual(matched['router_probes'],3)
            self.assertEqual(matched['methods']['baseline']['overall']['mean_answer_tokens'],2)

    def test_corpus_and_inference_recompute_paired_metrics(self):
        from scripts.corpus_control import snapshot_report
        from runner.corpus_records import protocol_identity
        from runner.tree_policy import DEFAULT_ACTIONS,actions
        for kind in ('collection','inference'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as td:
                root=Path(td)
                protocol=dict(prepared=td,kind=kind,dataset='ruler',hardware={},actions=DEFAULT_ACTIONS,selected_ids=['p','q'] if kind=='inference' else None)
                atomic_json(root/'protocol.json',protocol)
                atomic_json(root/'tranche.json',dict(limit_per_task=2))
                (root/'manifest.jsonl').write_text(''.join(json.dumps(dict(id=p,subtask='a',ordinal=i))+'\n' for i,p in enumerate(['p','q'])))
                cases=['nocache','router'] if kind=='inference' else list(actions(DEFAULT_ACTIONS))
                for case in cases:
                    row=record('q');row['decision']=dict(action='nocache')
                    commit(root,case,row,protocol_identity(protocol))
                commit(root,'nocache',record('p',score=0.,ttft=100.),protocol_identity(protocol))
                atomic_json(root/'records'/'probe'/'q'/'validated.json',dict(complete=True))
                atomic_json(root/'report.json',dict(original=True));before=file_hash(root/'report.json')
                with patch('scripts.corpus_control.verify',side_effect=AssertionError('No model verification in status')),patch('scripts.corpus_control.load',return_value={'provenance':{'hardware':{}}}):
                    report=snapshot_report(root,same_count=True)
                self.assertEqual(report['matching']['prompt_ids'],['q'])
                self.assertEqual(set(report['accepted_answers'].values()),{1})
                self.assertEqual(report['methods']['nocache']['overall']['accuracy'],1.)
                self.assertEqual(report['output_statistics']['nocache']['mean_thinking_tokens'],8)
                self.assertEqual(file_hash(root/'report.json'),before)

    def test_all_five_shell_launchers_dispatch_cpu_status(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fake=root/'python'
            fake.write_text(f'#!{sys.executable}\nimport json,os,sys\nprint(json.dumps([os.environ.get("CUDA_VISIBLE_DEVICES"),sys.argv[1:]]))\n')
            fake.chmod(0o755)
            (root/'server.env').write_text('')
            env=dict(os.environ,PYTHON_BIN=str(fake),UCM_ENV_FILE=str(root/'server.env'))
            for name in ('a800_longbench','l20_ruler','a800_router','ruler_corpus','router_infer'):
                result=subprocess.run(['bash',str(ROOT/'scripts'/f'{name}.sh'),'status_same_count'],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                visible,args=json.loads(result.stdout)
                self.assertEqual(visible,'')
                self.assertIn('--same-count' if name in ('a800_longbench','l20_ruler') else 'status_same_count',args)


if __name__=='__main__':unittest.main()
