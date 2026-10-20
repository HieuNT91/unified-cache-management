"""Committed-result reports: exact shared prompts and all thirteen task tables."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from runner.corpus_records import publish, record_dir
from runner.reporting import OutputAnalyzer
from runner.setups import atomic_json, file_hash
from runner.thinking_budget import KEY, BudgetState
from scripts import l20_ruler_traj4f_data as collection
from scripts.ruler import TASKS
from test_l20_traj4f import fixture, Tokenizer
from scripts import experiment_task_breakdown as breakdown


class L20ReportTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.args=fixture(Path(temp.name))
        with patch('scripts.router_control.code_hashes',return_value={}),redirect_stdout(io.StringIO()):
            collection.adopt(self.args)
        self.base=self.args.root
        self.protocol,self.rows=collection.stage(self.base,'ruler')
        self.cases=self.protocol['scheduled_actions']
        atomic_json(self.base/'ruler/init.json',dict(validated=True,engine_config={}))

    def commit(self,case,row,score,ttft):
        p=row[KEY]; tokens=[65,p['close_id']]+p['transition_ids']+p['prefix_ids']+[65,3]
        group=row['ordinal']%5
        record=dict(prompt_id=row['id'],method=case,executed_action=case,input_sha256=row['sha256'],group=group,
            gpu_uuids=self.protocol['groups'][group],cache_immutable=True,initialization='init.json',
            initialization_sha256=file_hash(self.base/'ruler/init.json'),
            answer_validation=self.protocol['answer_validation'],evaluation_protocol=row['evaluation_protocol'],
            retirement=[dict(rank=r,quiescent=True,transfers={'pending':0},request_bookkeeping=0) for r in range(2)],
            accuracy=score,subtask=row['subtask'],scoring='ruler_all',prediction='A',output_cap_reached=False,
            output_token_ids=tokens,max_output_tokens=16384,
            timings=dict(ttft_seconds=ttft,first_answer_content_seconds=ttft+.1),**{KEY:p})
        record.update(OutputAnalyzer(Tokenizer()).analyze(tokens,[],True),**BudgetState(p).consume(tokens).metrics(Tokenizer()))
        publish(self.base/'ruler',case,row,self.protocol,record,[])

    def test_exact_intersection_task_tables_and_readonly(self):
        first, second = self.rows[:2]
        for i, case in enumerate(self.cases):
            self.commit(case, first, 1., 2.)
            self.commit(case, second, 0., 4.)
            self.commit(case, self.rows[13+i%2], 1., 100.)
        atomic_json(record_dir(self.base/'ruler', self.cases[0], self.rows[2]['id'])/'result.json', {})
        pins = {p: (file_hash(p), p.stat().st_mtime_ns) for p in self.base.rglob('*') if p.is_file()}
        # Reporting must work even when the original prepared path is unavailable.
        with patch.object(collection, 'load', side_effect=AssertionError('No launcher loading')):
            raw_text, raw, _ = breakdown.report(self.base)
            text, tables, matching = breakdown.report(self.base, same_count=True)
        self.assertEqual(matching['matched_samples'], 2)
        self.assertEqual(set(matching['available_counts'].values()), {3})
        self.assertEqual(len(tables), 14)
        for case in self.cases:
            self.assertEqual(raw['overall'][case]['completed'], 3)
            self.assertAlmostEqual(raw['overall'][case]['mean_ttft_seconds'], 106/3)
            self.assertEqual(tables['overall'][case]['completed'], 2)
            self.assertEqual(tables['overall'][case]['expected'], 65)
            self.assertEqual(tables['overall'][case]['mean_ttft_seconds'], 3.)
            self.assertEqual(tables['overall'][case]['accuracy_percent'], 50.)
        for task in TASKS:
            self.assertIn('task/'+task, text)
            self.assertIn('task/'+task, raw_text)
        self.assertIn('N/A', text)
        self.assertEqual(pins, {p: (file_hash(p), p.stat().st_mtime_ns)
                               for p in self.base.rglob('*') if p.is_file()})

    def test_empty_methods_stay_in_intersection_and_corruption_fails(self):
        self.commit(self.cases[0], self.rows[0], 1., 2.)
        _, tables, matching = breakdown.report(self.base, same_count=True)
        self.assertEqual(matching['matched_samples'], 0)
        for value in tables['overall'].values():
            self.assertEqual(value['completed'], 0)
            self.assertIsNone(value['accuracy_percent'])
        body = record_dir(self.base/'ruler', self.cases[0], self.rows[0]['id'])/'result.json'
        body.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Corrupt'):
            breakdown.report(self.base)

    def test_cli_both_scripts_accept_only_experiment_dir(self):
        import subprocess
        root = Path(__file__).resolve().parents[1]
        for name in ('experiment_task_breakdown', 'experiment_same_count'):
            result = subprocess.run([sys.executable, str(root/'scripts'/f'{name}.py'), str(self.base)],
                                    cwd='/tmp', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('task/'+TASKS[-1], result.stdout)
            self.assertIn('0/520', result.stdout)

    def test_seeded_collection_paths_and_file_hash_plan(self):
        from scripts.collection_v3 import declaration
        protocol = dict(self.protocol, **declaration('ruler', 1, thinking=True),
                        plan_sha256=file_hash(self.base/'plan.json'))
        atomic_json(self.base/'ruler/protocol.json', protocol)
        row = self.rows[0]
        # Exercise L40's seed subdirectory and plan file hash convention;
        # actual committed result validation is covered by the other fixtures.
        for case in self.cases:
            atomic_json(record_dir(self.base/'ruler', case, row['id'], 0)/'validated.json', {})
        def result(root, case, source, saved, seed):
            self.assertEqual(seed, 0)
            return dict(prompt_id=source['id'], subtask=source['subtask'], seed=seed,
                        sequence_id=source['id'], accuracy=1., timings={'ttft_seconds': 2.},
                        thinking_tokens=0, answer_tokens=1, control_tokens=0, output_tokens=1,
                        output_cap_reached=False, unfinished_thinking=False)
        with patch('runner.corpus_records.accepted', side_effect=result):
            _, tables, matching = breakdown.report(self.base, same_count=True)
        self.assertEqual(matching['matched_samples'], 1)
        self.assertEqual(tables['overall'][self.cases[0]]['accuracy_percent'], 100.)


if __name__ == '__main__':
    unittest.main()
