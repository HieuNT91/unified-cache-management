"""Reproduction reporting preserves full coverage and exact shared-input identities."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runner.cache.setups import atomic_json, file_hash, fingerprint
from runner.reporting import run
import status


class StatusTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name) / 'results' / 'longbench-chunk2048'
        self.settings = dict(schema='tree-prophetkv-thinking-collection-v1',
                             dataset='longbench-v2', kv_chunk_size=2048)
        self.rows = [dict(id=f'prompt-{i}', subtask='task-' + str(i % 2),
                          token_sha256=str(i + 1) * 64) for i in range(3)]
        atomic_json(self.base / 'settings.json', self.settings)
        atomic_json(self.base / 'rows.json', self.rows)
        plan = dict(settings_sha256=file_hash(self.base / 'settings.json'),
                    rows_sha256=file_hash(self.base / 'rows.json'))
        atomic_json(self.base / 'plan.json', plan)
        self.protocol = dict(self.settings, collection_protocol='tp2-data-v3',
                             answers_per_question=2, answer_seeds=[0, 1],
                             scheduled_actions=['adatree-v1', 'nocache'],
                             plan_sha256=fingerprint(plan))
        atomic_json(self.base / 'tree/protocol.json', self.protocol)
        atomic_json(self.base / 'tree/init.json', dict(validated=True))

    def commit(self, case, index, seed=0, *, score=1., ttft=2., thinking=10, answer=4):
        row = self.rows[index]
        body = dict(prompt_id=row['id'], sequence_id=row['id'], seed=seed, method=case,
                    token_sha256=row['token_sha256'], subtask=row['subtask'], accuracy=score,
                    timings=dict(ttft_seconds=ttft), thinking_tokens=thinking,
                    answer_tokens=answer, control_tokens=2, output_tokens=thinking + answer + 2,
                    output_cap_reached=False, unfinished_thinking=False,
                    initialization='init.json', initialization_sha256=file_hash(self.base / 'tree/init.json'))
        folder = self.base / 'tree/records' / case / row['id'] / f'seed-{seed}'
        atomic_json(folder / 'result.json', body)
        atomic_json(folder / 'validated.json', dict(files={'result.json': file_hash(folder / 'result.json')}))
        return folder

    def accepted(self, root, case, row, protocol, *, seed):
        self.assertEqual(root, self.base / 'tree')
        self.assertEqual(protocol, self.protocol)
        return json.loads((root / 'records' / case / row['id'] / f'seed-{seed}/result.json').read_text())

    def test_default_full_and_same_input_have_distinct_coverage(self):
        self.commit('adatree-v1', 0, score=1., ttft=2., thinking=8, answer=2)
        self.commit('adatree-v1', 1, score=0., ttft=8., thinking=12, answer=6)
        self.commit('nocache', 0, score=0., ttft=4., thinking=20, answer=8)
        self.commit('nocache', 2, score=1., ttft=10., thinking=30, answer=10)
        before = {path: (file_hash(path), path.stat().st_mtime_ns)
                  for path in self.base.rglob('*') if path.is_file()}
        with patch('runner.corpus.records.accepted', side_effect=self.accepted):
            full_text, full, matching = run.report(self.base)
            paired_text, paired, _ = run.report(self.base, mode='same-input')
        self.assertIn('mode: full', full_text)
        self.assertIn('chunk size: 2048', full_text)
        self.assertEqual(matching['matched_samples'], 1)
        self.assertEqual(set(full), {'overall', 'task/task-0', 'task/task-1'})
        route = full['overall']['adatree-v1']
        self.assertEqual(route['completed'], 2)
        self.assertEqual(route['expected'], 6)
        self.assertEqual(route['accuracy_percent'], 50.)
        self.assertEqual(route['mean_ttft_seconds'], 5.)
        self.assertEqual(route['mean_thinking_tokens'], 10.)
        self.assertEqual(route['mean_answer_tokens'], 4.)
        self.assertEqual(paired['overall']['adatree-v1']['accuracy_percent'], 100.)
        self.assertEqual(paired['overall']['adatree-v1']['completed'], 1)
        self.assertEqual(paired['overall']['nocache']['mean_ttft_seconds'], 4.)
        self.assertIn('| adatree-v1 | 2 | 1/6 |', paired_text)
        self.assertEqual(before, {path: (file_hash(path), path.stat().st_mtime_ns) for path in before})

    def test_empty_method_and_uncommitted_result_remain_incomplete(self):
        self.commit('adatree-v1', 0)
        folder = self.commit('nocache', 0)
        (folder / 'validated.json').unlink()
        with patch('runner.corpus.records.accepted', side_effect=self.accepted):
            _, full, _ = run.report(self.base)
            text, paired, matching = run.report(self.base, mode='same-input')
        self.assertEqual(full['overall']['adatree-v1']['completed'], 1)
        self.assertEqual(full['overall']['nocache']['completed'], 0)
        self.assertEqual(matching['matched_samples'], 0)
        self.assertTrue(all(value['completed'] == 0 for value in paired['overall'].values()))
        self.assertIn('N/A', text)

    def test_seed_is_part_of_intersection(self):
        self.commit('adatree-v1', 0, seed=0)
        self.commit('nocache', 0, seed=1)
        with patch('runner.corpus.records.accepted', side_effect=self.accepted):
            _, _, matching = run.report(self.base, mode='same-input')
        self.assertEqual(matching['matched_samples'], 0)

    def test_arbitrary_seed_order_reports_committed_answers(self):
        self.protocol.update(answers_per_question=3, answer_seeds=[42, 0, 7])
        atomic_json(self.base / 'tree/protocol.json', self.protocol)
        for seed in (42, 0, 7):
            self.commit('adatree-v1', 0, seed=seed)
            self.commit('nocache', 0, seed=seed)
        with patch('runner.corpus.records.accepted', side_effect=self.accepted):
            for mode in ('full', 'same-input'):
                _, tables, matching = run.report(self.base, mode=mode)
                self.assertEqual(matching['matched_samples'], 3)
                for value in tables['overall'].values():
                    self.assertEqual(value['completed'], 3)
                    self.assertEqual(value['expected'], 9)
                    self.assertEqual(value['accuracy_percent'], 100.)

    def test_changed_token_identity_fails_in_both_modes(self):
        self.commit('adatree-v1', 0)
        self.commit('nocache', 0)
        original = self.accepted

        def changed(*args, **kwargs):
            record = original(*args, **kwargs)
            if args[1] == 'nocache':
                record['token_sha256'] = 'f' * 64
            return record

        for mode in ('full', 'same-input'):
            with patch('runner.corpus.records.accepted', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'token identity'):
                    run.report(self.base, mode=mode)

    def test_invalid_unmatched_metric_and_changed_result_are_rejected(self):
        folder = self.commit('adatree-v1', 0)
        self.commit('nocache', 1)
        original = self.accepted

        def invalid(*args, **kwargs):
            record = original(*args, **kwargs)
            record['thinking_tokens'] = -1
            return record

        with patch('runner.corpus.records.accepted', side_effect=invalid):
            with self.assertRaisesRegex(ValueError, 'token counts'):
                run.report(self.base, mode='same-input')
        body = json.loads((folder / 'result.json').read_text())
        body['accuracy'] = 0.
        atomic_json(folder / 'result.json', body)
        with patch('runner.corpus.records.accepted', side_effect=self.accepted):
            with self.assertRaisesRegex(ValueError, 'changed during reporting'):
                run.report(self.base)

    def test_metadata_hash_and_no_baseline_schedule(self):
        self.protocol['scheduled_actions'] = ['adatree-v1']
        atomic_json(self.base / 'tree/protocol.json', self.protocol)
        text, tables, _ = run.report(self.base)
        self.assertIn('No-cache is absent', text)
        self.assertEqual(set(tables['overall']), {'adatree-v1'})
        atomic_json(self.base / 'rows.json', [])
        with self.assertRaisesRegex(ValueError, 'Changed rows'):
            run.report(self.base)

    def test_cli_defaults_full_and_scans_runs_without_entering_artifacts(self):
        artifact = self.base / 'tree/records/nested/settings.json'
        atomic_json(artifact, {})
        self.assertEqual(status.runs(self.base.parent), [self.base])
        with patch('status.report', return_value=('report\n', {}, {})) as mocked:
            output = io.StringIO()
            with redirect_stdout(output):
                status.main([str(self.base.parent)])
            mocked.assert_called_once_with(self.base, mode='full')
            self.assertEqual(output.getvalue(), 'report\n')

    def test_real_committed_answers_validate_policy_and_report_without_prepared_inputs(self):
        from runner.corpus.records import publish
        from runner.inputs.layout import sample_provenance
        from runner.reporting.aggregate import OutputAnalyzer
        from runner.router.adatree import generation_policy
        from runner.runtime.thinking_budget import KEY, BudgetState
        from runner.runtime.thinking_loop import ADATREE_CONFIG
        from scripts.collection import run_controller as collection
        from run_fixtures import fixture
        from test_ruler_thinking import Tokenizer
        from ucm.sparse.prophetkv.adatree import protocol_for

        for dataset in ('ruler', 'longbench-v2'):
            for chunk in (4096, 2048):
                with self.subTest(dataset=dataset, chunk=chunk), tempfile.TemporaryDirectory() as temp:
                    args = fixture(Path(temp), dataset, chunk)
                    collection.adopt(args)
                    protocol, rows = collection.stage(args.root, 'tree')
                    row = rows[0]
                    sample = json.loads((args.prepared / row['prepared']).read_text())
                    root = args.root / 'tree'
                    atomic_json(root / 'init.json', dict(validated=True, engine_config={}))
                    policy = generation_policy(row[KEY], protocol)
                    tokens = [65, policy['close_id']] + policy['transition_ids'] + policy['prefix_ids'] + [65, 3]
                    for case in protocol['scheduled_actions']:
                        record = dict(sample_provenance(sample), prompt_id=row['id'], sequence_id=row['id'],
                                      seed=0, method=case, executed_action=case, input_sha256=row['sha256'],
                                      group=0, gpu_uuids=protocol['groups'][0], cache_immutable=True,
                                      num_cached_tokens=0, initialization='init.json',
                                      initialization_sha256=file_hash(root / 'init.json'),
                                      answer_validation=protocol['answer_validation'],
                                      retirement=[dict(rank=r, quiescent=True, transfers={'pending': 0},
                                                       request_bookkeeping=0) for r in range(protocol['tp'])],
                                      accuracy=1., subtask=row['subtask'], scoring=row['scoring'],
                                      prediction='A', output_cap_reached=False, output_token_ids=tokens,
                                      max_output_tokens=10000, prepared_generation_policy=row[KEY],
                                      timings=dict(ttft_seconds=2. if case == 'adatree-v1' else 4.,
                                                   first_answer_content_seconds=5.))
                        record.update(OutputAnalyzer(Tokenizer()).analyze(tokens, [], True))
                        record.update(BudgetState(policy, ADATREE_CONFIG).consume(tokens).metrics(Tokenizer()))
                        record[KEY] = policy
                        if case == 'adatree-v1':
                            features = ({'G5_mean': .3, 'S60_p90': .7} if chunk == 4096 else
                                        {'D60_p90': .7, 'T5_1_mean': .7, 'D20_mean': .6})
                            record.update(executed_action='tree-prophetkv-5', independent_probes=0,
                                          routing=dict(schema=protocol_for(chunk), features=features,
                                                       action='tree-prophetkv-5', ratio=.05, chunk_size=chunk,
                                                       independent_probes=0, feature_mask_only=True,
                                                       included_in_native_probe=True, request_id='request'))
                        publish(root, case, row, protocol, record, [], seed=0)
                    args.prepared.rename(args.prepared.with_name('unavailable-prepared'))
                    text, tables, matching = run.report(args.root)
                    self.assertEqual(matching['matched_samples'], 1)
                    self.assertEqual(tables['overall']['adatree-v1']['mean_ttft_seconds'], 2.)
                    self.assertEqual(tables['overall']['nocache']['mean_ttft_seconds'], 4.)
                    self.assertNotIn('followup/', text)
                    body = root / 'records/nocache' / row['id'] / 'seed-0/result.json'
                    record = json.loads(body.read_text())
                    record['thinking_loop_breaker'] = dict(ADATREE_CONFIG, min_count=3)
                    atomic_json(body, record)
                    receipt_path = body.with_name('validated.json')
                    receipt = json.loads(receipt_path.read_text())
                    receipt['files']['result.json'] = file_hash(body)
                    atomic_json(receipt_path, receipt)
                    with self.assertRaisesRegex(ValueError, 'loop breaker differs'):
                        run.report(args.root)

    def test_legacy_run_marker_is_readable_without_rewriting_saved_metadata(self):
        from scripts.collection import run_controller as collection
        from run_fixtures import fixture

        for chunk in (4096, 2048):
            with self.subTest(chunk=chunk), tempfile.TemporaryDirectory() as temp:
                args = fixture(Path(temp), 'longbench-v2', chunk)
                collection.adopt(args)
                settings, plan, _ = collection.load(args.root)
                protocol, _ = collection.stage(args.root, 'tree')
                settings['reproduction'] = settings.pop('run_profile')
                protocol['reproduction'] = protocol.pop('run_profile')
                atomic_json(args.root / 'settings.json', settings)
                plan['settings_sha256'] = file_hash(args.root / 'settings.json')
                atomic_json(args.root / 'plan.json', plan)
                protocol['plan_sha256'] = fingerprint(plan)
                atomic_json(args.root / 'tree/protocol.json', protocol)
                pins = {p: (file_hash(p), p.stat().st_mtime_ns) for p in (
                    args.root / 'settings.json', args.root / 'rows.json',
                    args.root / 'plan.json', args.root / 'tree/protocol.json')}
                collection.load(args.root)
                collection.stage(args.root, 'tree')
                collection.adopt(args)
                text, tables, _ = run.report(args.root)
                self.assertIn('mode: full', text)
                self.assertEqual(set(tables['overall']), {'adatree-v1', 'nocache'})
                self.assertEqual(pins, {p: (file_hash(p), p.stat().st_mtime_ns) for p in pins})


if __name__ == '__main__':
    unittest.main()
