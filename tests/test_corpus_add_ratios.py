"""CPU acceptance for in-place action extension; no model/GPU execution."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from runner.corpus import selection
from runner.corpus_records import accepted, publish, record_dir, protocol_identity
from runner.setups import atomic_json, file_hash
from runner.tree_policy import DEFAULT_ACTIONS
from scripts import corpus_add_ratios as extension
from test_corpus import archives, tiny_sample


def diagnostics(sample, ratio=None):
    prefix, end, length = sample['boundaries'][1], sample['boundaries'][-2], len(sample['token_ids'])
    scores = np.full(end-prefix, 1/end, np.float32)
    chosen = selection(scores, prefix, ratio).tolist() if ratio is not None else []
    start = prefix if ratio is not None else 0
    positions = chosen+list(range(end, length)) if ratio is not None else list(range(length))
    events = [dict(kind='prefill_step', start=start, end=length, scheduled_tokens=length-start,
                   recomputed_tokens=len(positions), prefill_complete=True, no_forward=False)]
    if ratio is not None:
        events.append(dict(kind='prophetkv_selection', scores=scores.tolist(), selected_positions=chosen,
                           scoring_layers=list(range(64)), fusion='mean_layers_fp32', alignment_count=64))
        events.extend(dict(kind='layer_counts', start=start, end=length,
                           layer=f'model.layers.{i}.self_attn.attn', selected_positions=positions,
                           selected_set_verified=True, projection_tokens=len(positions),
                           attention_tokens=len(positions), ffn_tokens=len(positions)) for i in range(64))
    return [dict(rank=i, diagnostics=copy.deepcopy(events)) for i in range(4)]


def result(row, case, protocol, initialization, digest):
    return dict(prompt_id=row['id'], method=case, executed_action=case, subtask=row['subtask'],
        input_sha256=row['sha256'], group=row['ordinal'] % len(protocol['groups']),
        gpu_uuids=protocol['groups'][row['ordinal'] % len(protocol['groups'])],
        cache_immutable=True, initialization=initialization, initialization_sha256=digest,
        retirement=[dict(rank=i, quiescent=True, transfers={'pending': 0}, request_bookkeeping=0) for i in range(4)],
        num_cached_tokens=0, accuracy=1., timings=dict(answer_engine_ttft_seconds=2., ttft_seconds=2.5),
        thinking_tokens=0, answer_tokens=3, control_tokens=1, output_tokens=4,
        output_cap_reached=False, unfinished_thinking=False)


class AddRatiosTests(unittest.TestCase):
    def fixture(self, root, groups=1):
        prepared = root/'prepared'
        state = root/extension.STATE
        runtime = state/'frozen-code'
        runtime.mkdir(parents=True)
        (runtime/'stub.py').write_text('# synthetic CPU fixture\n')
        code = {'stub.py': file_hash(runtime/'stub.py')}
        atomic_json(runtime/'runtime.json', dict(base_code=code, files=code))
        atomic_json(prepared/'plan.json', {'synthetic': True})
        protocol = dict(kind='collection', dataset='ruler', actions=DEFAULT_ACTIONS,
            prepared=str(prepared), model='/not-used', cache_root=str(root/'cache'), code=code,
            plan_sha256=file_hash(prepared/'plan.json'),
            groups=[[f'GPU-{i:08x}-0000-0000-0000-000000000000' for i in range(g*4, (g+1)*4)] for g in range(groups)])
        atomic_json(root/'protocol.json', protocol)
        atomic_json(root/'settings.json', protocol)
        atomic_json(root/'tranche.json', dict(limit_per_task=2))
        atomic_json(root/'supervisor.json', dict(state='partial-complete', pid=-1, identity=None))
        atomic_json(root/'cleanup.json', dict(complete=True, owned_engines_exited=True))
        atomic_json(root/'initialization.json', dict(validated=True, engine_config={}))
        rows = []
        for ordinal in range(3):  # A prepared third row must NEVER enter the two-row tranche.
            pid = f'task-{ordinal:03d}'
            sample = dict(tiny_sample(), thinking=False, max_output_tokens=256)
            path = prepared/f'{pid}.json'
            atomic_json(path, sample)
            raw = prepared/f'{pid}-raw.json'
            atomic_json(raw, dict(source=pid))
            row = dict(id=pid, ordinal=ordinal, subtask='task', prepared=path.name, sha256=file_hash(path),
                       raw=raw.name, raw_sha256=file_hash(raw), references=['answer'], scoring='ruler_all')
            atomic_json(prepared/'generation'/pid/'receipt.json', dict(row=row))
            rows.append(row)
            if ordinal >= 2:
                continue
            for case in ['probe', *(a['id'] for a in DEFAULT_ACTIONS)]:
                record = result(row, case, protocol, 'initialization.json', file_hash(root/'initialization.json'))
                if case == 'probe':
                    folder = record_dir(root, case, pid)
                    folder.mkdir(parents=True)
                    record.update(archives(folder), internal_tokens=1)
                    ratio = .01
                else:
                    ratio = next(a['ratio'] for a in DEFAULT_ACTIONS if a['id'] == case)
                publish(root, case, row, protocol, record, diagnostics(sample, ratio))
        atomic_json(root/'partial-2-validation.json', dict(status='partial-complete', tranche_complete=True,
            scheduled_samples=2, accepted_answers={a['id']: 2 for a in DEFAULT_ACTIONS}, accepted_probes=2, answers=8))
        with patch('scripts.corpus_inputs.prepared_rows', return_value=rows), patch('scripts.corpus_inputs.TASKS', ('task',)), \
                patch('runner.tree_profiles.validate'), patch.object(extension, 'environment_check'):
            corpus = extension.prepare(root, runtime)
        return corpus, rows

    def fake_engine(self, events):
        class Engine:
            def __init__(self, root, protocol, group, phase, attempt, rows):
                events.append(('engine', group))
                self.root, self.protocol = root, protocol
                self.session = root/'sessions'/f'cached-group{group}-{attempt}'
                atomic_json(self.session/'initialization.json', dict(validated=True, engine_config={}))

            def begin(self, row):
                self.row = row
                events.append(('begin', row['id']))

            def answer(self, definition, case, attention):
                events.append(('answer', case, self.row['id']))
                path = self.session/'initialization.json'
                record = result(self.row, case, self.protocol, str(path.relative_to(self.root)), file_hash(path))
                sample = extension.read(Path(self.protocol['prepared'])/self.row['prepared'])
                expected = selection(attention['scores'], sample['boundaries'][1], definition['ratio'])
                np.testing.assert_array_equal(expected, attention['selections'][case])
                return record, diagnostics(sample, definition['ratio'])

            def end(self):
                events.append('delete')
                return dict(deleted_shards=8)

            def close(self):
                events.append('close')
        return Engine

    def collect(self, corpus, group=0, events=None, attempt='fixture'):
        events = [] if events is None else events
        with patch('runner.corpus_runtime.Engine', self.fake_engine(events)), \
                patch('runner.setups.check_environment', return_value=corpus.protocol['groups'][group]):
            extension.collect(corpus.root, group, attempt)
        return events

    def test_exact_tranche_and_original_artifacts_are_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            corpus, rows = self.fixture(Path(td))
            self.assertEqual([r['id'] for r in corpus.rows], [r['id'] for r in rows[:2]])
            pins = dict(corpus.sources['files'])
            events = self.collect(corpus)
            self.assertEqual([e for e in events if isinstance(e, tuple) and e[0] == 'answer'],
                [('answer', action, row['id']) for row in rows[:2] for action in extension.EXTRA])
            self.assertEqual(events.count('delete'), 2)
            self.assertEqual(events.count('close'), 1)
            self.assertEqual([e for e in events if isinstance(e, tuple) and e[0] == 'engine'], [('engine', 0)])
            for row in rows[:2]:
                for action in extension.EXTRA:
                    self.assertIsNotNone(corpus.outcome(row, action, full=True))
                    self.assertTrue((corpus.root/'records'/action/row['id']/'validated.json').is_file())
            self.assertFalse((corpus.state/'records').exists())
            self.assertEqual(pins, {name: file_hash(name) for name in pins})
            self.assertEqual(corpus.pending(0), [])
            self.assertEqual(self.collect(corpus, attempt='empty'), [])

    def test_two_groups_retain_original_uuid_and_ordinal_assignments(self):
        with tempfile.TemporaryDirectory() as td:
            corpus, rows = self.fixture(Path(td), groups=2)
            self.assertEqual(corpus.pending(0), [rows[0]])
            self.assertEqual(corpus.pending(1), [rows[1]])
            with patch('runner.setups.check_environment', return_value=corpus.protocol['groups'][1]), \
                    patch('runner.corpus_runtime.Engine') as engine:
                with self.assertRaisesRegex(ValueError, 'UUID'):
                    extension.collect(corpus.root, 0, 'bad')
                engine.assert_not_called()
            self.collect(corpus, group=0)
            self.collect(corpus, group=1)
            for row in rows[:2]:
                record = corpus.outcome(row, extension.EXTRA[0], full=True)
                self.assertEqual(record['group'], row['ordinal'])
                self.assertEqual(record['gpu_uuids'], corpus.protocol['groups'][row['ordinal']])

    def test_missing_only_resume_and_incomplete_attempt_archival(self):
        with tempfile.TemporaryDirectory() as td:
            corpus, rows = self.fixture(Path(td), groups=2)
            self.collect(corpus, group=0)
            # Simulate an interrupted second action, leaving its unaccepted payload.
            folder = corpus.root/'records'/extension.EXTRA[1]/rows[0]['id']
            (folder/'validated.json').unlink()
            retained = file_hash(corpus.root/'records'/extension.EXTRA[0]/rows[0]['id']/'validated.json')
            events = self.collect(corpus, group=0, attempt='resume')
            self.assertEqual([e for e in events if isinstance(e, tuple) and e[0] == 'answer'],
                             [('answer', extension.EXTRA[1], rows[0]['id'])])
            self.assertEqual(retained, file_hash(corpus.root/'records'/extension.EXTRA[0]/rows[0]['id']/'validated.json'))
            self.assertEqual(len(list((corpus.state/'incomplete').glob('*'))), 1)

    def test_replay_exact_floors_ties_and_corrupt_attention(self):
        with tempfile.TemporaryDirectory() as td:
            corpus, rows = self.fixture(Path(td))
            attention = corpus.attention(rows[0], replay=True)
            self.assertEqual(attention['selections']['prophetkv-5'].tolist(), [64, 65, 66])
            self.assertEqual(attention['selections']['prophetkv-10'].tolist(), list(range(64, 70)))
            archive = corpus.root/'records'/'probe'/rows[0]['id']/'attention.rank0.npz'
            archive.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'artifact'):
                corpus.attention(rows[0], replay=True)

    def test_changed_input_runtime_and_original_protocol_are_rejected(self):
        for kind in ('input', 'runtime', 'protocol'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as td:
                corpus, rows = self.fixture(Path(td))
                path = {'input': Path(corpus.protocol['prepared'])/rows[0]['prepared'],
                        'runtime': corpus.state/'frozen-code'/'stub.py',
                        'protocol': corpus.root/'protocol.json'}[kind]
                path.write_text('{}')
                with self.assertRaises(ValueError):
                    if kind == 'protocol':
                        extension.Extension(corpus.root)
                    else:
                        corpus.verify_sources()

    def test_new_records_use_their_own_protocol_and_reject_unexpected_ids(self):
        with tempfile.TemporaryDirectory() as td:
            corpus, rows = self.fixture(Path(td))
            self.collect(corpus)
            self.assertIsNotNone(accepted(corpus.root, 'nocache', rows[0], corpus.base_protocol))
            with self.assertRaisesRegex(ValueError, 'protocol'):
                accepted(corpus.root, extension.EXTRA[0], rows[0], corpus.base_protocol)
            atomic_json(corpus.root/'records'/extension.EXTRA[0]/'unexpected'/'validated.json', {})
            with self.assertRaisesRegex(ValueError, 'Unexpected'):
                corpus.validate_new()

    def test_combined_status_matching_completion_and_immutable_reports(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            corpus, _ = self.fixture(Path(td), groups=2)
            report = extension.summarize(corpus.root)
            self.assertEqual((report['accepted_answers'], report['expected_answers']), (8, 12))
            self.assertEqual(report['accepted_probes'], 2)
            self.collect(corpus, group=0)
            matched = extension.summarize(corpus.root, same_count=True)
            self.assertEqual(matched['matching']['prompt_ids'], ['task-000'])
            self.assertEqual(matched['reports']['prophetkv-5']['overall']['mean_ttft_seconds'], 2.)
            atomic_json(corpus.state/'cleanup.json', dict(owned_engines_exited=True))
            with self.assertRaisesRegex(ValueError, 'Missing answers'):
                extension.finalize(corpus.root)
            self.collect(corpus, group=1)
            done = extension.finalize(corpus.root)
            self.assertEqual((done['new_answers'], done['answers'], done['reused_probes']), (4, 12, 2))
            self.assertEqual(done, extension.finalize(corpus.root))
            self.assertFalse((corpus.root/'report.json').exists())
            (corpus.state/'report.md').write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'Pinned file'):
                extension.finalize(corpus.root)

    def test_finalization_requires_exit_receipt_and_dead_owned_engines(self):
        with tempfile.TemporaryDirectory() as td:
            corpus, _ = self.fixture(Path(td))
            atomic_json(corpus.state/'processes'/'worker.json', dict(pid=os.getpid(), identity=None))
            with patch('runner.router_process.alive', return_value=True):
                with self.assertRaisesRegex(ValueError, 'remain alive'):
                    extension.finalize(corpus.root)
            (corpus.state/'processes'/'worker.json').unlink()
            atomic_json(corpus.state/'cleanup.json', dict(owned_engines_exited=False))
            with self.assertRaisesRegex(ValueError, 'exit receipt'):
                extension.finalize(corpus.root)

    def test_completion_accepts_120_tranche_and_rejects_incomplete_or_failed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            protocol = dict(actions=DEFAULT_ACTIONS)
            done = dict(status='partial-complete', tranche_complete=True, scheduled_samples=1560,
                        accepted_answers={a['id']:1560 for a in DEFAULT_ACTIONS}, accepted_probes=1560, answers=6240)
            atomic_json(root/'partial-1560-validation.json', done)
            atomic_json(root/'cleanup.json', dict(owned_engines_exited=True, complete=True))
            atomic_json(root/'supervisor.json', dict(state='partial-complete'))
            self.assertEqual(extension.completion(root, protocol, 1560), 'partial-1560-validation.json')
            atomic_json(root/'supervisor.json', dict(state='failed'))
            with self.assertRaisesRegex(ValueError, 'successfully exited'):
                extension.completion(root, protocol, 1560)
            done['accepted_answers']['prophetkv-40'] -= 1
            atomic_json(root/'partial-1560-validation.json', done)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                extension.completion(root, protocol, 1560)

    def test_tranche_selection_all_13_tasks_and_missing_row(self):
        tasks = tuple(f't{i}' for i in range(13))
        rows = {f'{t}-{i}':dict(id=f'{t}-{i}', ordinal=i, subtask=t) for i in range(200) for t in tasks}
        base = SimpleNamespace(rows=rows)
        selected = extension.selected_rows(base, 120, tasks)
        self.assertEqual(len(selected), 1560)
        self.assertEqual(len(selected)*2, 3120)
        self.assertEqual(len(selected)*len(extension.combined_actions(DEFAULT_ACTIONS)), 9360)
        del rows['t0-119']
        with self.assertRaisesRegex(ValueError, 'Missing'):
            extension.selected_rows(base, 120, tasks)
        with self.assertRaises(ValueError):
            extension.combined_actions(extension.combined_actions(DEFAULT_ACTIONS))

    def test_base_lock_excludes_original_and_duplicate_launches(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with extension.base_exclusive(root):
                with self.assertRaises((ValueError, BlockingIOError)):
                    with extension.base_exclusive(root):
                        self.fail('Acquired duplicate base lock')
                from scripts.corpus_control import idle
                with self.assertRaisesRegex(ValueError, 'lock is held'):
                    idle(root)

    def test_bootstrap_freezes_only_pinned_code_and_rejects_active_base(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/'results'
            source = Path(td)/'original'
            source.mkdir()
            (source/'run.py').write_text('# pinned original\n')
            (source/'unrelated.py').write_text('# must not enter frozen runtime\n')
            protocol = dict(actions=DEFAULT_ACTIONS, code={'run.py': file_hash(source/'run.py')})
            atomic_json(root/'protocol.json', protocol)
            atomic_json(root/'tranche.json', dict(limit_per_task=1))
            atomic_json(root/'partial-13-validation.json', dict(status='partial-complete', tranche_complete=True,
                scheduled_samples=13, accepted_answers={a['id']:13 for a in DEFAULT_ACTIONS}, accepted_probes=13, answers=52))
            atomic_json(root/'cleanup.json', dict(complete=True, owned_engines_exited=True))
            from runner.router_process import identity
            atomic_json(root/'supervisor.json', dict(pid=os.getpid(), identity=identity(os.getpid()), state='partial-complete'))
            with self.assertRaisesRegex(ValueError, 'still alive'):
                extension.bootstrap(root, source)
            self.assertFalse((root/extension.STATE/'frozen-code').exists())
            atomic_json(root/'supervisor.json', dict(pid=-1, identity=None, state='partial-complete'))
            runtime = extension.bootstrap(root, source)
            extension.verify_runtime(runtime, protocol['code'])
            self.assertFalse((runtime/'unrelated.py').exists())
            self.assertEqual(file_hash(runtime/'run.py'), protocol['code']['run.py'])
            self.assertEqual(file_hash(runtime/extension.CONTROLLER), file_hash(extension.__file__))
            # The frozen controller has a real working CLI independent of cwd.
            check = subprocess.run([sys.executable, str(runtime/extension.CONTROLLER), '--help'],
                                   cwd='/tmp', capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)
            self.assertIn('status_same_count', check.stdout)
            (source/'run.py').write_text('# later checkout edits cannot change frozen execution\n')
            self.assertEqual(extension.bootstrap(root, source), runtime)
            (runtime/'run.py').write_text('# corrupt\n')
            with self.assertRaisesRegex(ValueError, 'Pinned file'):
                extension.verify_runtime(runtime, protocol['code'])

    def test_no_commit_until_cache_deletion_succeeds(self):
        with tempfile.TemporaryDirectory() as td:
            corpus, _ = self.fixture(Path(td))
            engine = self.fake_engine([])
            with patch.object(engine, 'end', return_value={'deleted_shards': 0}), \
                    patch('runner.corpus_runtime.Engine', engine), \
                    patch('runner.setups.check_environment', return_value=corpus.protocol['groups'][0]):
                with self.assertRaisesRegex(ValueError, 'cannot be committed'):
                    extension.collect(corpus.root, 0, 'failure')
            for case in extension.EXTRA:
                self.assertFalse((corpus.root/'records'/case).exists())

    def test_supervisor_one_operational_retry_but_no_validation_retry(self):
        for codes, success in (([75, 0], True), ([76], False), ([75, 75], False)):
            with self.subTest(codes=codes), tempfile.TemporaryDirectory() as td:
                corpus, _ = self.fixture(Path(td))
                launches = []
                expected = list(codes)

                class Child:
                    def __init__(self, cmd, **kwargs):
                        self.pid = 8000000+len(launches)
                        self.code = expected[len(launches)]
                        launches.append(cmd)
                    def poll(self):
                        return self.code
                    def wait(self, **kwargs):
                        return self.code

                runtime = corpus.state/'frozen-code'
                with patch.object(extension, 'prepare', return_value=corpus), \
                        patch('scripts.corpus_control.check_hardware'), \
                        patch('scripts.router_control.cleanup_caches'), \
                        patch('scripts.router_control.terminate_owned') as terminate, \
                        patch('runner.router_process.identity', return_value={'start': '1', 'cmd': 'fixture'}), \
                        patch('runner.router_process.group_alive', return_value=False), \
                        patch.object(extension, 'assert_engines_exited'), \
                        patch.object(extension, 'finalize') as finalize, \
                        patch.object(extension.subprocess, 'Popen', Child), \
                        patch.object(extension.signal, 'signal'), \
                        patch.object(extension.time, 'sleep'):
                    if success:
                        extension.supervise(corpus.root, runtime)
                        finalize.assert_called_once_with(corpus.root)
                        self.assertTrue(extension.read(corpus.state/'cleanup.json')['owned_engines_exited'])
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'failed'):
                            extension.supervise(corpus.root, runtime)
                        finalize.assert_not_called()
                    terminate.assert_not_called()
                self.assertEqual(len(launches), len(codes))
                self.assertTrue(all('worker' in cmd for cmd in launches))
                self.assertEqual(extension.read(corpus.state/'supervisor.json')['state'], 'complete' if success else 'failed')

    def test_shell_uses_env_and_explicit_root_precedence(self):
        script = Path(__file__).resolve().parents[1]/'scripts/ruler_corpus_add_ratios.sh'
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            python = root/'fake-python'
            python.write_text('#!/usr/bin/env python3\nimport json,os,sys\nprint(json.dumps(dict(argv=sys.argv[1:],cuda=os.environ["CUDA_VISIBLE_DEVICES"])))\n')
            python.chmod(0o755)
            envfile = root/'server.env'
            envfile.write_text(f'PYTHON_BIN="{python}"\nEXPERIMENT_DIR="{root}/env root"\n')
            env = {k:v for k,v in os.environ.items() if k not in ('PYTHON_BIN', 'EXPERIMENT_DIR')}
            env.update(UCM_ENV_FILE=str(envfile), CUDA_VISIBLE_DEVICES='must-be-cleared')
            run = subprocess.run(['bash', str(script), 'verify', '--root', str(root/'explicit root')],
                                 env=env, capture_output=True, text=True, check=True)
            output = json.loads(run.stdout)
            self.assertEqual(output['cuda'], '')
            self.assertEqual(output['argv'][-2:], ['--root', str(root/'explicit root')])
            self.assertIn(str(root/'env root'), output['argv'])


if __name__ == '__main__':
    unittest.main()
