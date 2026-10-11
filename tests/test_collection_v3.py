"""Synthetic CPU fixtures for nine-action multi-answer collection; no model/data."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stdout
import numpy as np

from runner.setups import atomic_json, file_hash
from runner.corpus_records import accepted, publish, record_dir
from runner.thinking_budget import KEY, BudgetState, validate_thinking_outcome
from runner import output_budget as budget
from scripts import collection_v3 as v3
from scripts import longbench_a800_data as data
from scripts import longbench_a800_control as control
from scripts import collection_v3_export as export
from scripts import collection_v3_report as report
from scripts import export_features as ef
from test_ruler_thinking import Tokenizer, raw_row
from scripts.ruler import TASKS, CAPS, format_thinking_sample

ROOT = Path(__file__).resolve().parents[1]


def fixture(root, dataset='longbench-v2', thinking=False, answers=4):
    from runner.longbench_features import DEFINITIONS
    settings = dict(schema='tp2-data-config-v3', dataset=dataset, tp=2, seed=42,
                    model='/missing/model', data='/missing/data', prepared='/missing/prepared', cache_root='/missing/cache',
                    samples_per_task=500, code={}, hardware_profile='l40-tp2' if thinking else 'server')
    settings.update(v3.declaration(dataset, answers if dataset == 'longbench-v2' else 1, thinking))
    if thinking:
        settings.update(execution_profile='ruler-thinking', evaluation_protocol=budget.RULER_PROTOCOL)
    rows = []
    for task, i in ([('domain', i) for i in range(503)] if dataset == 'longbench-v2' else [(t, i) for t in TASKS for i in range(500)]):
        row = dict(id=f'{task}-{i:03d}', ordinal=i, dataset=dataset, subtask=task,
                   prepared=f'samples/{task}-{i}.json', sha256=ef.digest([task, i]), input_tokens=576,
                   source_id=f'{task}-{i}', references=['A'], scoring='longbench_v2' if dataset == 'longbench-v2' else 'ruler_all')
        if thinking or dataset == 'longbench-v2':
            row.update(evaluation_protocol=budget.RULER_PROTOCOL if thinking else budget.LONGBENCH_PROTOCOL,
                       **{KEY: budget.make_policy(Tokenizer(), ' Answer:' if thinking else '')})
        else:
            from scripts.ruler import EVALUATION_PROTOCOL
            row['evaluation_protocol'] = EVALUATION_PROTOCOL
        if dataset == 'longbench-v2':
            row.update(length=data.LENGTHS[i % 3], domain=f'domain{i % 2}', sub_domain=f'sub{i % 3}')
            row['subtask'] = row['domain']+' / '+row['sub_domain']
        rows.append(row)
    groups = [[f'GPU-00000000-0000-0000-0000-{g*2+j:012x}' for j in range(2)]
              for g in range((4 if thinking else 5) if dataset == 'ruler' else 2)]
    settings['groups'] = groups
    atomic_json(root/'settings.json', settings); atomic_json(root/'rows.json', rows)
    plan = dict(schema='tp2-data-plan-v3', settings_sha256=file_hash(root/'settings.json'), rows_sha256=file_hash(root/'rows.json'),
                samples=len(rows), answers=len(rows)*9*len(v3.seeds(settings)), probes=len(rows), actions=v3.ACTIONS,
                schedule={role: data.scheduled(settings, role) for role in data.roles(settings)}, feature_profile=DEFINITIONS,
                folds=data.assign_folds(rows), **v3.validate_declaration(settings))
    atomic_json(root/'plan.json', plan)
    role = 'ruler' if dataset == 'ruler' else 'primary'
    atomic_json(root/role/'devices.json', dict(tp=2, groups=groups)); data.publish_protocols(root)
    for role in data.roles(settings):
        atomic_json(root/role/'session/initialization.json', dict(validated=True, engine_config={}))
    return settings, rows


def commit(root, row, case, seed=None):
    settings = data.read(root/'settings.json')
    role = 'features' if case == 'probe' else 'ruler' if settings['dataset'] == 'ruler' else 'primary'
    protocol, _ = data.stage(root, role)
    group = row['ordinal'] % len(protocol['groups'])
    result = dict(prompt_id=row['id'], method=case, input_sha256=row['sha256'], group=group,
                  gpu_uuids=protocol['groups'][group], cache_immutable=True, prompt_protocol=ef.PROMPT_PROTOCOL,
                  token_sha256=row['sha256'], evaluation_protocol=row['evaluation_protocol'],
                  retirement=[dict(rank=r, quiescent=True, transfers=dict(pending=0), request_bookkeeping=0) for r in range(2)],
                  initialization='session/initialization.json', initialization_sha256=file_hash(root/role/'session/initialization.json'),
                  answer_validation=protocol['answer_validation'])
    folder = record_dir(root/role, case, row['id'], seed)
    if case == 'probe':
        from runner.longbench_features import features
        folder.mkdir(parents=True, exist_ok=True)
        local = np.full((64, 320), 1/320, np.float32)
        mean = np.zeros(320, np.float32)
        for layer in local:
            mean += layer
        mean /= np.float32(64)
        scores = mean[64:]; heads = np.full((2, 8, 32, 320), 1/320, np.float32)
        artifacts = []
        for rank in range(2):
            path = folder/f'attention.rank{rank}.npz'
            arrays = dict(layers=local, local_mean=mean, scores=scores, heads=heads[rank],
                          head_layers=np.array(ef.HEAD_LAYERS, np.int64), boundaries=np.array([0, 64, 320, 576], np.int64),
                          question_positions=np.array([320], np.int64), context_positions=np.arange(320, dtype=np.int64),
                          original_to_formatted=np.arange(576, dtype=np.int64))
            arrays.update({f'prophetkv-{p}': ef.mask(ef.ranked(scores), p)+64 for p in v3.RATIOS})
            np.savez_compressed(path, **arrays)
            artifacts.append(dict(rank=rank, path=path.name, sha256=file_hash(path)))
        result.update(internal_tokens=1, artifacts=artifacts, timings=dict(routing_overhead_seconds=.2),
                      features=features(dict(layers=np.stack([local]*2), scores=scores, heads=heads, head_layers=ef.HEAD_LAYERS), 64, 320))
    else:
        result.update(sequence_id=row['id'], seed=seed, subtask=row['subtask'], scoring=row['scoring'], references=row['references'],
                      executed_action=case, accuracy=float(seed % 2 == 0), num_cached_tokens=0, prediction='A',
                      output_cap_reached=False, unfinished_thinking=False, timings=dict(ttft_seconds=4. if case == 'nocache' else 2.))
        if KEY in row:
            p = row[KEY]; tokens = [65, p['close_id']] + p['transition_ids'] + p['prefix_ids'] + [65, 3]
            state = BudgetState(p).consume(tokens)
            from runner.reporting import OutputAnalyzer
            result.update(OutputAnalyzer(Tokenizer()).analyze(tokens, [], True), **state.metrics(Tokenizer()))
            result[KEY] = p; result['output_token_ids'] = tokens; result['max_output_tokens'] = 16384
            result['timings']['first_answer_content_seconds'] = result['timings']['ttft_seconds']+.1
        else:
            result.update(thinking_tokens=0, answer_tokens=1, control_tokens=1, output_tokens=2, forced_tokens=0,
                          thinking_cap_reached=False, answer_cap_reached=False, output_token_ids=[65, 3], max_output_tokens=CAPS[row['subtask']])
    publish(root/role, case, row, protocol, result, [], seed=seed)
    return result


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_declared_counts_inventory_shards_and_round_robin(self):
        for dataset, thinking, shards, expected in [('longbench-v2', False, 2, 18108), ('ruler', False, 5, 58500), ('ruler', True, 4, 58500)]:
            root = self.root/(dataset+str(thinking)); settings, rows = fixture(root, dataset, thinking)
            _, plan, _ = data.load(root); self.assertEqual(plan['answers'], expected)
            role = 'ruler' if dataset == 'ruler' else 'primary'; protocol, _ = data.stage(root, role)
            split = [v3.ordered_rows(rows, protocol, g) for g in range(shards)]
            self.assertEqual(len({r['id'] for part in split for r in part}), len(rows))
            if dataset == 'ruler':
                self.assertEqual([r['subtask'] for r in split[0][:13]], list(TASKS))
                self.assertEqual([len(part) for part in split], [6500//shards]*shards)
            else:
                self.assertEqual(list(map(len, split)), [252, 251])
            self.assertEqual(protocol['actions'], v3.ACTIONS); self.assertNotIn('extra', data.roles(settings))
        self.assertEqual(v3.seeds(v3.declaration('longbench-v2', 7)), list(range(7)))
        for n in (0, -1, 1.2, True):
            with self.assertRaises(ValueError): v3.declaration('longbench-v2', n)

    def test_phase_barrier_and_no_extra_supervisor(self):
        settings, rows = fixture(self.root)
        phases = []
        with patch.object(control, 'execute_phase', side_effect=lambda b, r, p, state: phases.append(p)), \
                patch('scripts.longbench_a800_report.report'), patch('scripts.router_dataset.export_collection'), \
                patch.object(control, 'wait_extra') as extra:
            control.supervise(self.root, 'primary')
        self.assertEqual(phases, ['baseline', 'features', 'cached']); extra.assert_not_called()
        self.assertEqual(data.read(self.root/'primary/controls-complete.json')['answers'], 18108)

    def test_seed_publication_and_partial_resume_worker(self):
        settings, rows = fixture(self.root); row = rows[0]
        commit(self.root, row, 'nocache', 0); commit(self.root, row, 'nocache', 2)
        protocol, _ = data.stage(self.root, 'primary')
        before = file_hash(record_dir(self.root/'primary', 'nocache', row['id'], 0)/'validated.json')
        engine = Mock(); engine.answer.return_value = ({}, []); engine.end.return_value = {}
        with patch.object(control, 'stage', return_value=(protocol, [row])), \
                patch('runner.setups.check_environment', return_value=protocol['groups'][0]), \
                patch('runner.corpus_runtime.Engine', return_value=engine), patch.object(control, 'publish') as published:
            control.worker(self.root, 'primary', 'baseline', 'attempt')
        self.assertEqual([c.kwargs['seed'] for c in engine.answer.call_args_list], [1, 3])
        self.assertEqual([c.kwargs['seed'] for c in published.call_args_list], [1, 3])
        engine.begin.assert_called_once(); engine.end.assert_called_once(); engine.close.assert_called_once()
        self.assertEqual(before, file_hash(record_dir(self.root/'primary', 'nocache', row['id'], 0)/'validated.json'))
        with self.assertRaises(FileExistsError): commit(self.root, row, 'nocache', 0)
        bad = dict(protocol, answer_seeds=[0, 1])
        with self.assertRaises(ValueError): accepted(self.root/'primary', 'nocache', row, bad, seed=0)

    def test_cached_method_then_seed_order_resume_and_retirement(self):
        settings, rows = fixture(self.root)
        row = rows[0]; protocol, _ = data.stage(self.root, 'primary')
        commit(self.root, row, 'prophetkv-1', 0)
        engine = Mock(); engine.answer.return_value = ({}, []); engine.end.return_value = {}
        with patch.object(control, 'stage', return_value=(protocol, [row])), \
                patch('runner.setups.check_environment', return_value=protocol['groups'][0]), \
                patch('runner.corpus_runtime.Engine', return_value=engine), patch.object(control, 'publish'):
            control.worker(self.root, 'primary', 'cached', 'attempt')
        self.assertEqual([(c.args[1], c.kwargs['seed']) for c in engine.answer.call_args_list],
                         [(a['id'], seed) for a in v3.ACTIONS[1:] for seed in range(4) if (a['id'], seed) != ('prophetkv-1', 0)])
        engine.begin.assert_called_once(); engine.end.assert_called_once()

    def test_configure_pins_answers_and_rejects_reused_paths(self):
        args = NS(root=self.root/'run', model=self.root/'model', data=self.root/'source',
                  prepared=self.root/'prepared', cache_root=self.root/'cache', devices='0,1,2,3', role='primary',
                  tp=2, seed=42, samples_per_task=500, collection_version='v3', answers_per_question=7)
        groups = [[f'GPU-00000000-0000-0000-0000-{j:012x}' for j in range(i, i+2)] for i in (0, 2)]
        with patch.object(control, 'resolve_groups', return_value=groups), patch.object(control, 'code_hashes', return_value={}):
            control.configure(args)
            before = file_hash(args.root/'settings.json')
            args.answers_per_question = 4
            with self.assertRaisesRegex(ValueError, 'Frozen'): control.configure(args)
            self.assertEqual(before, file_hash(args.root/'settings.json'))
            args.root = self.root/'new'; args.prepared.mkdir(); (args.prepared/'old').write_text('preserve')
            with self.assertRaisesRegex(ValueError, 'empty new'): control.configure(args)
        self.assertEqual((args.prepared/'old').read_text(), 'preserve')

    def test_preparation_builds_seven_seed_plan_without_changing_tokens(self):
        settings, source_rows = fixture(self.root, answers=7)
        prepared = self.root/'prepared'; settings['prepared'] = str(prepared)
        atomic_json(self.root/'settings.json', settings)
        manifest, files = [], {}
        policy = budget.make_policy(Tokenizer())
        from runner.layout import stamp_sample
        sample = stamp_sample(dict(token_ids=[65]*576, thinking=True, boundaries=[0, 64, 320, 576], question_positions=[320],
                                   max_output_tokens=16384, evaluation_protocol=budget.LONGBENCH_PROTOCOL, **{KEY: policy}), 'template')
        for row in source_rows:
            value = dict(sample, source_id=row['source_id'], source_metadata={k: row[k] for k in ('length', 'domain', 'sub_domain')})
            atomic_json(prepared/row['prepared'], value); files[row['prepared']] = file_hash(prepared/row['prepared'])
            manifest.append({k: row[k] for k in ('id', 'prepared', 'subtask', 'references', 'scoring')})
        (prepared/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
        files['manifest.jsonl'] = file_hash(prepared/'manifest.jsonl')
        atomic_json(prepared/'preparation.json', dict(files=files))
        # Remove only fixture-created metadata to exercise real initial publication.
        for name in ('plan.json', 'rows.json', 'primary/protocol.json', 'features/protocol.json'):
            (self.root/name).unlink()
        with patch('scripts.longbench_v2.prepare') as prepare_inputs:
            plan = data.prepare(self.root)
        self.assertEqual(plan['answers'], 503*9*7); self.assertEqual(plan['probes'], 503)
        self.assertEqual(prepare_inputs.call_args.args[0].answers_per_question, 7)
        for name, sha in files.items(): self.assertEqual(file_hash(prepared/name), sha)

    def test_status_seed_intersection_categories_and_stdout(self):
        settings, rows = fixture(self.root)
        for action in v3.ACTIONS:
            commit(self.root, rows[0], action['id'], 0)
        commit(self.root, rows[0], 'nocache', 1)
        out = io.StringIO()
        with redirect_stdout(out): result = report.report(self.root, same_count=True)
        table = result['tables']['overall']; self.assertEqual(table['nocache']['completed'], 1)
        self.assertEqual(table['nocache']['expected'], 2012)
        self.assertEqual(table['prophetkv-1']['ttft_speedup'], 2.)
        self.assertEqual(result['tables']['seed-1/overall']['nocache']['accuracy_percent'], None)
        self.assertEqual(result['tables']['seed-0/overall']['nocache']['expected'], 503)
        self.assertIn('category/domain0 / sub0', result['tables'])
        self.assertNotIn('seed-', out.getvalue()); self.assertNotIn('domain', out.getvalue())
        self.assertIn('seed-3/overall', (self.root/'status_same_count.md').read_text())
        self.assertEqual(len(out.getvalue().splitlines()), 11)

    def test_unmatched_means_and_paired_speedup_use_sequence_seed(self):
        settings, rows = fixture(self.root)
        data_by_action = {}
        for a in v3.ACTIONS:
            data_by_action[a['id']] = [commit(self.root, rows[0], a['id'], 0)]
        extra = commit(self.root, rows[0], 'nocache', 1)
        extra['timings']['ttft_seconds'] = 100
        data_by_action['nocache'].append(extra)
        value = report.summarize(data_by_action, rows, settings)
        self.assertEqual(value['tables']['overall']['nocache']['accuracy_percent'], 50.)
        self.assertEqual(value['tables']['overall']['nocache']['mean_ttft_seconds'], 52.)
        self.assertEqual(value['tables']['overall']['prophetkv-1']['ttft_speedup'], 2.)

    def test_snapshot_export_missing_seed_atomic_refresh_and_corruption(self):
        settings, rows = fixture(self.root); row = rows[0]
        stdout = io.StringIO()
        with redirect_stdout(stdout): self.assertIsNone(export.export_compact(self.root, workers=1))
        self.assertIn('chưa có data feature probe', stdout.getvalue())
        commit(self.root, row, 'probe')
        for action in v3.ACTIONS:
            for seed in range(4):
                if (action['id'], seed) != ('prophetkv-80', 3): commit(self.root, row, action['id'], seed)
        stdout = io.StringIO()
        with redirect_stdout(stdout): self.assertIsNone(export.export_compact(self.root, workers=1))
        self.assertIn('chưa đủ data', stdout.getvalue())
        commit(self.root, row, 'prophetkv-80', 3)
        path = export.export_compact(self.root, workers=1, quiet=True)
        arrays, meta = ef.verify_npz(path)
        self.assertEqual(arrays['X'].shape, (4, 100)); self.assertEqual(arrays['y_accuracy'].shape, (4, 9))
        self.assertEqual(len(set(arrays['prompt_ids'])), 4); self.assertEqual(len(set(arrays['folds'])), 1)
        np.testing.assert_array_equal(arrays['X'][0], arrays['X'][3])
        before = file_hash(path)
        with patch.object(ef, 'extracted_prompts', side_effect=RuntimeError('worker failed')):
            with self.assertRaises(RuntimeError): export.export_compact(self.root, workers=1)
        self.assertEqual(file_hash(path), before)
        original = ef.extracted_prompts
        def during_export(jobs, workers):
            commit(self.root, rows[1], 'nocache', 0)
            yield from original(jobs, workers)
        with patch.object(ef, 'extracted_prompts', side_effect=during_export):
            export.export_compact(self.root, workers=1, quiet=True)
        self.assertEqual(ef.verify_npz(path)[0]['X'].shape, (4, 100))
        result_path = record_dir(self.root/'primary', 'nocache', row['id'], 0)/'result.json'
        result_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Corrupt'): export.export_compact(self.root, workers=1)
        with self.assertRaisesRegex(ValueError, 'Protected'):
            export.protected_output(ROOT/'.analysis/longbenchv2.npz')

    def test_export_lock_active_collector_and_foreign_output(self):
        import fcntl
        settings, rows = fixture(self.root, answers=2); row = rows[0]
        commit(self.root, row, 'probe')
        for action in v3.ACTIONS:
            for seed in range(2): commit(self.root, row, action['id'], seed)
        with (self.root/'primary/run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = export.export_compact(self.root, workers=2, quiet=True)
        arrays, metadata = ef.verify_npz(path)
        self.assertEqual(arrays['X'].shape, (2, 100))
        with (self.root/'feature-export.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): export.export_compact(self.root, workers=1)
        before = file_hash(path); metadata['plan_sha256'] = 'a'*64
        with self.assertRaisesRegex(ValueError, 'another run'):
            export.publish(path, arrays, metadata)
        self.assertEqual(before, file_hash(path))
        with patch.object(export.os, 'replace', side_effect=OSError('rename failed')):
            with self.assertRaises(OSError): export.export_compact(self.root, workers=1)
        self.assertEqual(before, file_hash(path))

    def test_portable_full_seed_inventory_and_final_completion_gate(self):
        from scripts.router_dataset import example, save, load as load_portable, export_collection
        settings, rows = fixture(self.root, answers=2); row = rows[0]
        probe = commit(self.root, row, 'probe')
        measured = {seed: {a['id']: commit(self.root, row, a['id'], seed) for a in v3.ACTIONS} for seed in range(2)}
        values = []
        for original in rows:
            for seed in range(2):
                value = example(original, probe, measured[seed], {'fixture/validated.json': 'a'*64}, 'longbench-v2')
                value.update(id=f"{original['id']}--seed-{seed}", sequence_id=original['id'], seed=seed,
                             group_id=original['id'], ordinal=original['ordinal'], input_tokens=original['input_tokens'],
                             fold=original['ordinal'] % 5, domain=original['domain'], sub_domain=original['sub_domain'])
                for outcome in value['outcomes'].values():
                    outcome['gpu_uuids'] = settings['groups'][original['ordinal'] % 2]
                values.append(value)
        output = self.root/'portable.json'
        value = save(output, 'longbench-v2', v3.ACTIONS, values, settings)
        self.assertEqual(len(load_portable(output)['rows']), 1006)
        self.assertEqual(value['schema'], 'attention-router-dataset-v3')
        with self.assertRaises(ValueError): save(self.root/'missing.json', 'longbench-v2', v3.ACTIONS, values[:-1], settings)
        with self.assertRaises((ValueError, FileNotFoundError)): export_collection(self.root, self.root/'final.json')
        # CPU fixture for the supervisor's real final-export path: all rows read
        # via the committed template, retaining full declared membership.
        snapshot = v3.Snapshot(self.root)
        for role, protocol in snapshot.protocols.items():
            n = len(rows)*(1 if role == 'features' else 18)
            atomic_json(self.root/role/('complete.json' if role == 'features' else 'controls-complete.json'),
                        dict(owned_engines_exited=True, protocol_sha256=file_hash(self.root/role/'protocol.json'),
                             **{('probes' if role == 'features' else 'answers'): n}))
        for original in rows[1:]:
            for (role, case, sequence, seed), path in list(snapshot.receipts.items()):
                if sequence == row['id']:
                    snapshot.receipts[(role, case, original['id'], seed)] = path
        def portable_rows():
            for r in rows:
                outcomes = copy.deepcopy(measured)
                for answers in outcomes.values():
                    for answer in answers.values():
                        answer['gpu_uuids'] = settings['groups'][r['ordinal'] % 2]
                yield r, probe, outcomes
        with patch.object(export, 'Snapshot', return_value=snapshot), \
                patch.object(export, 'complete_rows', return_value=portable_rows()):
            final = export_collection(self.root, self.root/'final.json')
        self.assertEqual(len(final['rows']), 1006)

    def test_launcher_compute_aliases_and_seed_flag(self):
        script = self.root/'python'; script.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n'); script.chmod(0o755)
        env = self.root/'test.env'; env.write_text(f'PYTHON_BIN={script}\nGPU_DEVICES=explicit-uuids\n')
        for launcher in ('a800_longbench_primary', 'l20_ruler_data', 'l40_ruler_thinking_data'):
            for args in (['compute', 'feature'], ['compute-feature']):
                result = subprocess.run(['bash', str(ROOT/f'scripts/launcher/{launcher}.sh'), *args], cwd='/tmp',
                                        env=dict(os.environ, UCM_ENV_FILE=str(env), ANSWERS_PER_QUESTION='7'), capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr); self.assertIn('\ncompute-feature\n', result.stdout)
                self.assertIn('\n--collection-version\nv3\n', result.stdout)
                if launcher.startswith('a800'): self.assertIn('\n--answers-per-question\n7\n', result.stdout)


class BudgetTests(unittest.TestCase):
    def test_natural_forced_closure_total_cap_and_prefix_scoring(self):
        from runner.reporting import OutputAnalyzer
        tokenizer = Tokenizer()
        for prefix in ('', ' Answer:'):
            p = budget.make_policy(tokenizer, prefix)
            for count in (0, 10, 15000):
                first = [65]*count + [p['close_id']] + p['transition_ids'] + p['prefix_ids']
                for answer in ([66, 3], [66]*(16384-len(first))):
                    tokens = first+answer; state = BudgetState(p).consume(tokens)
                    self.assertEqual(state.phase, 'done'); self.assertEqual(state.closure, 'forced' if count == 15000 else 'natural')
                    self.assertEqual(state.metrics(tokenizer)['scored_text'], 'B'*(len(answer)-(answer[-1] == 3)))
                    metrics = dict(OutputAnalyzer(tokenizer).analyze(tokens, [], True), **state.metrics(tokenizer))
                    metrics.update(output_cap_reached=metrics['answer_cap_reached'], ttft_seconds=1., first_answer_content_seconds=2.)
                    validate_thinking_outcome(metrics, p)
                    self.assertEqual(metrics['answer_cap_reached'], len(tokens) == 16384)
            with self.assertRaisesRegex(ValueError, 'forced token'):
                BudgetState(p).consume([65]*15000+[66])
        s = format_thinking_sample(tokenizer, raw_row(), 'qa_1', collection_v3=True)
        self.assertEqual(s['max_output_tokens'], 16384); self.assertNotIn('answer_cap', s[KEY])
        legacy = format_thinking_sample(tokenizer, raw_row(), 'qa_1')
        self.assertEqual(s['token_ids'], legacy['token_ids']); self.assertEqual(legacy[KEY]['answer_cap'], 32)

    def test_logits_forced_close_and_scheduler_total_budget(self):
        import torch
        from runner.thinking_budget import enforce_logits, stopping_hook
        p = budget.make_policy(Tokenizer())
        tokens = [65]*15000
        request = NS(sampling_params=NS(extra_args={KEY: p}), output_token_ids=tokens,
                     num_tokens=20000, num_output_tokens=len(tokens), max_tokens=16384)
        runner = NS(requests={'r': request}, input_batch=NS(req_id_to_index={'r': 0}), speculative_config=None)
        logits = torch.zeros((1, 256)); enforce_logits(runner, logits)
        self.assertEqual(logits[0, 2], 0); self.assertEqual(torch.isfinite(logits).sum(), 1)
        first = tokens+[2]+p['transition_ids']
        request.output_token_ids = first+[66]*(16384-len(first)); request.num_output_tokens = 16384
        stop = stopping_hook(Mock(), NS(FINISHED_STOPPED='eos', FINISHED_LENGTH_CAPPED='cap'))
        self.assertTrue(stop(request, 82304)); self.assertEqual(request.status, 'cap')
        request = NS(sampling_params=NS(extra_args={KEY: p}), output_token_ids=[2]+p['transition_ids']+[3],
                     num_tokens=10, num_output_tokens=4, max_tokens=16384)
        self.assertTrue(stop(request, 82304)); self.assertEqual(request.status, 'eos')

    def test_seed_submission_and_probe_populate_exclusion(self):
        from runner.generation import generate
        s = format_thinking_sample(Tokenizer(), raw_row(), 'qa_1', collection_v3=True)
        p = s[KEY]
        class Engine:
            active = False
            def has_unfinished_requests(self): return self.active
            def add_request(self, rid, prompt, params): self.rid, self.params, self.active = rid, params, True
            def step(self):
                self.active = False
                tokens = [2]+p['transition_ids']+p['prefix_ids']+[65, 3] if self.params.max_tokens > 1 else [65]
                return [NS(request_id=self.rid, outputs=[NS(token_ids=tokens)], finished=True)]
        with patch.dict(sys.modules, {'vllm': NS(SamplingParams=lambda **kw: NS(**kw)), 'vllm.inputs': NS(TokensPrompt=lambda **kw: kw)}):
            engine = Engine(); generate(engine, s['token_ids'], 16384, 'answer', True, sample=s, seed=3)
            self.assertEqual(engine.params.seed, 3); self.assertEqual(engine.params.extra_args[KEY], p)
            generate(engine, s['token_ids'], 1, 'probe', False, sample=s)
            self.assertNotIn(KEY, engine.params.extra_args)
            generate(engine, s['token_ids'][:64], 1, 'populate', phase='populate')
            self.assertNotIn(KEY, engine.params.extra_args)

    def test_engine_preserves_seed_scores_cache_and_request_retirement(self):
        from runner.corpus_runtime import Engine
        from runner.reporting import OutputAnalyzer
        from runner.corpus_records import NATIVE_ANSWER_VALIDATION
        engine = Engine.__new__(Engine)
        engine.tp = 2; engine.protocol = dict(actions=v3.ACTIONS, answer_validation=NATIVE_ANSWER_VALIDATION)
        engine.sample = format_thinking_sample(Tokenizer(), raw_row(), 'qa_1', collection_v3=True)
        engine.row = dict(id='qa_1-000', subtask='qa_1', scoring='ruler_any', references=['B'])
        engine.cached = True; engine.pc = Mock(); engine.llm = Mock(); engine.scheduler = 'scheduler'
        engine.rid = lambda suffix: suffix; engine.analyzer = OutputAnalyzer(Tokenizer())
        engine.common = lambda case, timings, retired: dict(timings=timings)
        p = engine.sample[KEY]
        def generate(*args, seed, **kwargs):
            tokens = [65, 2]+p['transition_ids']+p['prefix_ids']+[66+seed, 3]
            return NS(outputs=[NS(token_ids=tokens, text='raw')], num_cached_tokens=64,
                      ruler_thinking_state=BudgetState(p).consume(tokens), first_answer_content_seconds=.3), .2, .4
        with patch.dict(sys.modules, {'runner.worker': NS(drain=object(), router_dense_receipt=object())}), \
                patch('runner.generation.generate', side_effect=generate) as generated, \
                patch('runner.generation.verify_diagnostics'), patch('runner.corpus_runtime.clean_capture'), \
                patch('runner.corpus_runtime.arm_tree'), patch('runner.corpus_runtime.verify_retired') as retired:
            records = [engine.answer(v3.ACTIONS[2], v3.ACTIONS[2]['id'], seed=seed)[0] for seed in range(2)]
        self.assertEqual([r['seed'] for r in records], [0, 1])
        self.assertEqual([r['accuracy'] for r in records], [1., 0.])
        self.assertEqual([c.kwargs['seed'] for c in generated.call_args_list], [0, 1])
        self.assertEqual(engine.pc.unchanged.call_count, 2); self.assertEqual(retired.call_count, 2)
        self.assertEqual(records[0]['answer_tokens'], len(p['transition_ids'])+len(p['prefix_ids'])+1)


if __name__ == '__main__':
    unittest.main()
