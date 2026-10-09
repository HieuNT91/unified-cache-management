"""CPU fixtures for immutable 30+170 thinking collection and export."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from runner.setups import atomic_json, file_hash, fingerprint
from runner.longbench_features import FEATURES
from runner.thinking_budget import KEY, PROFILE, PROTOCOL, make_policy
from scripts import longbench_a800_data as data, longbench_a800_control as control
from scripts import ruler_extension as extension, router_dataset as portable
from scripts.ruler import TASKS, CAPS, EVALUATION_PROTOCOL
from test_ruler_thinking import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
GPUS = [f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(8)]


def prepared_fixture(root, count, thinking=True, adapter=None, chunker=None):
    spec = dict(samples=count, seed=42, protocol=PROTOCOL if thinking else EVALUATION_PROTOCOL,
                adapter=adapter or extension.ADAPTER_IMPORT_MOVE[0])
    if thinking:
        spec['execution_profile'] = PROFILE
    if chunker is not None:
        spec['chunker'] = chunker
    files = {}; manifest = []
    for t, task in enumerate(TASKS):
        policy = make_policy(Tokenizer(), ' Answer:', CAPS[task])
        raw = []
        for i in range(count):
            row_id = f'{task}-{i:03d}'; name = f'samples/{row_id}.json'
            sample = dict(token_ids=[t, i], task=task, source_row=i, thinking=True,
                          evaluation_protocol=PROTOCOL, execution_profile=PROFILE,
                          max_output_tokens=policy['output_reserve'], **{KEY: policy},
                          generator_config_sha256=fingerprint(spec))
            if not thinking:
                sample.pop(KEY); sample.pop('execution_profile')
                sample.update(thinking=False, evaluation_protocol=EVALUATION_PROTOCOL, max_output_tokens=CAPS[task])
            atomic_json(root/name, sample); files[name] = file_hash(root/name)
            manifest.append(dict(id=row_id, prepared=name, subtask=task, references=['A'], scoring='ruler_all'))
            raw.append(dict(input=f'{task} question {i}', outputs=['A']))
        name = f'raw/{task}/validation.jsonl'
        (root/name).parent.mkdir(parents=True)
        (root/name).write_text(''.join(json.dumps(r)+'\n' for r in raw)); files[name] = file_hash(root/name)
    (root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    files['manifest.jsonl'] = file_hash(root/'manifest.jsonl')
    atomic_json(root/'preparation.json', dict(spec=spec, files=files))
    return manifest


def complete(root):
    _, plan, _ = data.load(root)
    for role, filename, key, count in [('ruler', 'controls-complete.json', 'answers', plan['answers']),
                                       ('features', 'complete.json', 'probes', plan['probes'])]:
        atomic_json(root/role/filename, dict(owned_engines_exited=True, **{key: count},
                    protocol_sha256=file_hash(root/role/'protocol.json')))
    atomic_json(root/'ruler/complete.json', dict(complete=True, owned_engines_exited=True,
                                              plan_sha256=file_hash(root/'plan.json')))


def dataset_fixture(root):
    settings, _, rows = data.load(root); values = []
    groups = data.read(root/'ruler/devices.json')['groups']
    for row in rows:
        policy = row.get(KEY)
        forced = len(policy['transition_ids'])+len(policy['prefix_ids']) if policy else 0
        outcome = dict(accuracy=1., ttft_seconds=1., gpu_uuids=groups[row['ordinal'] % len(groups)],
                       thinking_tokens=4, answer_tokens=forced+2, control_tokens=2, output_tokens=forced+8,
                       output_cap_reached=False, generated_answer_tokens=2, generated_thinking_open_tokens=0,
                       forced_tokens=forced, thinking_cap_reached=False, answer_cap_reached=False,
                       thinking_closure='natural', first_answer_content_seconds=2.)
        values.append(dict(id=row['id'], dataset='ruler', subtask=row['subtask'], input_sha256=row['sha256'],
                           source_pins={'record': fingerprint(row['id'])}, features=dict.fromkeys(FEATURES, .5),
                           probe_overhead_seconds=.1, evaluation_protocol=row['evaluation_protocol'],
                           **({KEY: policy} if policy else {}),
                           outcomes={a['id']: dict(outcome) for a in data.ACTIONS}))
    provenance = {k: settings[k] for k in ('tp', 'seed', 'samples_per_task', 'execution_profile', 'evaluation_protocol', 'hardware_profile') if k in settings}
    if settings['hardware_profile'] == 'l20-tp2' and 'extension' not in settings:
        provenance.pop('hardware_profile')  # Original L20 export schema.
    provenance['plan_sha256'] = file_hash(root/'plan.json')
    if 'extension' in settings:
        provenance['extension'] = settings['extension']
    return portable.save(root/'ruler-data.json', 'ruler', data.ACTIONS, values, provenance)


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old = self.root/'old'; self.new = self.root/'new'
        args = NS(root=self.old, role='ruler', model=Path('/m'), data=Path('/ruler'),
                  prepared=self.root/'old-inputs', cache_root=self.root/'old-cache',
                  tp=2, samples_per_task=30, seed=42, devices=','.join(GPUS), thinking=True, hardware_profile='l40-tp2')
        with patch.object(control, 'code_hashes', return_value={}):
            control.configure(args)
        prepared_fixture(args.prepared, 30)
        with patch('scripts.ruler.prepare'):
            data.prepare(self.old)
        complete(self.old); self.parent = dataset_fixture(self.old)
        self.args = NS(**vars(args)); self.args.root = self.new
        self.args.prepared = self.root/'new-inputs'; self.args.cache_root = self.root/'new-cache'
        self.args.samples_per_task = 200; self.args.extend_from = self.old

    def configure(self):
        with patch.object(control, 'code_hashes', return_value={}):
            control.configure(self.args)
        return data.read(self.new/'settings.json')

    def test_delta_plan_union_exports_and_parent_immutability(self):
        originals = {p: (file_hash(p), p.stat().st_mtime_ns) for d in (self.old, self.root/'old-inputs') for p in d.rglob('*') if p.is_file()}
        self.configure(); prepared_fixture(self.args.prepared, 200)
        with patch('scripts.ruler.prepare') as generate:
            plan = data.prepare(self.new)
        self.assertEqual(generate.call_args.args[0].samples, 200)
        self.assertEqual((plan['samples'], plan['answers'], plan['probes']), (2210, 26520, 2210))
        settings, _, rows = data.load(self.new, check_code=False)
        self.assertEqual([sum(r['ordinal'] % 4 == g for r in rows) for g in range(4)], [546,546,559,559])
        self.assertFalse({r['id'] for r in rows} & {r['id'] for r in self.parent['rows']})
        complete(self.new)
        from scripts.export_features import Collection
        self.assertEqual(len(Collection(self.new).rows), 2210)
        delta = dataset_fixture(self.new)
        combined = extension.combine_exports(self.new, delta, self.new/'ruler-data.json')
        self.assertEqual(len(portable.load(self.new/'ruler-data-200.json')['rows']), 2600)
        by_id = {r['id']: r for r in combined['rows']}
        for r in self.parent['rows']:
            self.assertEqual(by_id[r['id']], r)
        self.assertEqual(combined['provenance']['components'][0]['provenance'], self.parent['provenance'])
        self.assertEqual(originals, {p: (file_hash(p), p.stat().st_mtime_ns) for p in originals})
        # Frozen delta metadata and resumed preparation keep exactly the same cohort.
        with patch('scripts.ruler.prepare'):
            self.assertEqual(data.prepare(self.new), plan)
        atomic_json(self.new/'extension.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'receipt changed'):
            data.load(self.new)

    def test_reject_incomplete_parent_and_reused_paths(self):
        marker = self.old/'ruler/controls-complete.json'; saved = marker.read_bytes()
        atomic_json(marker, dict(owned_engines_exited=False))
        with self.assertRaisesRegex(ValueError, 'completion receipt is invalid'):
            self.configure()
        marker.write_bytes(saved)
        for key, bad in [('root', self.old), ('prepared', self.root/'old-inputs'),
                         ('cache_root', self.root/'old-cache'/'child')]:
            original = getattr(self.args, key); setattr(self.args, key, bad)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'fresh'):
                self.configure()
            setattr(self.args, key, original)
        self.args.tp = 4; self.args.hardware_profile = 'l40-tp4'
        with self.assertRaisesRegex(ValueError, 'preserve parent tp'):
            self.configure()

    def test_reject_prefix_policy_duplicate_and_generator_mismatch(self):
        settings = self.configure(); manifest = prepared_fixture(self.args.prepared, 200)
        receipt = data.read(self.args.prepared/'preparation.json')
        extension.verify_prepared_extension(settings, receipt, manifest)
        for mode in ('raw', 'tokens', 'policy', 'duplicate', 'versions'):
            with self.subTest(mode=mode):
                changed = copy.deepcopy(receipt)
                if mode == 'versions':
                    changed['spec']['adapter'] = 'different'
                    with self.assertRaisesRegex(ValueError, 'differ'):
                        extension.verify_prepared_extension(settings, changed, manifest)
                    continue
                name = 'raw/qa_1/validation.jsonl' if mode == 'raw' else 'samples/niah_single_1-000.json'
                if mode == 'duplicate': name = 'samples/niah_single_1-030.json'
                path = self.args.prepared/name; original = path.read_bytes()
                if mode == 'raw': path.write_text(original.decode().replace('question 0', 'changed', 1))
                else:
                    value = data.read(path)
                    if mode == 'policy': value[KEY]['answer_cap'] += 1
                    else: value['token_ids'] = [0, 0] if mode == 'duplicate' else [999]
                    atomic_json(path, value)
                changed['files'][name] = file_hash(path)
                with self.assertRaisesRegex(ValueError, 'differs|Duplicate'):
                    extension.verify_prepared_extension(settings, changed, manifest)
                path.write_bytes(original)
        (self.old/'rows.json').write_text('[]')
        with self.assertRaisesRegex(ValueError, 'Parent experiment changed'):
            extension.verify_prepared_extension(settings, receipt, manifest)

    def test_count_guards_and_launcher_forwarding(self):
        for n in (30, 200):
            self.assertEqual(data.ruler_samples(dict(execution_profile=PROFILE, samples_per_task=n)), n)
        with self.assertRaises(ValueError):
            data.ruler_samples(dict(execution_profile=PROFILE, samples_per_task=170))
        settings = self.configure()
        self.assertEqual(data.ruler_samples(settings), 170)
        bad = copy.deepcopy(settings); bad['extension']['start'] = 0
        with self.assertRaises(ValueError): data.ruler_samples(bad)
        launchers = self.root/'checkout/scripts/launcher'; shutil.copytree(ROOT/'scripts/launcher', launchers)
        fake = self.root/'fake-python'
        fake.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n'); fake.chmod(0o755)
        config = self.root/'checkout/.env.l40.thinking'
        config.write_text(f'PYTHON_BIN="{fake}"\nGPU_DEVICES=from-env\nSAMPLES_PER_TASK=30\n')
        env = dict(PATH=os.environ['PATH'], EXTEND_FROM=str(self.old), SAMPLES_PER_TASK='200',
                   EXPERIMENT_DIR=str(self.new), PREPARED_DIR=str(self.args.prepared), CACHE_ROOT=str(self.args.cache_root))
        for cmd in ('configure', 'prepare', 'detach', 'resume', 'status', 'merge'):
            result = subprocess.run(['bash', str(launchers/'l40_ruler_thinking_data.sh'), cmd], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(result.stdout)
            self.assertEqual(args[args.index('--extend-from')+1], str(self.old))
            self.assertEqual(args[args.index('--samples-per-task')+1], '200')

    def test_known_adapter_import_move_preserves_prefix_checks_and_existing_inputs(self):
        settings = self.configure(); manifest = prepared_fixture(self.args.prepared, 200)
        receipt = data.read(self.args.prepared/'preparation.json')
        receipt['spec']['adapter'] = extension.ADAPTER_IMPORT_MOVE[1]
        # Model the new generator configuration hash without changing any tokens.
        for row in manifest:
            path = self.args.prepared/row['prepared']; sample = data.read(path)
            sample['generator_config_sha256'] = fingerprint(receipt['spec'])
            atomic_json(path, sample); receipt['files'][row['prepared']] = file_hash(path)
        atomic_json(self.args.prepared/'preparation.json', receipt)
        pins = {p: (file_hash(p), p.stat().st_mtime_ns) for directory in
                (self.root/'old-inputs', self.args.prepared) for p in directory.rglob('*') if p.is_file()}
        with patch('scripts.ruler.prepare'):
            plan = data.prepare(self.new)
        self.assertEqual(plan['samples'], 2210)
        evidence = data.read(self.new/'extension.json')
        self.assertEqual(evidence['prefix_prompts_checked'], 390)
        self.assertEqual(evidence['adapter_compatibility']['parent_sha256'], extension.ADAPTER_IMPORT_MOVE[0])
        self.assertEqual(evidence['adapter_compatibility']['new_sha256'], extension.ADAPTER_IMPORT_MOVE[1])
        self.assertEqual(pins, {p: (file_hash(p), p.stat().st_mtime_ns) for p in pins})
        for key in ('versions', 'tokenizer_hashes', 'assets', 'source_hashes', 'thinking_adapter', 'chunker', 'adapter'):
            bad = copy.deepcopy(receipt); bad['spec'][key] = 'unapproved-change'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                extension.verify_prepared_extension(settings, bad, manifest)
        path = self.args.prepared/manifest[0]['prepared']; sample = data.read(path)
        sample['token_ids'] = [999]; atomic_json(path, sample)
        receipt['files'][manifest[0]['prepared']] = file_hash(path)
        with self.assertRaisesRegex(ValueError, 'token/layout/policy prefix differs'):
            extension.verify_prepared_extension(settings, receipt, manifest)

    def test_tp4_parent_pending_probes_collects_delta_and_merges_later(self):
        # Adapt only this synthetic fixture to the user's original TP4 profile.
        old = data.read(self.old/'settings.json'); old.update(tp=4, hardware_profile='l40-tp4')
        atomic_json(self.old/'settings.json', old)
        plan = data.read(self.old/'plan.json'); plan['settings_sha256'] = file_hash(self.old/'settings.json')
        atomic_json(self.old/'plan.json', plan)
        groups = [GPUS[:4], GPUS[4:]]
        atomic_json(self.old/'ruler/devices.json', dict(tp=4, groups=groups))
        for role in ('ruler', 'features'):
            atomic_json(self.old/role/'protocol.json', data.protocol_for(old, role, file_hash(self.old/'plan.json'), groups))
        complete(self.old)
        for name in ('ruler/complete.json', 'features/complete.json', 'ruler-data.json', 'ruler-data.json.sha256'):
            (self.old/name).unlink()
        partial = self.old/'features/partial-probe.json'
        atomic_json(partial, dict(retained=True))
        before = {p: (file_hash(p), p.stat().st_mtime_ns) for p in self.old.rglob('*') if p.is_file()}
        self.args.tp = 4; self.args.hardware_profile = 'l40-tp4'
        settings = self.configure()
        self.assertEqual(settings['extension']['parent_scope'], 'controls-complete')
        prepared_fixture(self.args.prepared, 200)
        with patch('scripts.ruler.prepare'):
            data.prepare(self.new)
        _, _, rows = data.load(self.new)
        self.assertEqual([sum(r['ordinal'] % 2 == g for r in rows) for g in range(2)], [1105,1105])
        complete(self.new); delta = dataset_fixture(self.new)
        self.assertIsNone(extension.combine_exports(self.new, delta, self.new/'ruler-data.json', allow_pending=True))
        self.assertFalse((self.new/'ruler-data-200.json').exists())
        with self.assertRaisesRegex(ValueError, 'Parent probes/final export pending'):
            extension.merge_extension(self.new)
        self.assertEqual(before, {p: (file_hash(p), p.stat().st_mtime_ns) for p in before})
        # Later parent probe completion must not change the frozen delta settings.
        complete(self.old); parent = dataset_fixture(self.old)
        self.assertEqual(self.configure(), settings)
        combined = extension.merge_extension(self.new)
        self.assertEqual(len(combined['rows']), 2600)
        self.assertEqual(combined['provenance']['tp'], 4)
        combined_rows = {r['id']: r for r in combined['rows']}
        for row in parent['rows']:
            self.assertEqual(combined_rows[row['id']], row)
        self.assertEqual(extension.merge_extension(self.new), combined)
        # Previously configured complete-parent receipts remain readable.
        legacy = copy.deepcopy(settings)
        legacy['extension'].pop('parent_scope')
        legacy['extension']['parent_files'] = {n: file_hash(self.old/n) for n in extension.PARENT_FILES}
        extension.validate_extension(legacy)
        extension.check_parent(legacy['extension'])


if __name__ == '__main__':
    unittest.main()
