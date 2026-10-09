"""CPU-only 200+300 L20 non-thinking cohorts; no model/GPU execution."""
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

from runner.setups import atomic_json, file_hash
from runner.thinking_budget import KEY
from runner.tree_profiles import config
from scripts import ruler_extension as extension, longbench_a800_data as data
from scripts import longbench_a800_control as control, router_dataset as portable
from scripts.export_features import Collection
from test_ruler_extension import prepared_fixture, dataset_fixture, complete

ROOT = Path(__file__).resolve().parents[1]
GPUS = [f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(10)]
INVENTORY = '\n'.join(f'{i}, {u}' for i, u in enumerate(GPUS))


class L20ExtensionTests(unittest.TestCase):
    def test_full200_plus300_config_prepare_export_and_immutable_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); parent = root/'parent'; delta = root/'delta'
            args = NS(root=parent, role='ruler', model=Path('/m'), data=Path('/source'),
                      prepared=root/'prepared200', cache_root=root/'cache200', tp=2, seed=42,
                      samples_per_task=200, devices=','.join(map(str, range(10))), thinking=False)
            with patch.object(control, 'code_hashes', return_value={}), patch.object(control.subprocess, 'check_output', return_value=INVENTORY):
                control.configure(args)
            prepared_fixture(args.prepared, 200, thinking=False, adapter=extension.LEGACY_NONTHINKING_ADAPTER,
                             chunker=extension.NONTHINKING_CHUNKER_MOVE[0])
            with patch('scripts.ruler.prepare'):
                data.prepare(parent)
            complete(parent); parent_data = dataset_fixture(parent)
            self.assertNotIn('hardware_profile', parent_data['provenance'])
            (parent/'ruler-features.npz').write_bytes(b'preserve original compact export')
            pins = {p: (file_hash(p), p.stat().st_mtime_ns) for directory in (parent, args.prepared)
                    for p in directory.rglob('*') if p.is_file()}
            args.root = delta; args.prepared = root/'prepared500'; args.cache_root = root/'cache300'
            args.samples_per_task = 500; args.extend_from = parent
            with patch.object(control, 'code_hashes', return_value={}), patch.object(control.subprocess, 'check_output', return_value=INVENTORY):
                args.devices = ','.join(map(str, reversed(range(10))))
                with self.assertRaisesRegex(ValueError, 'UUID pairs'):
                    control.configure(args)
                args.devices = ','.join(map(str, range(10)))
                args.samples_per_task = 300
                with self.assertRaisesRegex(ValueError, 'target500'):
                    control.configure(args)
                args.samples_per_task = 500
                control.configure(args)
            settings = data.read(delta/'settings.json')
            self.assertEqual(settings['samples_per_task'], 300)
            self.assertNotIn('execution_profile', settings)
            self.assertEqual(settings['extension']['start'], 200)
            self.assertEqual(settings['extension']['target_samples_per_task'], 500)
            for mutator in (lambda s: s.update(execution_profile='ruler-thinking'),
                            lambda s: s['extension'].update(start=0),
                            lambda s: s.update(samples_per_task=500)):
                bad = copy.deepcopy(settings); mutator(bad)
                with self.assertRaises(ValueError): data.ruler_samples(bad)
            manifest = prepared_fixture(args.prepared, 500, thinking=False, adapter=extension.ADAPTER_IMPORT_MOVE[1],
                                        chunker=extension.NONTHINKING_CHUNKER_MOVE[1])
            prepared_pins = {p: (file_hash(p), p.stat().st_mtime_ns)
                             for p in args.prepared.rglob('*') if p.is_file()}
            with patch('scripts.ruler.prepare') as generate:
                plan = data.prepare(delta)
            self.assertEqual(prepared_pins, {p: (file_hash(p), p.stat().st_mtime_ns) for p in prepared_pins})
            evidence = data.read(delta/'extension.json')
            self.assertEqual(evidence['prefix_prompts_checked'], 2600)
            self.assertEqual(evidence['chunker_compatibility']['parent_sha256'], extension.NONTHINKING_CHUNKER_MOVE[0])
            self.assertEqual(evidence['chunker_compatibility']['new_sha256'], extension.NONTHINKING_CHUNKER_MOVE[1])
            self.assertEqual(generate.call_args.args[0].samples, 500)
            self.assertFalse(generate.call_args.args[0].thinking)
            self.assertEqual((plan['samples'], plan['answers'], plan['probes']), (3900,46800,3900))
            _, rows = data.stage(delta, 'ruler')
            self.assertEqual([sum(r['ordinal'] % 5 == g for r in rows) for g in range(5)], [780]*5)
            self.assertTrue(all(200 <= r['ordinal'] < 500 and KEY not in r for r in rows))
            self.assertFalse({r['id'] for r in rows} & {r['id'] for r in parent_data['rows']})
            cfg = config('/m', 'ruler', False, '/cache', hardware='l20-tp2', tp=2)
            self.assertEqual(cfg['gpu_memory_utilization'], .95)
            self.assertEqual(cfg['max_model_len'], 65920)
            self.assertNotIn('num_gpu_blocks_override', cfg)
            complete(delta); dataset_fixture(delta)
            self.assertEqual(len(Collection(delta).rows), 3900)
            combined = extension.merge_extension(delta)
            self.assertEqual(len(combined['rows']), 6500)
            self.assertEqual(combined['provenance']['samples_per_task'], 500)
            self.assertEqual(combined['provenance']['collection'], 'verified-200-plus-300')
            self.assertEqual(len(portable.load(delta/'ruler-data-500.json')['rows']), 6500)
            combined_by_id = {r['id']: r for r in combined['rows']}
            for row in parent_data['rows']:
                self.assertEqual(combined_by_id[row['id']], row)
            self.assertEqual(pins, {p: (file_hash(p), p.stat().st_mtime_ns) for p in pins})
            receipt = data.read(args.prepared/'preparation.json')
            for key in ('chunker', 'adapter', 'versions', 'assets'):
                changed = copy.deepcopy(receipt); changed['spec'][key] = 'unapproved-change'
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    extension.verify_prepared_extension(settings, changed, manifest)
            # Known file mappings must still reject changes to tokens/layout.
            path = args.prepared/manifest[0]['prepared']; original = path.read_bytes()
            for key, value in (('token_ids', [999]), ('boundaries', [0, 1, 2])):
                changed = copy.deepcopy(receipt); sample = json.loads(original)
                sample[key] = value; atomic_json(path, sample)
                changed['files'][manifest[0]['prepared']] = file_hash(path)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'token/layout/policy prefix differs'):
                    extension.verify_prepared_extension(settings, changed, manifest)
            path.write_bytes(original)
            # A shifted/reordered prefix cannot be accepted, even under a known adapter mapping.
            receipt = data.read(args.prepared/'preparation.json')
            name = 'raw/qa_1/validation.jsonl'; path = args.prepared/name
            raw = path.read_text().splitlines(); raw[0], raw[1] = raw[1], raw[0]
            path.write_text('\n'.join(raw)+'\n'); receipt['files'][name] = file_hash(path)
            with self.assertRaisesRegex(ValueError, 'original 200 rows'):
                extension.verify_prepared_extension(settings, receipt, manifest)

    def test_relocated_launcher_keeps_nonthinking_target500_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'new checkout'; launchers = root/'scripts/launcher'
            shutil.copytree(ROOT/'scripts/launcher', launchers)
            fake = Path(tmp)/'python'
            fake.write_text(f'#!{sys.executable}\nimport json,sys,os\nprint(json.dumps([sys.argv[1:],os.environ["CUDA_VISIBLE_DEVICES"]]))\n')
            fake.chmod(0o755)
            (root/'.env.l20').write_text(f'PYTHON_BIN="{fake}"\nSAMPLES_PER_TASK=200\n')
            env = dict(PATH=os.environ['PATH'], EXTEND_FROM='/old200', SAMPLES_PER_TASK='500',
                       EXPERIMENT_DIR='/new300', PREPARED_DIR='/prepared500', CACHE_ROOT='/cache300')
            for command in ('configure', 'prepare', 'detach', 'resume', 'status', 'merge'):
                result = subprocess.run(['bash', str(launchers/'l20_ruler_data.sh'), command],
                                        cwd='/tmp', env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                args, cuda = json.loads(result.stdout)
                self.assertEqual(cuda, '')
                self.assertNotIn('--thinking', args)
                self.assertEqual(args[args.index('--samples-per-task')+1], '500')
                self.assertEqual(args[args.index('--extend-from')+1], '/old200')
                self.assertEqual(args[args.index('--root')+1], '/new300')


if __name__ == '__main__':
    unittest.main()
