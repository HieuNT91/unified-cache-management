"""Prepared-input ownership and protocol checks for the real-data debug CLI."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from runner.cache.setups import atomic_json, file_hash
from runner.inputs.layout import stamp_sample
from runner.runtime.output_budget import RULER_PROTOCOL, LONGBENCH_PROTOCOL, make_policy
from runner.runtime.thinking_budget import KEY
from scripts.router.debug_adatree import prepare_run, unchanged_inputs
from test_ruler_thinking import Tokenizer


def fixture(root, dataset='ruler'):
    model, prepared = root/'model', root/'prepared'
    atomic_json(model/'config.json', dict(model_type='qwen3', num_hidden_layers=64, hidden_size=5120))
    atomic_json(model/'tokenizer.json', {'test': True})
    hashes = {name: file_hash(model/name) for name in ('config.json', 'tokenizer.json')}
    sample = dict(token_ids=[42]*8448, boundaries=[0, 4096, 8192, 8448], question_positions=[8192],
        thinking=True, max_output_tokens=16384, source_row=0, task='niah_single_1',
        model_config_sha256=hashes['config.json'], execution_profile='ruler-thinking',
        evaluation_protocol=RULER_PROTOCOL if dataset == 'ruler' else LONGBENCH_PROTOCOL,
        **{KEY: make_policy(Tokenizer(), ' Answer:' if dataset == 'ruler' else '')})
    stamp_sample(sample, 'native-template')
    atomic_json(prepared/'samples/one.json', sample)
    row = dict(id='one', prepared='samples/one.json', subtask='task', references=['A'],
               scoring='ruler_all' if dataset == 'ruler' else 'longbench_v2')
    (prepared/'manifest.jsonl').write_text(json.dumps(row)+'\n')
    spec = dict(sampling={'seed': 0}, answer_seeds=[0, 1, 2, 3])
    spec['tokenizer_hashes' if dataset == 'ruler' else 'tokenizer_files'] = hashes
    atomic_json(prepared/'preparation.json', dict(spec=spec,
        files={name: file_hash(prepared/name) for name in ('samples/one.json', 'manifest.jsonl')}))
    return SimpleNamespace(model=model, prepared=prepared, output=root/'result', cache_root=root/'cache',
        dataset=dataset, prompt_id='one', seed=0,
        devices=','.join(f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(1, 5)))


class DebugAdaTreeTests(unittest.TestCase):
    def test_reuses_original_inputs_and_full_budgets_for_both_datasets(self):
        for dataset in ('ruler', 'longbench-v2'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temp:
                args = fixture(Path(temp), dataset)
                before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in args.prepared.rglob('*') if p.is_file()}
                protocol, row, pins = prepare_run(args)
                unchanged_inputs(pins)
                self.assertEqual(protocol['hardware_profile'], 'rtx4500ada-tree-validation')
                self.assertEqual(protocol['percentages'], [5, 20, 40])
                self.assertEqual(protocol['debug_method'], 'adatree-v1')
                self.assertEqual(row['input_tokens'], 8448)
                self.assertFalse(args.output.exists())
                self.assertFalse(args.cache_root.exists())
                self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})

    def test_rejects_input_corruption_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp))
            with (args.prepared/'samples/one.json').open('a') as stream:
                stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'input differs'):
                prepare_run(args)
            self.assertFalse(args.output.exists())

    def test_rejects_manifest_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp))
            with (args.prepared/'manifest.jsonl').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'manifest differs'):
                prepare_run(args)

    def test_rejects_smaller_chunk_layout_for_fixed_4096_features(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp))
            path = args.prepared/'samples/one.json'
            sample = json.loads(path.read_text())
            sample['boundaries'] = [0, 2048, 4096, 6144, 8192, 8448]
            atomic_json(path, sample)
            receipt_path = args.prepared/'preparation.json'
            receipt = json.loads(receipt_path.read_text())
            receipt['files']['samples/one.json'] = file_hash(path)
            atomic_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, 'chunk4096'):
                prepare_run(args)

    def test_rejects_changed_model_tokenizer(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp))
            atomic_json(args.model/'tokenizer.json', {'changed': True})
            with self.assertRaisesRegex(ValueError, 'tokenizer changed'):
                prepare_run(args)

    def test_rejects_unsafe_paths_and_reused_outputs(self):
        for change in ('nested', 'old_output', 'cache', 'same'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                args = fixture(Path(temp))
                if change == 'nested': args.output = args.prepared/'result'
                elif change == 'old_output': args.output.mkdir()
                elif change == 'cache': atomic_json(args.cache_root/'old.json', {})
                else: args.cache_root = args.output
                with self.assertRaises(ValueError):
                    prepare_run(args)

    def test_requires_explicit_distinct_four_device_uuids(self):
        for devices in ('0,1,2,3', 'GPU-00000000-0000-0000-0000-000000000001',
                        ','.join(['GPU-00000000-0000-0000-0000-000000000001']*4)):
            with self.subTest(devices=devices), tempfile.TemporaryDirectory() as temp:
                args = fixture(Path(temp)); args.devices = devices
                with self.assertRaisesRegex(ValueError, 'GPU UUIDs'):
                    prepare_run(args)

    def test_rejects_unknown_prompt_and_unprepared_seed(self):
        for change in ('prompt', 'ruler_seed', 'longbench_seed'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                args = fixture(Path(temp), 'longbench-v2' if change == 'longbench_seed' else 'ruler')
                if change == 'prompt': args.prompt_id = 'missing'
                else: args.seed = 7
                with self.assertRaises(ValueError):
                    prepare_run(args)

    def test_postrun_input_check_includes_mtime(self):
        import os
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp))
            _, _, pins = prepare_run(args)
            path = args.prepared/'samples/one.json'
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns+1))
            with self.assertRaisesRegex(RuntimeError, 'changed during debug'):
                unchanged_inputs(pins)


if __name__ == '__main__':
    unittest.main()
