"""Runtime output budgets, generic GPU admission and measured TP1 KV capacity."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from runner.cache.setups import atomic_json, file_hash
from runner.inputs.layout import boundaries_for, stamp_sample
from runner.operations import check_hardware
from runner.router.profiles import allocation, config, validate_initialization
from scripts.collection import run_controller as collection
from run_fixtures import fixture, GPUS


class RuntimeCapacityTests(unittest.TestCase):
    def test_runtime_window_and_engine_settings_for_all_tp(self):
        for dataset, tokens, blocks in [('longbench-v2', 124688, 1949), ('ruler-thinking', 75536, 1181)]:
            self.assertEqual(allocation(dataset)['kv_blocks'], blocks)
            for tp in (1, 2, 4):
                for cached in (False, True):
                    cfg = config('/model', dataset, cached, '/cache', tp=tp)
                    self.assertEqual(cfg['max_model_len'], tokens)
                    self.assertEqual(cfg['tensor_parallel_size'], tp)
                    self.assertEqual(cfg['cpu_offload_gb'], 0)
                    self.assertNotIn('num_gpu_blocks_override', cfg)
                    self.assertEqual('kv_transfer_config' in cfg, cached)

    def test_memory_uses_block_rounded_kv_without_gpu_name_or_workspace_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = int(61.02420997619629 * 2**30)
            atomic_json(root/'model.safetensors.index.json', dict(metadata=dict(total_size=weights)))
            settings = dict(model=str(root), dataset='longbench-v2', tp=1,
                            groups=[GPUS[:1]], gpu_memory_utilization=.98)
            raw = f'{GPUS[0]},Unlisted GPU,98304,98304'
            with patch('runner.operations.subprocess.check_output', side_effect=[raw, '']):
                receipt = check_hardware(settings)
            self.assertEqual(receipt['kv_bytes_per_rank'], 1949*64*262144)
            self.assertEqual(receipt['required_mib_per_rank'], (weights+1949*64*262144)/.98/2**20)
            self.assertEqual(receipt['workspace_validation'], 'runtime-kv-capacity-audit')
            for inventory, busy in [(raw.replace('98304', '80000'), ''), (raw, GPUS[0])]:
                with patch('runner.operations.subprocess.check_output', side_effect=[inventory, busy]):
                    with self.assertRaises(ValueError): check_hardware(settings)

    def test_tp1_requires_rank_zero_and_enough_measured_blocks(self):
        row = dict(rank=0, kv_tokens=124688, kv_blocks=1949, block_size=64,
                   rope=[dict(formula_bitwise_equal=True, table_positions=131072)]*64)
        validate_initialization([row], 'longbench-v2', tp=1)
        validate_initialization([dict(row, kv_blocks=2000)], 'longbench-v2', tp=1)
        for bad in (dict(row, kv_blocks=1948), dict(row, rank=1), dict(row, kv_tokens=131072)):
            with self.assertRaises(ValueError): validate_initialization([bad], 'longbench-v2', tp=1)
        with self.assertRaises(ValueError): validate_initialization([row, row], 'longbench-v2', tp=1)

    def test_full_prepared_input_uses_runtime_budget_before_measured_capacity_check(self):
        from runner.corpus.runtime import Engine
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp), 'longbench-v2', 4096)
            args.tp = args.number_gpus = 1
            args.devices = GPUS[0]
            path = args.prepared/'samples/0.json'
            sample = json.loads(path.read_text())
            sample.update(token_ids=list(range(114688)), boundaries=boundaries_for(114688,114400,4096),
                          question_positions=[114400,114401])
            stamp_sample(sample); atomic_json(path, sample)
            receipt = json.loads((args.prepared/'preparation.json').read_text())
            receipt['files']['samples/0.json'] = file_hash(path)
            atomic_json(args.prepared/'preparation.json', receipt)
            before = path.read_bytes()
            collection.adopt(args)
            protocol, rows = collection.stage(args.root, 'tree')
            llm = Mock()
            llm.collective_rpc.return_value = [dict(rank=0, kv_tokens=124688, kv_blocks=1948,
                block_size=64, rope=[dict(formula_bitwise_equal=True,table_positions=131072)]*64)]
            for phase in ('cached', 'baseline'):
                with patch.dict(sys.modules, {'runner.runtime.worker': NS(setup=Mock())}), \
                        patch('runner.corpus.runtime.start_engine', return_value=llm) as start, \
                        patch('runner.runtime.generation.generate') as generate:
                    with self.assertRaisesRegex(ValueError, 'capacity'):
                        Engine(args.root/'tree', protocol, 0, phase, phase, rows)
                self.assertEqual(start.call_args.args[0]['max_model_len'], 124688)
                self.assertEqual(start.call_args.args[0]['cpu_offload_gb'], 0)
                generate.assert_not_called()
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(json.loads(before)['max_output_tokens'], 16384)
