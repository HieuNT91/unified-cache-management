"""L20 46068 MiB admission defers estimated workspace to measured KV capacity."""
import copy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from runner.setups import atomic_json
from runner.tree_profiles import config, validate_initialization
from scripts.corpus_control import check_hardware


class L20ThinkingMemoryTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        # Reproduce the reported rounded 46952 MiB static requirement.
        atomic_json(self.root/'model.safetensors.index.json',dict(metadata=dict(total_size=31244*2*2**20)))
        self.settings=dict(dataset='ruler',execution_profile='ruler-thinking',hardware_profile='l20-tp2',
                           model=str(self.root),tp=2,groups=[[str(i),str(i+1)] for i in range(0,10,2)])

    def inventory(self,total=46068,free=46065,name='NVIDIA L20'):
        return '\n'.join(f'{i},{name},{total},{free}' for i in range(10))

    def test_idle_actual_l20_inventory_reaches_runtime_capacity_profiling(self):
        with patch('scripts.corpus_control.subprocess.check_output',side_effect=[self.inventory(),'']):
            result=check_hardware(self.settings)
        self.assertEqual(round(result['estimated_required_mib_per_rank']),46952)
        self.assertLess(result['required_mib_per_rank'],46068)
        self.assertEqual(result['workspace_validation'],'runtime-kv-capacity-audit')
        self.assertEqual(result['groups'][0][0]['free_mib'],46065)
        self.assertEqual(result['workspace_bytes_per_rank'],3*2**30)
        self.assertEqual(result['kv_bytes_per_rank'],1286*64*262144/2)
        cfg=config('/model','ruler-thinking',True,'/cache',hardware='l20-tp2',tp=2)
        self.assertEqual(cfg['gpu_memory_utilization'],.95)
        self.assertEqual(cfg['dtype'],'bfloat16');self.assertEqual(cfg['max_model_len'],82304)
        self.assertNotIn('num_gpu_blocks_override',cfg)

    def test_insufficient_physical_capacity_free_budget_or_busy_gpu_still_rejected(self):
        cases=[(self.inventory(total=42000,free=42000),'','before runtime workspace'),
               (self.inventory(free=43700),'','cannot fit'),
               (self.inventory(free=43750),'','automatic KV budget'),
               (self.inventory(),'9','occupied'),
               (self.inventory(name='NVIDIA A800'),'','explicit hardware profile')]
        for inventory,busy,message in cases:
            with self.subTest(message=message), \
                    patch('scripts.corpus_control.subprocess.check_output',side_effect=[inventory,busy]):
                with self.assertRaisesRegex(ValueError,message): check_hardware(self.settings)

    def test_last_rank_is_checked_and_insufficient_actual_kv_still_fails(self):
        raw=self.inventory().replace('9,NVIDIA L20,46068,46065','9,NVIDIA L20,46068,43750')
        with patch('scripts.corpus_control.subprocess.check_output',side_effect=[raw,'']):
            with self.assertRaisesRegex(ValueError,'GPU 9.*automatic KV budget'):check_hardware(self.settings)
        receipts=[dict(rank=r,kv_tokens=82304,kv_blocks=1286,block_size=64,
            rope=[dict(formula_bitwise_equal=True,table_positions=131072)]*64) for r in range(2)]
        validate_initialization(receipts,'ruler-thinking','l20-tp2',2)
        for blocks in ([1285,1285],[1286,1285],[1286,1287]):
            bad=copy.deepcopy(receipts)
            for row,n in zip(bad,blocks): row['kv_blocks']=n
            with self.subTest(blocks=blocks),self.assertRaisesRegex(ValueError,'capacity'):
                validate_initialization(bad,'ruler-thinking','l20-tp2',2)

    def test_other_profiles_keep_static_workspace_reserve(self):
        for profile,dataset,tp,name in [('l20-tp2','ruler',2,'L20'),('server','longbench-v2',2,'A800'),
                                       ('rtx4500ada','ruler',4,'RTX 4500 Ada Generation')]:
            settings=dict(self.settings,execution_profile=dataset,hardware_profile=profile,tp=tp,
                          groups=[[str(i) for i in range(tp)]])
            with self.subTest(profile=profile),patch('scripts.corpus_control.subprocess.check_output',
                    side_effect=[self.inventory(100000,100000,'NVIDIA '+name),'']):
                result=check_hardware(settings)
            self.assertNotIn('workspace_validation',result)
            expected=(result['weights_bytes_per_rank']+result['kv_bytes_per_rank']+result['workspace_bytes_per_rank'])/result['gpu_memory_utilization']/2**20
            self.assertEqual(result['required_mib_per_rank'],expected)
            self.assertNotIn('free_mib',result['groups'][0][0])

    def test_engine_stops_before_warmup_when_measured_kv_is_insufficient(self):
        from runner.corpus_runtime import Engine
        from test_l20_traj4f import fixture, GPUS
        from scripts.ruler import TASKS
        args=fixture(self.root/'engine',n=1)
        protocol=dict(dataset='ruler',execution_profile='ruler-thinking',hardware_profile='l20-tp2',
            tp=2,groups=[GPUS[:2]],prepared=str(args.prepared),model=str(args.model),cache_root=str(args.cache_root))
        rows=[dict(id=f'{TASKS[0]}-000',prepared=f'samples/{TASKS[0]}-000.json')]
        llm=Mock()
        llm.collective_rpc.return_value=[dict(rank=r,kv_tokens=82304,kv_blocks=1285,block_size=64,
            rope=[dict(formula_bitwise_equal=True,table_positions=131072)]*64) for r in range(2)]
        with patch.dict(sys.modules,{'runner.worker':NS(setup=Mock())}),patch.dict(os.environ,{}), \
                patch('runner.corpus_runtime.start_engine',return_value=llm), \
                patch('scripts.router_control.code_hashes',return_value={}), \
                patch('runner.generation.generate') as generate:
            with self.assertRaisesRegex(ValueError,'capacity'):
                Engine(args.root,protocol,0,'cached','insufficient',rows)
        generate.assert_not_called();llm.llm_engine.engine_core.shutdown.assert_called_once()
        self.assertFalse(list(args.root.rglob('initialization.json')))


if __name__ == '__main__': unittest.main()
