"""CPU checks for variable suffix probing and immutable per-method prompt scheduling."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
import tempfile
from contextlib import nullcontext
from unittest.mock import patch
import torch
from prophetkv_gpu import ROOT
from settings import PARENT, CASES, TASKS

sys.path.insert(0,str(ROOT/'private_ucm/ucm/sparse'))
from prophetkv.selection import ExpansionConfig,RequestMetadata,request_mask
import prophetkv.runtime as current
from warmup_readiness import wait_for_warmup_cache


class VariableSuffixTests(unittest.TestCase):
    def test_metadata_and_masks_keep_complete_suffix(self):
        for size in (256,257,871):
            meta=RequestMetadata('r',(0,64,128,128+size),(128,128+size-1))
            positions=torch.arange(64,128+size)
            chosen=torch.tensor([66,78])
            mask=request_mask(positions,chosen,64,128,torch.tensor([True]))
            self.assertTrue(mask[-size:].all())
            self.assertEqual(positions[mask][:-size].tolist(),chosen.tolist())
        with self.assertRaises(ValueError):RequestMetadata('r',(0,64,128,383),(128,382))

    def run_probe(self,module,suffix):
        ns=types.SimpleNamespace
        layers=[ns(self_attn=ns(o_proj=lambda x:(x,None)),post_attention_layernorm=lambda h,r:(h,r),mlp=lambda h:h) for _ in range(36)]
        cache=torch.ones(2,2,64,1,2)
        context=ns(virtual_engine=0,no_compile_layers={f'model.layers.{i}.self_attn.attn':ns(kv_cache=[cache]) for i in range(36)})
        forward=types.ModuleType('vllm.forward_context');forward.get_forward_context=lambda:context
        names=set();connector=ns(prophet_aligned=names,wait_for_layer_load=lambda n:names.add(n))
        sparse=ns(request=ns(boundaries=(0,64,128,128+suffix),request_id='r',question_positions=(128,128+suffix-1)),
            attn_metadata=ns(block_table=torch.tensor([[0,1]])),model=ns(layers=layers),connector=connector,
            method='prophetkv_with_expansion',ratio=.4,expansion_config=ExpansionConfig(total_ratio=.4,anchor_ratio=.2,max_gap=4),selection_diagnostics=[])
        lengths=[]
        def project(layer,hidden,residual,positions):
            lengths.append(len(hidden));v=hidden.reshape(-1,1,2)
            return v,v,v,hidden
        with patch.dict(sys.modules,{'vllm.forward_context':forward}),patch.object(module,'project',project),patch.object(torch.cuda,'synchronize',lambda:None):
            selected=module.probe(sparse,torch.arange(64,128+suffix),torch.ones(64+suffix,2))
        self.assertEqual(lengths,[suffix]*36)
        event=sparse.selection_diagnostics[0]
        self.assertEqual(event['alignment_count'],36)
        self.assertEqual(event['suffix_tokens'],suffix)
        self.assertEqual(event['probe_tokens_per_layer'],suffix)
        self.assertEqual(len(selected),25)
        return selected,event['scores']

    def test_original_256_results_unchanged_and_long_query_supported(self):
        spec=importlib.util.spec_from_file_location('prophetkv._old_runtime',PARENT/'private_ucm/ucm/sparse/prophetkv/runtime.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        original=self.run_probe(old,256)
        updated=self.run_probe(current,256)
        self.assertTrue(torch.equal(original[0],updated[0]))
        self.assertTrue(torch.equal(original[1],updated[1]))
        self.run_probe(current,871)

    def test_complete_matrix_and_question_containment(self):
        p=json.loads((ROOT/'protocol.json').read_text())
        from checks import configuration_checks
        result=configuration_checks()
        self.assertEqual(result['scheduled_new_measurements'],p['measured_requests'])
        for sample in p['samples']:
            b=sample['boundaries'];q=sample['query']['positions']
            self.assertGreaterEqual(min(q),b[-2]);self.assertLess(max(q),b[-1])
            self.assertEqual(sample['fresh_suffix_tokens'],b[-1]-b[-2])
            if sample['dataset']=='longbench_v2':self.assertLess(sample['tokens'],65536)
        self.assertEqual(p['max_output_tokens'],256)

    def test_final_report_covers_seven_methods_and_longbench(self):
        import report
        from prophetkv_common import dump
        rows=[]
        for task in (*TASKS,'longbench_v2'):
            for i,case in enumerate(CASES):
                rows.append(dict(sample_id=task,label=task,dataset='longbench_v2' if task=='longbench_v2' else 'ruler',
                    source_row=0,case=case,physical_gpu=1,prompt_tokens=10000,score=.5,
                    ttft_seconds=10. if i==0 else 5.,generation_seconds=11.,cache_build_seconds=1.,
                    cache_readiness_seconds=.1,prime_seconds=1.,artifact_export_seconds=.1,
                    retirement_seconds=.1,output_tokens=256,finish_reason='length'))
        p=json.loads((ROOT/'protocol.json').read_text());p.update(samples=[{}]*8,measured_requests=56,longbench_count=1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dump(root/'protocol.json',p)
            dump(root/'supervisor.json',dict(pid=-1,identity={'fake':True},state='complete'))
            dump(root/'cleanup.json',dict(complete=True,owned_workers_exited=True))
            dump(root/'preserved-prior.json',{})
            with patch.object(report,'ROOT',root),patch.object(report,'REPO',root),\
                 patch.object(report,'verify',lambda:p),patch.object(report,'lock',nullcontext),\
                 patch.object(report,'ensure_no_workers',lambda:None),patch.object(report,'identity',lambda _:None),\
                 patch.object(report,'verify_preserved',lambda:1),patch.object(report.val,'all_records',lambda _:rows):
                result=report.finalize()
            self.assertEqual(len(result),70)
            self.assertEqual(len([r for r in result if r['task']=='longbench_v2']),7)
            self.assertTrue(all(r['paired_mean_ttft_speedup']==2. for r in result if r['method']!='baseline'))
            self.assertEqual(json.loads((root/'final/validation.json').read_text())['validated'],56)
            self.assertTrue((root/'prophetkv_expansion_half_gap4_results.txt').exists())


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
