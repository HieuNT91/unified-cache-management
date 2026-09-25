"""CPU regression checks for the new protocol, TP selection and reports."""
import copy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import common
from common import dump, load, PARAMETERS
import prepare
import report
import suite
from prophetkv_common import score
from selection_reference import native_reference


class RemoteTests(unittest.TestCase):
    def test_native_thinking_template_validation(self):
        class Tokenizer:
            def __init__(self, thinking_tail, off_tail='<think>\n\n</think>\n\n'):
                self.thinking_tail, self.off_tail = thinking_tail, off_tail
            def apply_chat_template(self, messages, **kwargs):
                return ('<|im_start|>user\nTest<|im_end|>\n<|im_start|>assistant\n' +
                        (self.thinking_tail if kwargs['enable_thinking'] else self.off_tail))
        for tail in ('', '<think>\n'):
            suite.validate_thinking_template(Tokenizer(tail))
        for tokenizer in (Tokenizer('<think>\n</think>'), Tokenizer('', ''), Tokenizer('answer')):
            with self.assertRaisesRegex(ValueError, 'thinking chat mode'):
                suite.validate_thinking_template(tokenizer)

    def test_restricted_pairs_and_scope(self):
        environment=dict(MODEL_PATH='/model',RESULT_ROOT='/results',CACHE_ROOT='/cache',
                         RULER_ROOT='/ruler',LONGBENCH_DATA='/lb.json')
        with patch.dict(os.environ,environment):
            for job,indices in [('0',[0,1]),('1',[2,3])]:
                os.environ['EXPANSION_JOB']=job
                cfg=common.settings()
                self.assertEqual(cfg['indices'],indices)
                self.assertEqual(common.CASES,('prophetkv-10','prophetkv-20','prophetkv-40','prophetkv-50','prophetkv-60'))
                self.assertEqual(cfg['max_output_tokens'],16384)
            os.environ['EXPANSION_JOB']='2'
            with self.assertRaises(ValueError): common.settings()
        devices={i:dict(index=i,uuid=f'GPU-{i}',name='A800',memory_mib=81920) for i in range(8)}
        with patch.object(common,'inventory',return_value=('',devices)):
            for indices in ([4,5],[6,7],[0,2],[0,1,2,3]):
                with self.assertRaises(RuntimeError):common.selected_devices(indices)
            self.assertEqual(len(common.selected_devices([0,1])[1]),2)

    def test_thinking_scoring_and_reject_old_inputs(self):
        sample=dict(dataset='longbench_v2',thinking_enabled=True,source_metadata={'answer':'B'})
        self.assertEqual(score(sample,'The correct answer is (B)')[0],0)
        self.assertEqual(score(sample,'The correct answer is (A)</think>The correct answer is (B)')[0],1)
        self.assertEqual(score(sample,'The correct answer is (B)</think>The correct answer is (A)')[0],0)
        from prophetkv_common import load_sample
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sample.json'
            dump(path,dict(dataset='longbench_v2',thinking_enabled=False,max_output_tokens=256))
            with self.assertRaisesRegex(ValueError,'thinking'):load_sample(path)

    def test_rope_original_entries_and_extra_positions(self):
        from rope_window import ensure_rope_window
        freq=torch.tensor([.1,.01])
        positions=torch.arange(65536,dtype=torch.float32)
        angles=torch.einsum('i,j -> ij',positions,freq)
        original=torch.cat((angles.cos()*1.1,angles.sin()*1.1),dim=-1)
        rope=SimpleNamespace(scaling_factor=2.,max_position_embeddings=32768,
            cos_sin_cache=original.clone(),mscale=1.1,_compute_inv_freq=lambda _:freq)
        audit=ensure_rope_window(rope,81920)
        self.assertTrue(torch.equal(original,rope.cos_sin_cache[:65536]))
        self.assertEqual(audit['added_positions'],16384)
        self.assertEqual(ensure_rope_window(rope,81920),audit)
        with self.assertRaises(ValueError):ensure_rope_window(rope,81984)

    def test_tp_global_scores_precede_native_ranking(self):
        sys.path.insert(0,str(common.HERE/'method'))
        from prophetkv.runtime import global_selection
        local=torch.arange(384,dtype=torch.float32)
        other=local.flip(0)*2
        group=SimpleNamespace(world_size=2,broadcast=lambda selected,src:selected)
        mock=SimpleNamespace(tensor_model_parallel_all_reduce=lambda scores:scores+other,
                             get_tp_group=lambda:group)
        from prophetkv.selection import select,RequestMetadata
        RequestMetadata('r',(0,128,256,1127),(300,301))
        with patch.dict(sys.modules,{'vllm.distributed':mock}):
            for parameters in PARAMETERS.values():
                eligible,actual,details=global_selection(local,128,parameters['total_ratio'])
                expected=select((local+other)[128:]/2,128,parameters['total_ratio'])
                self.assertTrue(torch.equal(actual,expected))
                self.assertEqual(len(actual),int(256*parameters['total_ratio']))

    def test_native_reference_ties_and_floor(self):
        sys.path.insert(0,str(common.HERE/'method'))
        from prophetkv.selection import select
        for scores in ([1.]*17, [0.,4.,4.,2.,4.,0.,3.]):
            for parameters in PARAMETERS.values():
                expected,_=native_reference(scores,128,parameters)
                actual=select(torch.tensor(scores),128,parameters['total_ratio']).tolist()
                self.assertEqual(actual,expected)
        self.assertEqual(native_reference([1.]*17,128,{'total_ratio':.1})[0],[128])

    def test_report_preserves_tasks_and_paired_denominator(self):
        def row(sid,case,ttft,label,score_value):
            return dict(sample_id=sid,case=case,ttft_seconds=ttft,label=label,score=score_value,
                dataset='longbench_v2',generation_seconds=10,output_tokens=100,cache_build_seconds=2,
                prime_seconds=1,prompt_tokens=10000,finish_reason='stop',thinking_enabled=False,
                prediction='The correct answer is (B)')
        rows=[row('a','baseline',8,'longbench_v2_non_thinking',1),
              row('a','prophetkv-10',2,'longbench_v2_non_thinking',0),
              row('b','prophetkv-10',20,'longbench_v2_non_thinking',1),
              row('c','baseline',1,'qa_2',1)]
        table=report.summaries(rows)
        selected=next(r for r in table if r['task']=='longbench_v2_non_thinking' and r['method']=='prophetkv-10')
        self.assertEqual(selected['paired_samples'],1)
        self.assertEqual(selected['paired_mean_ttft_speedup'],4)
        self.assertEqual(selected['samples'],2)
        self.assertEqual(selected['accuracy_percent'],50)

    def make_record(self,root):
        bounds=[0,128,256,556]
        tokens=list(range(128))*2+[1]*300
        tokens[127]=tokens[255]=999
        sample=dict(id='sample',dataset='longbench_v2',label='longbench_v2_thinking',context_target=65536,
            thinking_enabled=True,max_output_tokens=16384,tokens=556,token_ids=tokens,boundaries=bounds,
            fresh_suffix_tokens=300,query={'positions':[260,261]},source_metadata={'answer':'B','references':['B']})
        ip=root/'sample.json';dump(ip,sample)
        meta={k:v for k,v in sample.items() if k!='token_ids'}|dict(input_path=str(ip),input_sha256=prepare.sha(ip))
        devices=[dict(uuid='GPU-0'),dict(uuid='GPU-1')]
        p=dict(case_parameters=PARAMETERS,gpu_devices=devices,tensor_parallel_size=2,num_layers=64)
        dump(root/'protocol.json',p)
        audit=dict(table_positions=81920,factor=2.,original_entries_bitwise_equal=True,extended_formula_bitwise_equal=True)
        sp=root/'session.json'
        dump(sp,dict(setup_receipts=[{'rope_windows':[audit]*64}]*2,
            warmup_cache_readiness=dict(complete=True,verified_shards=4)))
        rid='a'*32+':measured';case='prophetkv-20'
        scores=list(range(128))
        selected,details=native_reference(scores,128,PARAMETERS[case])
        selection=dict(kind='prophetkv_selection',request_id=rid,eligible_count=128,selected_count=len(selected),
            probe_layers=64,alignment_count=64,question_positions=[260,261],scores=scores,
            selected_positions=selected,method='prophetkv',parameters=PARAMETERS[case],
            **{k:v for k,v in details.items() if k!='eligible_count'})
        layers=[dict(kind='layer_counts',request_id=rid,selected_positions=selected,
            layer=f'model.layers.{i}.self_attn.attn',projection_tokens=len(selected)+300,
            attention_tokens=len(selected)+300,ffn_tokens=len(selected)+300) for i in range(64)]
        workers=[dict(rank=i,visible_uuid=f'GPU-{i}',ucm_path=str(root/'private_ucm/ucm/__init__.py')) for i in range(2)]
        record=dict(input_sha256=meta['input_sha256'],protocol_sha256=prepare.sha(root/'protocol.json'),
            sample_id='sample',case=case,prompt_tokens=556,label='longbench_v2_thinking',context_target=65536,max_output_tokens=16384,
            smoke=False,thinking_enabled=True,method='prophetkv',parameters=PARAMETERS[case],
            timing_source=common.TIMING,cache_unchanged=True,online_mask_reused=False,gpu_devices=devices,tensor_parallel_size=2,
            ttft_seconds=1.,generation_seconds=2.,cache_build_seconds=1.,prime_seconds=1.,cache_readiness_seconds=0.,
            artifact_export_seconds=.1,retirement_seconds=.1,session_path=str(sp),output_tokens=1,output_token_ids=[1],
            references=['B'],score=1.,prediction='reasoning</think>The correct answer is (B)',finish_reason='stop',worker_imports=workers,
            request_id=rid,engine_policy=common.ENGINE_POLICY,namespace='a'*32,max_model_len=81920,
            rope_group=common.engine_group(65536),cache_files_after_retirement=0,
            retirement=dict(workers=[dict(rank=i,quiescent=True,transfers={'pending':0},request_bookkeeping=0) for i in range(2)],
                scheduler=dict(request_id=rid,requests_blend_meta=0,requests_meta=0)),
            cache_verification=dict(namespace='a'*32,complete=True,verified_shards=8,expected_unique_blocks=4))
        path=root/'prophetkv-20.json';dump(path,record)
        dump(path.with_suffix('.diagnostics.json'),[dict(rank=i,diagnostics=[selection,*layers]) for i in range(2)])
        path.with_suffix('.log').write_text('MEASURE_BEGIN\nrequest_id: '+rid+', req_stage: BlendStage.CACHE_BLEND, first chunk prefix hit: 2, chunks cache total hit: 2\nMEASURE_END\nWORKER_COMPLETE\n')
        return p,meta,case,path

    def test_full_tp_validation_and_artifact_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p,meta,case,path=self.make_record(root)
            suite.accept(root,p,meta,case,path)
            self.assertTrue(suite.valid(root,p,meta,case,path))
            diagnostics=load(path.with_suffix('.diagnostics.json'))
            diagnostics[1]['diagnostics'][3]['selected_positions']=[]
            dump(path.with_suffix('.diagnostics.json'),diagnostics)
            with self.assertRaises(ValueError):suite.validate(root,p,meta,case,path)
            with self.assertRaisesRegex(ValueError,'artifacts changed'):suite.valid(root,p,meta,case,path)

    def test_reject_incomplete_warmup(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p,meta,case,path=self.make_record(root)
            session=load(root/'session.json');session['warmup_cache_readiness']['verified_shards']=2
            dump(root/'session.json',session)
            with self.assertRaisesRegex(ValueError,'warmup'):suite.validate(root,p,meta,case,path)

    def test_formatted_longbench_boundary_is_inclusive(self):
        class Tokenizer:
            pad_token_id=0
            def apply_chat_template(self,messages,**kwargs):
                assert kwargs['enable_thinking'] is True
                return '<user>'+messages[0]['content']+'<|im_start|>assistant\n'
            def encode(self,text,**kwargs):return [ord(c)+1 for c in text]
            def __call__(self,text,**kwargs):
                result={'input_ids':self.encode(text)}
                if kwargs.get('truncation'):
                    result['input_ids']=result['input_ids'][:kwargs['max_length']]
                if kwargs.get('return_offsets_mapping'):
                    result['offset_mapping']=[(i,i+1) for i in range(len(text))]
                return result
        row=dict(_id='id',context='x'*62000,question='Choose the right answer.',
                 choice_A='a',choice_B='b',choice_C='c',choice_D='d',answer='B',domain='domain')
        tokenizer=Tokenizer()
        sample,_=prepare.encode_longbench(tokenizer,row,0,True)
        exact=sample['tokens']
        accepted,receipt=prepare.encode_longbench(tokenizer,row,0,True,limit=exact)
        self.assertTrue(receipt['eligible'])
        self.assertEqual(accepted['tokens'],exact)
        self.assertEqual(accepted['max_output_tokens'],16384)
        self.assertTrue(all(accepted['boundaries'][-2]<=i<exact for i in accepted['query']['positions']))
        rejected,receipt=prepare.encode_longbench(tokenizer,row,0,True,limit=exact-1)
        self.assertIsNone(rejected)
        self.assertFalse(receipt['eligible'])

    def test_decoding_parameters_and_first_token_timing(self):
        from prophetkv_common import generate
        parameters=[]
        class Engine:
            def __init__(self):self.steps=0
            def has_unfinished_requests(self):return 0<self.steps<3
            def add_request(self,rid,prompt,sampling):
                self.steps=1
                parameters.append(sampling)
            def step(self):
                self.steps+=1
                return [SimpleNamespace(request_id='r',outputs=[SimpleNamespace(token_ids=[7])],finished=self.steps==3)]
        fake={'vllm':SimpleNamespace(SamplingParams=lambda **kwargs:kwargs),
              'vllm.inputs':SimpleNamespace(TokensPrompt=lambda **kwargs:kwargs),
              'vllm.sampling_params':SimpleNamespace(RequestOutputKind=SimpleNamespace(CUMULATIVE='cumulative'))}
        with patch.dict(sys.modules,fake):
            for thinking,budget in [(False,4),(True,16384)]:
                _,ttft,total=generate(Engine(),[1,2],budget,'r',thinking=thinking)
                self.assertLessEqual(ttft,total)
                self.assertEqual(parameters[-1]['max_tokens'],budget)
                self.assertEqual(parameters[-1]['temperature'],.6 if thinking else 0.)
                self.assertEqual(parameters[-1]['top_k'],20 if thinking else -1)
            for thinking,budget in [(True,16385),(False,256)]:
                with self.assertRaisesRegex(ValueError,'Thinking'):
                    generate(Engine(),[1,2],budget,'r',thinking=thinking)

    def test_all_methods_share_rope_capacity_and_tp(self):
        from worker import build
        fake={'vllm':SimpleNamespace(LLM=lambda **kwargs:kwargs),
              'vllm.config':SimpleNamespace(KVTransferConfig=lambda **kwargs:kwargs)}
        p=dict(model='/model',gpu_memory_utilization=.9,tensor_parallel_size=2,
               rope_scaling_64k=common.rope_for_length(65536),case_parameters=PARAMETERS)
        sample=dict(context_target=65536,token_ids=[1]*128+[0],boundaries=[0,129])
        with patch.dict(sys.modules,fake):
            for case in common.CASES:
                cfg=build(SimpleNamespace(case=case,cache_dir=Path('/cache'),persistent=True),sample,p)
                self.assertEqual(cfg['max_model_len'],81920)
                self.assertEqual(cfg['num_gpu_blocks_override'],1280)
                self.assertEqual(cfg['tensor_parallel_size'],2)
                self.assertEqual(cfg['rope_scaling']['factor'],2.)
                self.assertFalse(cfg['enable_prefix_caching'])
                if case=='baseline':self.assertNotIn('kv_transfer_config',cfg)
                else:
                    sparse=cfg['kv_transfer_config']['kv_connector_extra_config']['ucm_sparse_config']['ProphetKV']
                    self.assertEqual(sparse['method'],'prophetkv')
                    self.assertNotIn('expansion',sparse)
                    self.assertEqual(sparse['ratio'],PARAMETERS[case]['total_ratio'])

    def test_census_balances_sources_and_requires_thinking(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'data.json'
            dump(source,[dict(_id=str(i)) for i in range(503)])
            # Sparse source row IDs must not cause uneven parity sharding.
            eligible={2,4,8,20,24,30}
            def encode(tokenizer,row,ordinal,thinking):
                self.assertTrue(thinking)
                if ordinal not in eligible:
                    return None,dict(eligible=False)
                return dict(source_row=ordinal,thinking_enabled=thinking),dict(eligible=True,tokens=10000)
            samples=[]
            with patch.object(prepare,'encode_longbench',side_effect=encode),patch('builtins.print'):
                census=prepare.longbench_census(None,source,samples.append)
            self.assertEqual(census['counts'],dict(thinking=6))
            self.assertEqual([s['assigned_job'] for s in samples],[0,1,0,1,0,1])
            for ordinal in eligible:
                self.assertEqual(len({s['assigned_job'] for s in samples if s['source_row']==ordinal}),1)

    def test_combined_matrix_only_thinking(self):
        shared=dict(sources_sha256={},runtime_versions={},tokenizer_sha256={},case_parameters=PARAMETERS,
                    cases=list(common.CASES),decoding={'thinking':{'temperature':.6}},
                    longbench_counts={'thinking':184},max_model_len=81920)
        samples=[dict(id=f'lb-{i}',label='longbench_v2_thinking',source_row=i,dataset='longbench_v2',
                         thinking_enabled=True,max_output_tokens=16384) for i in range(184)]
        protocols=[dict(**shared,settings={'job_id':str(job)},
                        samples=[m for m in samples if m['source_row']%2==job]) for job in range(2)]
        self.assertEqual(report.check_matrix(protocols),920)
        self.assertEqual([len(p['samples'])*5 for p in protocols],[460,460])
        missing=copy.deepcopy(protocols)
        missing[0]['samples'].pop()
        with self.assertRaisesRegex(ValueError,'coverage'):report.check_matrix(missing)
        protocols[0]['samples'][0]['thinking_enabled']=False
        with self.assertRaisesRegex(ValueError,'thinking'):report.check_matrix(protocols)

    def test_empty_partial_report(self):
        with tempfile.TemporaryDirectory() as folder:
            report.write_report(Path(folder),[],920,False,[])
            self.assertEqual(len(report.summaries([])),5)

if __name__=='__main__':unittest.main()
