from runner.layout import request_metadata, stamp_sample
"""CPU regressions for real chunk state/connector paths and causal execution."""
import ast
import copy
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from ucm.sparse.prophetkv.prefill import PrefillProgress, prefill_step
from ucm.sparse.prophetkv.selection import RequestMetadata, RequestState, LayerAlignment, select
from ucm.sparse.prophetkv.attention import dense_reference

NS = types.SimpleNamespace


class ProgressTests(unittest.TestCase):
    def test_boundaries_and_single_token_final_prefill(self):
        for length in (384, 16383, 16384, 16385, 32769, 65537):
            for prefix in (0, 64):
                steps = []
                for start in range(prefix, length, 16384):
                    count = min(16384, length-start)
                    self.assertEqual(prefill_step(start, count, length), (True, start+count == length))
                    steps.append(count)
                self.assertEqual(sum(steps), length-prefix)
                self.assertEqual(prefill_step(length, 1, length), (False, False))
        self.assertEqual(prefill_step(16384, 1, 16385), (True, True))
        with self.assertRaises(RuntimeError):
            prefill_step(0, 16385, 20000)

    def test_global_mask_empty_ranges_suffix_crossings_and_no_duplicates(self):
        for ratio in (0., .2, 1.):
            prefix, suffix, length = 64, 33024, 50001
            chosen = select(torch.ones(suffix-prefix), prefix, ratio).tolist()
            state = PrefillProgress(prefix, suffix, length)
            state.select_once(chosen)
            actual = []
            counts = []
            for start in range(prefix, length, 16384):
                positions = state.step(start, min(16384, length-start))
                actual.extend(positions);counts.append(len(positions))
            self.assertEqual(actual, chosen+list(range(suffix, length)))
            self.assertEqual(len(actual), len(set(actual)))
            if ratio == 0:
                self.assertEqual(counts[:2], [0, 0])
            with self.assertRaises(RuntimeError):
                state.select_once(chosen)
            with self.assertRaises(RuntimeError):
                state.step(prefix, 1)

    def test_deterministic_multilayer_causal_reference_and_repaired_kv(self):
        # Small analogue with gaps, empty windows and a suffix boundary crossing.
        torch.manual_seed(6)
        n, prefix, suffix, width, layers = 41, 4, 31, 4, 3
        initial = torch.randn(n, 2, width)
        weights = [torch.randn(width, width)*.2 for _ in range(layers)]
        caches = [(torch.randn(n, 2, width), torch.randn(n, 2, width)) for _ in range(layers)]
        for ratio in (0., .3, 1.):
            chosen = select(torch.ones(suffix-prefix), prefix, ratio).tolist()
            all_positions = chosen+list(range(suffix, n))
            def run(ranges):
                kv = [(k.clone(), v.clone()) for k, v in caches]
                final = {}
                for start, end in ranges:
                    pos = torch.tensor([p for p in all_positions if start <= p < end])
                    if not len(pos):
                        continue
                    h = initial[pos].clone()
                    for i, w in enumerate(weights):
                        q, k, v = h@w, h@w+.1, h@w-.1
                        kv[i][0][pos] = k;kv[i][1][pos] = v
                        h = dense_reference(q, kv[i][0][:end], kv[i][1][:end], pos)+h
                    final.update({int(p): value for p, value in zip(pos, h)})
                return torch.stack([final[p] for p in all_positions]), kv
            reference, reference_kv = run([(prefix, n)])
            actual, actual_kv = run([(a, min(a+7, n)) for a in range(prefix, n, 7)])
            torch.testing.assert_close(actual, reference)
            for actual_pair, expected_pair in zip(actual_kv, reference_kv):
                for actual_cache, expected_cache in zip(actual_pair, expected_pair):
                    torch.testing.assert_close(actual_cache, expected_cache)


class WorkerStateTests(unittest.TestCase):
    @staticmethod
    def sparse():
        from ucm.sparse.prophetkv.prophetkv import ProphetKV
        from ucm.sparse.blend.blend import BlendMetaData
        sparse = ProphetKV.__new__(ProphetKV)
        sparse.request_state = RequestState()
        sparse.prefill_state = None
        sparse.block_size = 64
        sparse.forward_mask = torch.zeros(65536, dtype=torch.bool)
        sparse.blend_req_metas = BlendMetaData()
        sparse.selection_diagnostics = []
        sparse.model = NS(get_input_embeddings=lambda ids: ids.float()[:, None])
        sparse.connector = NS(retire_worker_request=Mock())
        return sparse

    @staticmethod
    def entry(start, count, length, first=False, hit=True, reserved=True):
        blocks = list(range((length+63)//64)) if reserved else list(range((start+count+63)//64))
        layout_sample=stamp_sample(dict(token_ids=list(range(length)),boundaries=[0,64]+list(range(4160,33088,4096))+[33088,length],question_positions=[33089,length-1]))
        dispatch = NS(request_layout=request_metadata('r',layout_sample['token_ids'],'read',layout_sample),chunks_meta=[NS(start_token_dix=64,store_hits=[hit]*516)] if first else [],
                      full_block_ids=tuple(blocks),load_block_ids=([],list(range(517))))
        scheduler = NS(num_scheduled_tokens={'r': count}, scheduled_new_reqs=[NS(req_id='r')] if first else [],
            scheduled_cached_reqs=NS(req_ids=[] if first else ['r']),
            kv_connector_metadata=NS(request_meta={'r': dispatch}))
        request = NS(num_computed_tokens=start,prompt_token_ids=list(range(length)),block_ids=[blocks])
        attention = NS(query_start_loc=torch.tensor([0,count]),seq_lens=torch.tensor([start+count]),
                       block_table=torch.tensor(blocks)[None],slot_mapping=torch.arange(start,start+count))
        return scheduler, {'r': request}, None, attention

    def test_real_worker_mask_persists_and_empty_step_has_all_layer_events(self):
        from ucm.sparse.prophetkv import prophetkv
        sparse = self.sparse()
        length = 64+3*16384+1
        sparse.request_state.arm(RequestMetadata('r',(0,64,4160,8256,12352,16448,20544,24640,28736,32832,33088,length),(33089,length-1)))
        selected = torch.tensor([16450,33000])
        with patch.object(prophetkv, 'probe', return_value=selected) as probe:
            covered = []
            for start in range(64,length,16384):
                count = min(16384,length-start)
                sparse.build_sparse_meta(*self.entry(start,count,length,start==64))
                mask = sparse.prepare_step(torch.arange(start,start+count))
                self.assertLessEqual(len(mask),16384)
                covered.extend(torch.arange(start,start+count)[mask].tolist())
                if not mask.any():
                    events = [e for e in sparse.selection_diagnostics if e['kind']=='layer_counts']
                    self.assertEqual(len(events),64)
                    self.assertTrue(all(len(e['selected_positions'])==0 for e in events))
                self.assertEqual(sparse.attn_metadata.num_actual_tokens if not mask.all() else count, int(mask.sum()))
            probe.assert_called_once()
        self.assertEqual(covered,selected.tolist()+list(range(33088,length)))
        # Completion/cancellation releases masks, saved tokens and mappings.
        sparse.request_finished_in_worker('r')
        self.assertIsNone(sparse.prefill_state)
        self.assertIsNone(sparse.coverage_selected)
        self.assertIsNone(sparse.prompt_token_ids)
        sparse.connector.retire_worker_request.assert_called_once_with('r')

    def test_missing_hits_slots_and_mapping_changes_fail(self):
        for hits,reserved in [(False,True),(True,False)]:
            sparse = self.sparse()
            sparse.request_state.arm(RequestMetadata('r',(0,64,4160,8256,12352,16448,20544,24640,28736,32832,33088,40000),(33089,39999)))
            with self.assertRaises(RuntimeError):
                sparse.build_sparse_meta(*self.entry(64,16384,40000,True,hits,reserved))
        sparse = self.sparse()
        sparse.request_state.arm(RequestMetadata('r',(0,64,4160,8256,12352,16448,20544,24640,28736,32832,33088,40000),(33089,39999)))
        sparse.build_sparse_meta(*self.entry(64,16384,40000,True))
        entry = self.entry(16448,16384,40000)
        entry[1]['r'].block_ids[0][0] = 999
        with self.assertRaises(RuntimeError):
            sparse.build_sparse_meta(*entry)

    def test_cancel_before_first_step(self):
        sparse = self.sparse()
        sparse.request_state.arm(RequestMetadata('r',(0,64,128,384),(380,)))
        sparse.request_finished_in_worker('r')
        self.assertIsNone(sparse.request_state.pending)

    def test_causal_attention_gathers_only_current_endpoint(self):
        sparse = self.sparse()
        sparse.request_state.arm(RequestMetadata('r',(0,64,4160,8256,12352,16448,20544,24640,28736,32832,33088,40000),(33089,39999)))
        sparse.build_sparse_meta(*self.entry(64,16384,40000,True))
        sparse.current_positions=sparse.step_positions=torch.tensor([100,110])
        sparse.projection_count=2;sparse.audit=False
        context=NS(no_compile_layers={'layer':NS()})
        with patch('ucm.sparse.prophetkv.attention.install'):
            sparse.attention_begin(torch.zeros(2,1,2),None,None,'layer',context)
        self.assertEqual(len(sparse.fusion_slots),16448)
        self.assertEqual(sparse.fusion_k_len.tolist(),[16448])


class ConnectorTests(unittest.TestCase):
    def connector(self):
        from ucm.integration.vllm.persistent_connector import PersistentBlendConnector
        c = PersistentBlendConnector.__new__(PersistentBlendConnector)
        c.prompt_lengths={'r':33000};c.dispatches={};c.requests_blend_meta={'r':NS()}
        c.worker_request_id=None;c.prophet_aligned=set();c.block_size=64
        c.store=NS(drain=Mock());c.requests_meta={};c.req2rag_load_chunks={}
        return c

    def test_repeated_model_setup_preserves_normalized_yarn_delta_table(self):
        from ucm.sparse.prophetkv.runtime import delta_rotation_table
        # Model Q/K already carry YaRN's magnitude. Cache relocation must use
        # a unit-magnitude delta rotation even when setup runs every step.
        angles=torch.tensor([[0.,0.],[.25,.5]])
        for factor in (2.,4.):
            with self.subTest(factor=factor):
                magnitude=1.+.1*torch.log(torch.tensor(factor)).item()
                raw=torch.cat((angles.cos(),angles.sin()),dim=-1)*magnitude
                original=raw.clone()
                model=NS(model=NS(layers=[NS(self_attn=NS(rotary_emb=NS(cos_sin_cache=raw)))]))
                c=self.connector()
                c.setup_model(model)
                self.assertIs(c.cos_sin_cache,raw)
                c.cos_sin_cache=normalized=delta_rotation_table(c.cos_sin_cache)
                c.prophet_delta_normalized=True
                for _ in range(3):
                    c.setup_model(model)
                    self.assertIs(c.cos_sin_cache,normalized)
                    cos,sin=c.cos_sin_cache.chunk(2,dim=-1)
                    torch.testing.assert_close(cos.square()+sin.square(),torch.ones_like(cos))
                torch.testing.assert_close(raw,original,rtol=0,atol=0)

    def test_full_reservation_continuation_no_reload_or_double_alignment(self):
        from ucm.integration.vllm.blend_connector import BlendRequestDispatchMeta, UCMBlendConnector
        c=self.connector()
        c.active_layout=dict(request_id='r',phase='read')
        c.requests_blend_meta['r']=NS(blend_stage=NS(is_blend_cache=lambda:True))
        self.assertEqual(c.reservation_tokens(NS(request_id='r',num_prompt_tokens=33000),16448),33000)
        blocks=list(range(516))
        first=NS(scheduled_new_reqs=[NS(req_id='r',block_ids=[blocks])],
                 num_scheduled_tokens={'r':16384},scheduled_cached_reqs=NS(req_ids=[]))
        payload=BlendRequestDispatchMeta(([b'one',b'two'],[1,2]),([],[]),[])
        with patch.object(c,'_generate_blend_dispatch_meta',return_value=payload):
            meta=c.build_connector_meta(first)
        writes=[]
        guard=LayerAlignment();c.prophet_aligned=guard.layers
        with patch.object(UCMBlendConnector,'bind_connector_metadata'):
            c.bind_connector_metadata(meta)
            guard.apply('layer',writes.append)
            for start,count in [(16448,16384),(32999,1)]:
                step=NS(scheduled_new_reqs=[],num_scheduled_tokens={'r':count},
                    scheduled_cached_reqs=NS(req_ids=['r'],num_computed_tokens=[start],
                        resumed_from_preemption=[False],new_block_ids=[[[]]]))
                continued=c.build_connector_meta(step)
                item=continued.request_meta['r']
                self.assertEqual(item.full_block_ids,tuple(blocks))
                self.assertEqual(item.load_block_ids,([],[]))
                self.assertEqual(item.chunks_meta,[])
                c.bind_connector_metadata(continued)
                guard.apply('layer',writes.append)
            self.assertEqual(writes,['layer'])
        with patch.object(c,'clear_connector_metadata'):
            c.retire_worker_request('r')
        self.assertEqual(c.prophet_aligned,set())
        self.assertIsNone(c.worker_request_id)

    def test_actual_load_path_preserves_repaired_kv_on_continuation(self):
        from ucm.integration.vllm.blend_connector import BlendRequestDispatchMeta, UCMBlendConnectorMetadata
        c=self.connector()
        values={1:0,2:0}
        def load(hashes,shards,slots):
            for slot in slots:values[slot]=10
            return 'task'
        c.store=NS(load_data=Mock(side_effect=load),wait=Mock(),drain=Mock())
        c.global_rank=0;c.is_mla=False;c.is_dsa=False;c.load_only_first_rank=False
        c.rope_store=None;c.block_data_size=1;c.metrics_config=None;c._invalid_block_ids=set()
        c.delta_rope_vllm_ids=None;c.delta_rope_positions=None
        c._generate_task=lambda slots:(slots,None)
        c.bind_connector_metadata(UCMBlendConnectorMetadata({'r':
            BlendRequestDispatchMeta(([b'a',b'b'],[1,2]),([],[]),[],(1,2),False,dict(request_id='r',phase='read'))}))
        c.start_load_kv(None)
        self.assertEqual(values,{1:10,2:10})
        values[1]=99  # repaired KV must survive the next prefill range
        c.clear_connector_metadata()
        c.bind_connector_metadata(UCMBlendConnectorMetadata({'r':
            BlendRequestDispatchMeta(([],[]),([],[]),[],(1,2),True)}))
        c.start_load_kv(None)
        self.assertEqual(values,{1:99,2:10})
        c.store.load_data.assert_called_once()
        # A repeated initial load must fail before it can overwrite repairs.
        with self.assertRaises(RuntimeError):
            c.bind_connector_metadata(UCMBlendConnectorMetadata({'r':
                BlendRequestDispatchMeta(([b'a'],[1]),([],[]),[],(1,2),False,dict(request_id='r',phase='read'))}))

    def test_preemption_rejected_and_failed_load_rejected(self):
        from ucm.integration.vllm.blend_connector import UCMBlendConnector
        c=self.connector()
        step=NS(scheduled_new_reqs=[],scheduled_cached_reqs=NS(req_ids=['r'],resumed_from_preemption=[True]))
        with self.assertRaisesRegex(RuntimeError,'preemption'):
            c.build_connector_meta(step)
        c._invalid_block_ids={1}
        with patch.object(UCMBlendConnector,'start_load_kv'), self.assertRaisesRegex(RuntimeError,'Incomplete'):
            c.start_load_kv(None)


class ProbeTilingTests(unittest.TestCase):
    def test_long_suffix_uses_default_16k_projection_and_query_limit(self):
        from ucm.sparse.prophetkv import runtime
        suffix=16384+65
        aligned=set()
        sparse=NS(method='selective_prophetkv',scoring_layers=(0,),ratio=.2,
            model=NS(layers=[NS()]),selection_diagnostics=[],
            request=RequestMetadata('r',(0,64,128,128+suffix),tuple(range(128,128+suffix))),
            attn_metadata=NS(block_table=torch.arange(2)[None]),
            connector=NS(wait_for_layer_load=aligned.add,prophet_aligned=aligned))
        cache=torch.zeros(2,2,64,1,2)
        module=types.ModuleType('vllm.forward_context')
        module.get_forward_context=lambda:NS(virtual_engine=0,no_compile_layers={
            'model.layers.0.self_attn.attn':NS(kv_cache=[cache])})
        projection_sizes=[];query_sizes=[]
        def project(layer,h,r,positions):
            projection_sizes.append(len(h))
            q=positions.float()[:,None,None].expand(-1,1,2)
            return q,q,q,h
        def importance(q,k):
            query_sizes.append(len(q))
            return torch.full((len(k),),q[:,0,0].mean().item())
        with patch.dict(sys.modules,{'vllm.forward_context':module}), patch.object(runtime,'project',project), \
                patch.object(runtime,'context_importance',importance), patch('torch.cuda.synchronize'), \
                patch.object(runtime,'global_selection',lambda scores,prefix,ratio:(scores[prefix:],select(scores[prefix:],prefix,ratio))):
            runtime.probe(sparse,torch.arange(128,128+suffix),torch.zeros(suffix,2))
        self.assertEqual(projection_sizes,[16384,65])
        self.assertEqual(query_sizes,[16384,65])
        expected=torch.arange(128,128+suffix).float().mean()
        torch.testing.assert_close(sparse.selection_diagnostics[0]['scores'],expected.expand(64))

    def test_tiled_projection_query_weighting_and_causal_suffix_match(self):
        from ucm.sparse.prophetkv import runtime
        torch.manual_seed(123)
        cache=torch.randn(2,6,64,1,2)
        hidden=torch.randn(256,2)
        layers=[NS(index=i,self_attn=NS(o_proj=lambda x:(x,None)),
                   post_attention_layernorm=lambda h,r:(h,r),mlp=lambda x:x) for i in range(2)]
        contexts=NS(virtual_engine=0,no_compile_layers={f'model.layers.{i}.self_attn.attn':NS(kv_cache=[cache]) for i in range(2)})
        module=types.ModuleType('vllm.forward_context');module.get_forward_context=lambda:contexts
        observed=[]
        def project(layer,h,residual,positions):
            observed.append(len(h))
            h=h if residual is None else h+residual*.1
            q=(h+.01*positions[:,None]).reshape(-1,1,2)
            return q,q*.7,q*.9,h
        def selection(scores,prefix,ratio):
            return scores[prefix:],select(scores[prefix:],prefix,ratio)
        def run(budget):
            aligned=set()
            sparse=NS(method='selective_prophetkv',scoring_layers=(0,1),ratio=.4,
                model=NS(layers=layers),selection_diagnostics=[],
                request=RequestMetadata('r',(0,64,128,384),tuple(range(129,381,3))),
                attn_metadata=NS(block_table=torch.arange(6)[None]),
                connector=NS(wait_for_layer_load=aligned.add,prophet_aligned=aligned))
            runtime.probe(sparse,torch.arange(128,384),hidden,query_budget=budget)
            return sparse.selection_diagnostics[0]
        with patch.dict(sys.modules,{'vllm.forward_context':module}), patch.object(runtime,'project',project), \
                patch.object(runtime,'global_selection',selection), patch('torch.cuda.synchronize'):
            reference=run(16384)
            observed.clear()
            actual=run(37)
        self.assertLessEqual(max(observed),37)
        torch.testing.assert_close(actual['scores'],reference['scores'],rtol=2e-5,atol=1e-7)
        self.assertEqual(actual['selected_positions'].tolist(),reference['selected_positions'].tolist())


class ReadinessTests(unittest.TestCase):
    def test_standalone_cache_readiness_and_incomplete_rank_rejection(self):
        import hashlib
        import json
        import pickle
        import tempfile
        from runner.cache import verify_cache
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text(json.dumps(dict(num_attention_heads=2,
                num_key_value_heads=2,hidden_size=4,num_hidden_layers=2)))
            args=NS(model_path=root,tensor_parallel_size=2,cache_dir=root/'cache',hash_seed='seed')
            tokens=list(range(64))
            meta=[f'rpkv-bf16-local-block64-v3:{root}:2:torch.bfloat16:{rank}'.encode() for rank in range(2)]
            def hashed(m,value):
                return hashlib.md5(m+(value if isinstance(value,bytes) else pickle.dumps(value,protocol=pickle.HIGHEST_PROTOCOL))).digest()
            key=hashed(meta[0],(hashed(meta[0],'seed'),tuple(tokens)))
            files=[]
            for rank in range(2):
                name=(key if rank==0 else hashed(meta[rank],key)).hex()
                path=args.cache_dir/'kv'/name[:8]/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(bytes(64*1*2*2*2*2))
                files.append(path)
            result=verify_cache(args,[tokens])
            self.assertEqual(result['verified_shards'],2)
            files[1].write_bytes(b'incomplete')
            with self.assertRaisesRegex(RuntimeError,'Cache incomplete'):
                verify_cache(args,[tokens])


class RunnerControlTests(unittest.TestCase):
    def execute(self, start, count, length, selected=None, finish_logits=False):
        # Execute the actual patched runner function with CPU buffers and fake
        # device services. This exercises its return path, not a copied model.
        import contextlib
        source=Path('ucm/integration/vllm/patch/patch_funcs/v092/vllm_patch.py').read_text()
        function=next(n for n in ast.walk(ast.parse(source))
                      if isinstance(n,ast.FunctionDef) and n.name=='execute_model')
        model=Mock(return_value=torch.zeros(count if selected is None else len(selected),2))
        class ReachedLogits(Exception): pass
        model.compute_logits.side_effect=ReachedLogits
        group=NS(clear_connector_metadata=Mock())
        mask=None if selected is None else torch.isin(torch.arange(start,start+count),torch.tensor(selected))
        sparse=NS(prepare_step=Mock(return_value=mask))
        namespace=dict(torch=torch,has_ucm_sparse=lambda:selected is not None,
            get_ucm_sparse=lambda:sparse,has_kv_transfer_group=lambda:selected is not None,
            get_kv_transfer_group=lambda:group,ModelRunnerOutput=NS,
            get_pp_group=lambda:NS(is_first_rank=True,is_last_rank=True,ranks=[0]),
            set_forward_context=lambda *a,**kw:contextlib.nullcontext())
        exec('from __future__ import annotations\n'+ast.unparse(function),namespace)
        runner=NS(_update_states=Mock(),_prepare_inputs=lambda _: (None,False,torch.tensor([count-1]),None,[count]),
            use_cuda_graph=False,parallel_config=NS(tensor_parallel_size=1,distributed_executor_backend='mp'),
            compilation_config=NS(pass_config=NS(enable_sequence_parallelism=False)),
            get_dp_padding=lambda n:(0,None),is_multimodal_model=False,uses_mrope=False,
            input_ids=torch.ones(count,dtype=torch.long),positions=torch.arange(start,start+count),
            full_cuda_graph=False,vllm_config=NS(parallel_config=NS(tensor_parallel_size=1)),maybe_setup_kv_connector=Mock(),
            maybe_execute_ucm_sparse_begin=Mock(),model=model,maybe_wait_for_kv_save=Mock(),
            maybe_execute_ucm_sparse_finished=lambda indices:torch.tensor([max(0,(count if selected is None else len(selected))-1)]),
            get_finished_kv_transfers=lambda _:(None,None),eplb_step=Mock(),use_aux_hidden_state_outputs=False,
            input_batch=NS(req_ids=['r'],req_id_to_index={'r':0},pooling_params={}),
            requests={'r':NS(num_computed_tokens=start,prompt_token_ids=[1]*length)})
        output=NS(total_num_scheduled_tokens=count,num_scheduled_tokens={'r':count})
        if finish_logits:
            with self.assertRaises(ReachedLogits):namespace['execute_model'](runner,output)
        else:
            result=namespace['execute_model'](runner,output)
            self.assertEqual(result.sampled_token_ids,[[]])
            model.compute_logits.assert_not_called()
            runner.eplb_step.assert_called_once()
        return runner,group

    def test_empty_step_no_forward_and_no_sampling(self):
        runner,group=self.execute(64,16384,40000,selected=[])
        runner.model.assert_not_called()
        runner.maybe_wait_for_kv_save.assert_called_once()
        group.clear_connector_metadata.assert_called_once()

    def test_nonempty_partial_steps_never_sample_in_all_modes(self):
        for selected in (None,[70,80]):
            runner,_=self.execute(64,16384,40000,selected)
            runner.model.assert_called_once()
            if selected is not None:
                self.assertEqual(runner.model.call_args.kwargs['positions'].tolist(),selected)
        self.execute(16384,1,20000,None)

    def test_one_token_final_prefill_does_reach_logits(self):
        for selected in (None,[16448]):
            runner,_=self.execute(16448,1,16449,selected,finish_logits=True)
            self.assertEqual(runner.model.call_args.kwargs['positions'].tolist(),[16448])


class AllocationCompatibilityTests(unittest.TestCase):
    def test_stock_and_ucm_patched_vllm_keep_dense_slot_positions(self):
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from ucm.integration.vllm.patch.patch_funcs.v092.vllm_patch import _patch_kv_cache_manager
        for prepatched in (False, True):
            if prepatched:
                def original(*args, num_slots_sparsed=None):
                    self.assertEqual(num_slots_sparsed, -1)
                    return args[2]
            else:
                def original(*args):
                    return args[2]
            with patch.object(KVCacheManager, 'allocate_slots', original):
                _patch_kv_cache_manager()
                self.assertEqual(KVCacheManager.allocate_slots(object(), object(), 64256), 64256)


class DiagnosticTests(unittest.TestCase):
    def test_multistep_coverage_rank_agreement_and_empty_layers(self):
        from runner.generation import verify_diagnostics
        prefix,end,length=64,33088,40000
        event=dict(kind='prophetkv_selection',scores=[1.]*(end-prefix),selected_positions=[],
                   scoring_layers=[63],fusion='mean_layers_fp32',alignment_count=64)
        events=[event]
        for start in range(prefix,length,16384):
            stop=min(start+16384,length)
            positions=list(range(min(stop,max(start,end)),stop))
            events.append(dict(kind='prefill_step',start=start,end=stop,scheduled_tokens=stop-start,
                recomputed_tokens=len(positions),no_forward=not positions,prefill_complete=stop==length))
            events.extend(dict(kind='layer_counts',layer=f'model.layers.{i}.self_attn.attn',
                start=start,end=stop,selected_positions=positions,selected_set_verified=True,
                projection_tokens=len(positions),attention_tokens=len(positions),ffn_tokens=len(positions)) for i in range(64))
        workers=[dict(rank=i,diagnostics=copy.deepcopy(events)) for i in range(2)]
        sample=dict(boundaries=[0,prefix,end,length])
        verify_diagnostics(workers,sample,'selective_prophetkv',0.,2,[63])
        for kind in ('duplicate','missing_empty','duplicate_selection','rank_mask'):
            broken=copy.deepcopy(workers)
            if kind=='duplicate':broken[1]['diagnostics'][-1]['selected_positions'].append(length-1)
            if kind=='missing_empty':broken[0]['diagnostics'].pop(2)
            if kind=='duplicate_selection':broken[1]['diagnostics'].append(copy.deepcopy(event))
            if kind=='rank_mask':broken[1]['diagnostics'][0]['selected_positions']=[64]
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):
                verify_diagnostics(broken,sample,'selective_prophetkv',0.,2,[63])

    def test_baseline_reports_dense_16k_steps(self):
        from runner.generation import verify_diagnostics
        steps=[dict(kind='prefill_step',start=a,end=min(a+16384,32769),
               scheduled_tokens=min(16384,32769-a),recomputed_tokens=min(16384,32769-a),
               no_forward=False,prefill_complete=a==32768) for a in (0,16384,32768)]
        verify_diagnostics([dict(rank=0,diagnostics=steps)],dict(boundaries=[0,64,128,32769]),'baseline',0.,1,[])


if __name__ == '__main__':
    unittest.main()
