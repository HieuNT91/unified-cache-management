"""Original tokens, explicit transport, duplicate slots, and CPU causal repair."""
import copy
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from runner.layout import (PROMPT_PROTOCOL, METADATA_KEY, boundaries_for, stamp_sample,
                           request_metadata, validate_request_metadata, validate_layout)
from runner.config import validate_sample, engine_config
from runner.identity import BlockHasher


def sample(length=449):
    # Real EOT-like values occur inside blocks AND at their ends; three identical
    # chunks must share cache hashes but receive three independent global slots.
    chunk = [99]+[8]*62+[99]
    return stamp_sample(dict(token_ids=chunk*3+[7]*(length-192),boundaries=[0,64,128,192,length],
        question_positions=[length-1],thinking=False,max_output_tokens=32))


def connector():
    from ucm.integration.vllm.persistent_connector import PersistentBlendConnector
    c=PersistentBlendConnector.__new__(PersistentBlendConnector)
    c.block_size=64;c.min_blend_threshold=1;c.metrics_config=None;c.persistent_setup=None
    c.requests_blend_meta={};c.prompt_lengths={};c.request_hasher=BlockHasher('fixture',4,0)
    c.store=Mock();c.store.lookup.side_effect=lambda keys:[True]*len(keys)
    return c


def request(s,phase='read',rid=None):
    rid=rid or 'a'*32+':arbitrary-name-with-no-phase'
    tokens=s['token_ids']
    metadata=request_metadata(rid,tokens,phase,s if phase=='read' else None)
    return NS(request_id=rid,prompt_token_ids=tokens,all_token_ids=tokens,
        num_prompt_tokens=len(tokens),num_tokens=len(tokens),sampling_params=NS(extra_args={METADATA_KEY:metadata}))


class MetadataTests(unittest.TestCase):

    def test_real_store_pointer_flattening_repeated_files_load_separate_slots(self):
        import numpy as np
        from ucm.store.pcstore.pcstore_connector_v1 import UcmPcStoreV1
        from ucm.sparse.prophetkv.lifecycle import TrackedStore
        # Exercise the real Python-to-native pointer flattening. Native files
        # contain exactly two tensors; repeated IDs in one call must not turn
        # a second destination into a read past the end of that same file.
        payload={b'a':(11,12),b'b':(21,22)};memory={};calls=[]
        def load(ids,pointers):
            grouped={}
            for key,pointer in zip(ids,pointers):grouped.setdefault(key,[]).append(pointer)
            for key,pointers in grouped.items():
                if len(pointers)!=len(payload[key]):raise RuntimeError('oversized file read')
                for pointer,value in zip(pointers,payload[key]):memory[pointer]=value
            calls.append((list(ids),list(pointers)))
            return len(calls)
        backend=UcmPcStoreV1.__new__(UcmPcStoreV1)
        backend.store=NS(LoadToDevice=load,Wait=lambda _:0)
        ids=[b'a',b'b',b'a',b'a',b'b'];addresses=np.arange(10,dtype=np.uint64).reshape(5,2)
        with self.assertRaisesRegex(RuntimeError,'oversized'):
            backend.load_data(ids,[0]*5,addresses)
        wrapped=TrackedStore(backend);task=wrapped.load_data(ids,[0]*5,addresses)
        self.assertEqual(len(wrapped.pending),3)
        wrapped.wait(task)
        self.assertEqual(wrapped.pending,{})
        for i,key in enumerate(ids):self.assertEqual(tuple(memory[j] for j in addresses[i]),payload[key])
        self.assertEqual(payload,{b'a':(11,12),b'b':(21,22)})
        self.assertEqual(wrapped.drain(),dict(pending=0,completed=3))

    def test_repeated_load_failure_keeps_every_unfinished_transfer_for_retirement(self):
        from ucm.sparse.prophetkv.lifecycle import TrackedStore
        tasks=[object(),object(),object()];backend=Mock();backend.load_data.side_effect=tasks
        wrapped=TrackedStore(backend);task=wrapped.load_data([b'a']*3,[0]*3,[[1],[2],[3]])
        backend.wait.side_effect=[None,RuntimeError('missing shard')]
        with self.assertRaisesRegex(RuntimeError,'missing shard'):wrapped.wait(task)
        self.assertEqual(set(wrapped.pending),{id(tasks[1]),id(tasks[2])})
        backend.wait.side_effect=None
        self.assertEqual(wrapped.drain(),dict(pending=0,completed=3))

    def test_original_layout_partial_chunk_long_suffix_short_baseline(self):
        for length,q in [(65500,65400),(65506,62001),(600,200),(12,2)]:
            ids=list(range(length));bounds=boundaries_for(length,q)
            s=stamp_sample(dict(token_ids=ids,boundaries=bounds,question_positions=[q],thinking=False,max_output_tokens=30))
            validate_sample(s)
            self.assertEqual(bounds[-2],max(0,64*(min(q,length-256)//64)))
            self.assertEqual([t for a,b in zip(bounds,bounds[1:]) for t in ids[a:b]],ids)
            self.assertTrue(all(b-a<=4096 for a,b in zip(bounds[:-2],bounds[1:-1])))
        short=stamp_sample(dict(token_ids=[99]*12,boundaries=[0,12],question_positions=[2],thinking=False,max_output_tokens=30))
        validate_sample(short)
        self.assertNotIn('kv_transfer_config',engine_config('/model','baseline'))
        with self.assertRaisesRegex(ValueError,'at least two context chunks'):
            validate_layout(short['token_ids'],short['boundaries'],short['question_positions'],sparse=True)

    def test_eot_and_duplicate_chunks_reuse_keys_in_distinct_global_slots(self):
        s=sample();before=copy.deepcopy(s);c=connector();r=request(s)
        self.assertEqual(c.get_num_new_matched_tokens(r,0),(64,False))
        meta=c.requests_blend_meta[r.request_id]
        dispatch=c._generate_blend_dispatch_meta(meta,len(s['token_ids'])-64,list(range(8)))
        self.assertEqual(dispatch.load_block_ids[1],[0,1,2])
        self.assertEqual(len(set(dispatch.load_block_ids[0])),1)
        self.assertEqual([chunk.position_offset for chunk in dispatch.chunks_meta],[64,128])
        self.assertEqual(dispatch.dump_block_ids,([],[]))
        self.assertEqual(s,before)
        self.assertFalse(hasattr(c,'chunk_end_token_id'))

    def test_phase_is_metadata_even_if_request_name_or_last_token_disagrees(self):
        c=connector();s=sample();r=request(s,rid='a'*32+':populate-but-actually-read')
        self.assertEqual(c.get_num_new_matched_tokens(r,0),(64,False))
        c=connector();c.store.lookup.side_effect=lambda keys:[False]*len(keys)
        tokens=[99]*63+[1234]
        r=request(dict(token_ids=tokens),'populate','b'*32+':read-but-actually-populate')
        self.assertEqual(c.get_num_new_matched_tokens(r,0),(0,False))
        self.assertEqual(c.active_layout['phase'],'populate')
        dispatch=c._generate_blend_dispatch_meta(c.requests_blend_meta[r.request_id],64,[0])
        self.assertEqual(dispatch.dump_block_ids[1],[0])

    def test_missing_invalid_metadata_never_looks_up(self):
        s=sample()
        for key,value in [(None,None),('request_id','wrong'),('phase','unknown'),('token_sha256','bad'),
                          ('boundaries',[0,64,64,192,449]),('boundaries',[0,64,128,192,450]),
                          ('question_positions',[63]),('protocol','legacy')]:
            c=connector();r=request(s)
            if key is None:r.sampling_params.extra_args={}
            else:r.sampling_params.extra_args[METADATA_KEY][key]=value
            with self.assertRaises(ValueError):c.get_num_new_matched_tokens(r,0)
            c.store.lookup.assert_not_called()
        old=copy.deepcopy(s);old.pop('prompt_protocol')
        with self.assertRaisesRegex(ValueError,'rebuild'):validate_sample(old)

    def test_allocation_retry_cannot_change_valid_layout(self):
        c=connector();r=request(sample())
        c.get_num_new_matched_tokens(r,0)
        r.sampling_params.extra_args[METADATA_KEY]['boundaries']=[0,128,192,449]
        with self.assertRaisesRegex(ValueError,'allocation retry'):
            c.get_num_new_matched_tokens(r,0)

    def test_sparse_missing_shard_fails_dense_has_zero_operations(self):
        c=connector();r=request(sample());c.store.lookup.side_effect=lambda keys:[False]*len(keys)
        with self.assertRaisesRegex(RuntimeError,'incomplete'):c.get_num_new_matched_tokens(r,0)
        c=connector();r=request(sample(),rid='a'*32+':read|router-dense')
        self.assertEqual(c.get_num_new_matched_tokens(r,0),(0,False))
        self.assertEqual(c.store.mock_calls,[])

    def test_generation_transport_has_identical_tokens_and_task_decoding(self):
        from runner import generation
        s=sample();received=[]
        engine=NS(has_unfinished_requests=Mock(side_effect=[False,True,False]))
        engine.add_request=lambda rid,prompt,params:received.append((rid,prompt,params))
        engine.step=lambda:[NS(request_id='request',outputs=[NS(token_ids=[1])],finished=True)]
        generation.generate(engine,s['token_ids'],32,'request',False,sample=s)
        rid,prompt,params=received[0]
        self.assertEqual(prompt['prompt_token_ids'],s['token_ids'])
        self.assertEqual((params.temperature,params.top_p,params.top_k,params.max_tokens),(0.,1.,32,32))
        self.assertFalse(params.stop);self.assertFalse(params.ignore_eos)
        validate_request_metadata(params.extra_args[METADATA_KEY],rid,prompt['prompt_token_ids'])
        from vllm.v1.serial_utils import MsgpackEncoder,MsgpackDecoder
        from vllm import SamplingParams
        restored=MsgpackDecoder(SamplingParams).decode(MsgpackEncoder().encode(params))
        self.assertEqual(restored.extra_args,params.extra_args)

    def test_cpu_rope_duplicate_slots_and_causal_repair_zero_full(self):
        import torch
        from ucm.sparse.prophetkv.attention import dense_reference
        from ucm.sparse.prophetkv.selection import LayerAlignment
        torch.manual_seed(5)
        n=24;theta=torch.arange(n,dtype=torch.float64)*.07
        raw=torch.randn(8,2,dtype=torch.float64);stored=raw.clone()
        def rotate(x,a):
            c,s=a.cos(),a.sin()
            return torch.stack((x[:,0]*c-x[:,1]*s,x[:,0]*s+x[:,1]*c),dim=-1)
        local=rotate(raw,theta[:8]);frozen=local.clone();loaded=[]
        for offset in (0,8,16):loaded.append(rotate(local,theta[offset].expand(8)))
        aligned=torch.cat(loaded)
        torch.testing.assert_close(aligned,rotate(raw.repeat(3,1),theta))
        guard=LayerAlignment();calls=[]
        for _ in range(3):guard.apply('layer',lambda name:calls.append(name))
        self.assertEqual(calls,['layer']);torch.testing.assert_close(local,frozen)
        q=torch.randn(n,1,2);k=aligned.float()[:,None,:];v=torch.randn(n,1,2)
        for repair in (0,100):
            selected=list(range(8,16)) if repair else []
            positions=torch.tensor(selected+list(range(16,n)))
            actual=dense_reference(q[positions],k,v,positions)
            expanded=[]
            for p in positions:
                weights=(q[p,0]@k[:p+1,0].T/(2**.5)).softmax(-1)
                expanded.append((weights@v[:p+1,0])[None,:])
            torch.testing.assert_close(actual,torch.stack(expanded))
        torch.testing.assert_close(raw,stored);torch.testing.assert_close(local,frozen)

if __name__=='__main__':unittest.main()
