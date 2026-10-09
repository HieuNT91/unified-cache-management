"""Two-stage ProphetKV with configurable attention scoring layers."""
import torch
from ucm.sparse.blend.blend import Blend
from .selection import RequestState
from .prefill import PrefillProgress, prefill_step
from ucm.sparse.blend.blend import ReqMeta
from .runtime import probe
from .layers import resolve_layers

class ProphetKV(Blend):
    def __init__(self,config,role):
        cfg=config.kv_transfer_config.kv_connector_extra_config['ucm_sparse_config']
        cfg.setdefault('Blend',cfg['ProphetKV'])  # connector compatibility with UCM 0.3
        super().__init__(config,role)
        self.method=self.blend_config.get('method','prophetkv')
        self.ratio=self.blend_config.get('ratio',.2)
        self.scoring_layers=resolve_layers(self.method, self.blend_config.get('scoring_layers'), self.blend_config.get('num_scoring_layers'))
        if not 0 <= self.ratio <= 1: raise ValueError('ratio must be in [0, 1]')
        self.audit=self.blend_config.get('component_audit',False)
        self.request_state=RequestState()
        self.active=False
        self.pending_audit=None
        self.layer_index=0
        self.prefill_state=None
        if self.method in ('prophetkv','selective_prophetkv'):self.compute_meta={}

    def build_sparse_meta(self, scheduler_output, requests, input_batch, attn_metadata):
        self.active=False;self.pending_audit=None;self.layer_index=0
        if isinstance(attn_metadata, dict):
            attn_metadata=next(iter(attn_metadata.values()))
        self.attn_metadata=attn_metadata
        self.blend_req_metas.reset_blend_meta(self.forward_mask, attn_metadata, scheduler_output)
        if len(scheduler_output.num_scheduled_tokens)!=1:
            raise RuntimeError('One request per worker is supported')
        rid,count=next(iter(scheduler_output.num_scheduled_tokens.items()))
        worker_request=requests[rid]
        from runner.router_guard import is_dense_request
        if is_dense_request(rid):
            if (self.request_state.pending is not None or self.prefill_state is not None
                    or getattr(self.connector, 'router_dense_id', None) != rid):
                raise RuntimeError('Unarmed or stale dense bypass')
            return self.blend_req_metas
        start=worker_request.num_computed_tokens
        is_prefill,_=prefill_step(start,count,len(worker_request.prompt_token_ids))
        if not is_prefill:
            return self.blend_req_metas
        dispatch=scheduler_output.kv_connector_metadata.request_meta.get(rid)
        if self.request_state.pending is not None:
            self.request=self.request_state.take(rid)
            b=self.request.boundaries
            layout = getattr(dispatch, 'request_layout', None)
            from runner.layout import validate_request_metadata
            validate_request_metadata(layout, rid, list(worker_request.prompt_token_ids))
            if layout['phase'] != 'read' or tuple(layout['boundaries']) != b or tuple(layout['question_positions']) != self.request.question_positions:
                raise RuntimeError('Scheduler/worker request layouts disagree')
            if len(worker_request.prompt_token_ids)!=b[-1] or start!=b[1]:
                raise RuntimeError('Cached request prefix/layout mismatch')
            chunks=[] if dispatch is None else dispatch.chunks_meta
            hits=[hit for chunk in chunks for hit in chunk.store_hits]
            if (not chunks or chunks[0].start_token_dix!=b[1] or
                    len(hits)*self.block_size!=b[-2]-b[1] or not all(hits)):
                raise RuntimeError('Incomplete context cache hits')
            blocks=worker_request.block_ids[0]
            if len(blocks)*self.block_size<b[-1]:
                raise RuntimeError('Full prompt KV slots must be reserved before probing')
            if dispatch.load_block_ids[1]!=list(blocks[:b[-2]//64]):
                raise RuntimeError('Cache load does not cover the complete context exactly once')
            self.prompt_token_ids=tuple(worker_request.prompt_token_ids)
            self.prompt_blocks=tuple(blocks[:(b[-1]+63)//64])
            self.prefill_state=PrefillProgress(b[1],b[-2],b[-1])
            self.coverage_selected=None
            self.req=ReqMeta(request_id=rid,prefix_len=b[1],prefix_blk_len=b[1]//64,
                chunks_len=b[-2]-b[1],chunk_hit_mask=hits,need_blend=True)
        if self.prefill_state is None:
            # Offline construction and warmup are dense, single-step requests.
            return self.blend_req_metas
        if self.request.request_id!=rid:
            raise RuntimeError('Stale prefill request state')
        if tuple(worker_request.block_ids[0][:len(self.prompt_blocks)])!=self.prompt_blocks:
            raise RuntimeError('KV mapping changed during prefill; preemption is unsupported')
        if dispatch is None or tuple(dispatch.full_block_ids[:len(self.prompt_blocks)])!=self.prompt_blocks:
            raise RuntimeError('Missing persistent connector block mapping')
        self.active=True
        self.step_start=start;self.step_end=start+count
        self.full_slots=attn_metadata.slot_mapping.clone().long()
        self.prefix_blocks=attn_metadata.block_table[0][:self.req.prefix_blk_len].clone().long()
        self.blend_req_metas.requests=[self.req]
        return self.blend_req_metas

    def prepare_step(self, positions):
        """Probe before any repair, then compact this original-position range."""
        if not self.active:
            return None
        if self.coverage_selected is None:
            if getattr(self, 'naive_reuse', False):
                from runner.naive_reuse import select_without_scoring
                self.coverage_selected=select_without_scoring(self, positions.device)
            else:
                b=self.request.boundaries
                ids=torch.tensor(self.prompt_token_ids[b[-2]:],device=positions.device)
                embeddings=torch.cat([self.model.get_input_embeddings(ids[a:a+16384])
                                      for a in range(0,len(ids),16384)])
                suffix_positions=torch.arange(b[-2],b[-1],device=positions.device)
                self.coverage_selected=probe(self,suffix_positions,embeddings)
            self.prefill_state.select_once(self.coverage_selected.cpu().tolist())
        expected=self.prefill_state.step(self.step_start,self.step_end-self.step_start)
        self.step_positions=torch.tensor(expected,device=positions.device,dtype=positions.dtype)
        if not torch.equal(positions,torch.arange(self.step_start,self.step_end,device=positions.device)):
            raise RuntimeError('Scheduled positions changed')
        mask=torch.isin(positions,self.step_positions)
        meta=self.blend_req_metas
        meta.compute_mask.copy_(mask)
        removed=len(positions)-len(expected)
        meta.update_query_lens(0,removed)
        meta.update_need_re_index(removed!=0)
        if removed:
            self._update_attn_metadata()
            # FlashAttention's prebuilt AOT plan describes the unfiltered Q
            # count. Let the backend schedule the compact query itself.
            self.attn_metadata.scheduler_metadata=None
        self.selection_diagnostics.append(dict(kind='prefill_step',request_id=self.request.request_id,
            start=self.step_start,end=self.step_end,scheduled_tokens=len(positions),
            recomputed_tokens=len(expected),no_forward=not expected,
            prefill_complete=self.step_end==self.request.boundaries[-1]))
        if not expected:
            # Explicitly record all empty layer sets without invoking QKV/FFN.
            for i in range(64):
                self.record_layer(f'model.layers.{i}.self_attn.attn',self.step_positions)
        return mask

    def record_layer(self,name,positions):
        if not torch.equal(positions,self.step_positions):
            raise RuntimeError('Layer selected-set mismatch')
        self.selection_diagnostics.append(dict(kind='layer_counts',layer=name,
            request_id=self.request.request_id,start=self.step_start,end=self.step_end,
            selected_positions=positions.detach().clone(),selected_set_verified=True,
            projection_tokens=len(positions),attention_tokens=len(positions),ffn_tokens=len(positions)))

    def request_finished_in_worker(self,request_id):
        if self.request_state.pending is not None and self.request_state.pending.request_id==request_id:
            self.request_state.pending=None
        if getattr(self,'request',None) is not None and self.request.request_id==request_id:
            self.prefill_state=None;self.active=False;self.pending_audit=None
            self.layer_index=0
            self.blend_req_metas.requests.clear()
            for name in ('request','req','prompt_token_ids','prompt_blocks','coverage_selected',
                         'step_positions','full_slots','prefix_blocks','current_positions',
                         'fusion_positions','fusion_slots','fusion_zero','fusion_q_len',
                         'fusion_k_len','skipped_slots','attn_metadata'):
                setattr(self,name,None)
        if hasattr(self,'connector'):
            self.connector.retire_worker_request(request_id)

    def layer_begin(self,positions,hidden_states,residual):
        self.layer_index+=1
        self.current_positions=positions
        self.projection_count=len(positions)
        return positions,hidden_states,residual

    def attention_begin(self,query,key,value,layer_name,forward_context,output=None,
                        phase=None,k_hash=None,decode_ql_nope=None,decode_q_pe=None):
        if not self.active:return query,key,value,output
        from .attention import install
        install(forward_context.no_compile_layers[layer_name],self)
        self.fusion_positions=self.current_positions.contiguous()
        length=self.step_end
        pos=torch.arange(length,device=query.device)
        blocks=self.attn_metadata.block_table[0].long()
        self.fusion_slots=blocks[pos//64]*64+pos%64
        self.fusion_zero=torch.zeros(1,device=query.device,dtype=torch.int32)
        self.fusion_q_len=torch.tensor([len(query)],device=query.device,dtype=torch.int32)
        self.fusion_k_len=torch.tensor([length],device=query.device,dtype=torch.int32)
        slots=self.attn_metadata.slot_mapping.clone().long()
        self.skipped_slots=self.full_slots[~torch.isin(self.full_slots,slots)]
        event=dict(kind='layer_counts',layer=layer_name,request_id=self.request.request_id,
            projection_tokens=self.projection_count,attention_tokens=len(query),ffn_tokens=len(query),
            prefix_tokens=self.req.prefix_len,fresh_suffix_tokens=self.request.boundaries[-1]-self.request.boundaries[-2],skipped_tokens=len(self.skipped_slots))
        self.record_layer(layer_name,self.current_positions)
        event['selected_set_verified']=True
        if self.audit:
            cache=forward_context.no_compile_layers[layer_name].kv_cache[forward_context.virtual_engine]
            event['kind']='fusion_audit'
            self.pending_audit=(slots,key.clone(),value.clone(),
                cache[:,self.skipped_slots//64,self.skipped_slots%64].clone(),
                cache[:,self.prefix_blocks].clone(),event)

        return query,key,value,output

    def attention_finished(self,query,key,value,attn_output,layer_name,forward_context,phase=None):
        if self.pending_audit is None:return
        slots,k,v,skipped,prefix,event=self.pending_audit
        cache=forward_context.no_compile_layers[layer_name].kv_cache[forward_context.virtual_engine]
        written=cache[:,slots//64,slots%64]
        checks=dict(k_write_verified=torch.equal(written[0],k.reshape_as(written[0])),
            v_write_verified=torch.equal(written[1],v.reshape_as(written[1])),
            skipped_preserved=torch.equal(cache[:,self.skipped_slots//64,self.skipped_slots%64],skipped),
            prefix_preserved=torch.equal(cache[:,self.prefix_blocks],prefix),
            suffix_verified=torch.equal(self.fusion_positions[self.fusion_positions>=self.request.boundaries[-2]],torch.arange(
                min(self.step_end,max(self.step_start,self.request.boundaries[-2])),self.step_end,device=k.device)))
        if not all(checks.values()):raise RuntimeError('K/V write/preservation audit failed')
        event.update(checks)
        self.selection_diagnostics.append(event);self.pending_audit=None
