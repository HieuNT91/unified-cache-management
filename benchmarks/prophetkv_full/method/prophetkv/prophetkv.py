"""Registered two-stage ProphetKV method and causally corrected Blend control."""
import torch
from ucm.sparse.blend.blend import Blend
from .selection import request_mask, RequestState
from .runtime import probe

class ProphetKV(Blend):
    def __init__(self,config,role):
        cfg=config.kv_transfer_config.kv_connector_extra_config['ucm_sparse_config']
        cfg.setdefault('Blend',cfg['ProphetKV'])  # connector compatibility with UCM 0.3
        super().__init__(config,role)
        self.method=self.blend_config.get('method','prophetkv')
        self.ratio=self.blend_config.get('ratio',.2)
        self.audit=self.blend_config.get('component_audit',False)
        self.request_state=RequestState()
        self.active=False
        self.pending_audit=None
        self.layer_index=0
        if self.method=='prophetkv':self.compute_meta={}

    def build_sparse_meta(self,*args):
        result=super().build_sparse_meta(*args)
        self.active=False;self.pending_audit=None;self.layer_index=0
        candidates=[r for r in self.blend_req_metas.requests if r.need_blend]
        if not candidates:
            # Warmup, decode and a full cache miss stay on the dense path.
            if self.request_state.pending is not None:
                scheduled=args[0].scheduled_new_reqs
                if len(scheduled)!=1:raise RuntimeError('Request metadata on decode/concurrent batch')
                self.request_state.take(scheduled[0].req_id)
                self.selection_diagnostics.append(dict(kind='cache_miss',request_id=scheduled[0].req_id))
            return result
        if len(candidates)!=1 or len(self.blend_req_metas.requests)!=1:
            raise RuntimeError('One prefill request per worker is supported')
        self.req=candidates[0]
        self.request=self.request_state.take(self.req.request_id)
        b=self.request.boundaries
        if self.req.prefix_len>b[1] or self.req.prefix_len+self.req.chunks_len!=b[-2]:
            raise RuntimeError('Request/layout mismatch')
        if self.req.prefix_len<b[1] or not all(self.req.chunk_hit_mask):
            # Recompute the complete scheduled region on a partial miss.
            self.req.need_blend=False
            self.selection_diagnostics.append(dict(kind='cache_miss',request_id=self.request.request_id))
            return result
        self.active=True
        self.full_slots=self.attn_metadata.slot_mapping.clone().long()
        self.prefix_blocks=self.attn_metadata.block_table[0][:self.req.prefix_blk_len].clone().long()
        return result

    def layer_begin(self,positions,hidden_states,residual):
        i=self.layer_index;self.layer_index+=1
        if self.active and self.method=='prophetkv' and i==0:
            selected=probe(self,positions,hidden_states)
            mask=request_mask(positions,selected,self.req.prefix_len,self.request.boundaries[-2],
                torch.tensor(self.req.chunk_hit_mask,device=positions.device))
            meta=self.blend_req_metas
            meta.compute_mask.copy_(mask)
            meta.update_query_lens(0,int((~mask).sum()))
            meta.update_need_re_index(True)
            self._update_attn_metadata()
            positions=positions[mask];hidden_states=hidden_states[mask]
            residual=None if residual is None else residual[mask]
        else:
            positions,hidden_states,residual=super().layer_begin(positions,hidden_states,residual)
        self.current_positions=positions
        self.projection_count=len(positions)
        return positions,hidden_states,residual

    def attention_begin(self,query,key,value,layer_name,forward_context,output=None,
                        phase=None,k_hash=None,decode_ql_nope=None,decode_q_pe=None):
        if self.method=='cacheblend':
            before=len(query)
            query,key,value,output=super().attention_begin(query,key,value,layer_name,forward_context,
                output,phase,k_hash,decode_ql_nope,decode_q_pe)
            if self.active and len(query)!=before:
                self.current_positions=self.current_positions[self.blend_req_metas.compute_mask]
        if not self.active:return query,key,value,output
        from .attention import install
        install(forward_context.no_compile_layers[layer_name],self)
        self.fusion_positions=self.current_positions.contiguous()
        length=self.request.boundaries[-1]
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
        if self.audit:
            cache=forward_context.no_compile_layers[layer_name].kv_cache[forward_context.virtual_engine]
            event['kind']='fusion_audit'
            self.pending_audit=(slots,key.clone(),value.clone(),
                cache[:,self.skipped_slots//64,self.skipped_slots%64].clone(),
                cache[:,self.prefix_blocks].clone(),event)
        else:self.selection_diagnostics.append(event)
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
            suffix_verified=torch.equal(self.fusion_positions[-(self.request.boundaries[-1]-self.request.boundaries[-2]):],torch.arange(
                self.request.boundaries[-2],self.request.boundaries[-1],device=k.device)))
        if not all(checks.values()):raise RuntimeError('K/V write/preservation audit failed')
        event.update(checks)
        self.selection_diagnostics.append(event);self.pending_audit=None
