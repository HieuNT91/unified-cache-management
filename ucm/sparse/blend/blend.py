import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.cuda.nvtx as nvtx
from sympy import false
from torch import Tensor

from ucm.logger import init_logger

logger = init_logger(__name__)

from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.v1.request import Request

from ucm.integration.vllm.blend_connector import BlendRequestDispatchMeta
from ucm.sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseMetadata,
    UcmSparseRole,
)
from ucm.sparse.utils import round_up


@contextmanager
def nvtx_range(msg: str, enable: bool = True):
    if enable:
        nvtx.range_push(msg)
    try:
        yield
    finally:
        if enable:
            nvtx.range_pop()


def get_num_blks(num_tokens, block_size):
    return (num_tokens + block_size - 1) // block_size


@dataclass
class ReqMeta:
    req_idx: int = 0
    request_id: str = ""
    need_blend: bool = false

    prefix_len: int = 0
    prefix_blk_len: int = 0

    chunks_len: int = 0
    chunks_blk_len: int = 0

    suffix_len: int = 0
    suffix_blk_len: int = 0

    chunk_hit_mask: List[bool] = field(default_factory=list)

    chunk_hit_blk_len: int = 0


@dataclass
class BlendMetaData(UcmSparseMetadata):
    requests: list[ReqMeta] = field(default_factory=list)
    compute_mask: Tensor = None
    chunk_blks_hit_mask: Tensor = None
    query_lens: Tensor = None
    blend_start_req_idx: int = 0
    need_re_index: bool = False

    def reset_blend_meta(self, forward_mask, attn_metadata, scheduler_output):
        # current not support chunk prefill
        # for multi req in one batch, we should discard the decode req
        self.need_re_index = False
        self.requests = []
        self.compute_mask = forward_mask[: attn_metadata.query_start_loc[-1]]

        self.query_lens = (
            attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
        )

        self.blend_start_req_idx = len(scheduler_output.scheduled_cached_reqs.req_ids)

    def add_request(
        self,
        idx: int,
        req_dispatch_meta: BlendRequestDispatchMeta,
        seq_lens: Tensor,
        block_size: int,
    ) -> None:
        chunks_meta = req_dispatch_meta.chunks_meta
        if chunks_meta:
            hit_mask = []
            req_idx_batch = self.blend_start_req_idx + idx
            for meta in chunks_meta:
                hit_mask.extend(meta.store_hits)
            reqMeta = ReqMeta(
                req_idx=req_idx_batch,
                prefix_len=chunks_meta[0].start_token_dix,
                prefix_blk_len=get_num_blks(chunks_meta[0].start_token_dix, block_size),
                chunks_len=len(hit_mask) * block_size,
                chunks_blk_len=len(hit_mask),
                chunk_hit_mask=hit_mask,
                chunk_hit_blk_len=sum(hit_mask),
            )
            reqMeta.need_blend = reqMeta.chunk_hit_blk_len > 0
            reqMeta.suffix_len = (
                seq_lens[req_idx_batch].item() - reqMeta.prefix_len - reqMeta.chunks_len
            )
            reqMeta.suffix_blk_len = get_num_blks(reqMeta.suffix_len, block_size)

            self.requests.append(reqMeta)

    def reset_compute_mask(self) -> None:
        self.compute_mask.fill_(False)
        # for decode req in the front of the batch
        self.compute_mask[: self.blend_start_req_idx] = True

    def update_query_lens(self, req_idx: int, reused_num_tokens: int) -> None:
        self.query_lens[req_idx] -= reused_num_tokens

    def update_need_re_index(self, need_re_index: bool) -> None:
        self.need_re_index = need_re_index

    def update_req_compute_mask(
        self,
        req_query_start,
        req_chunk_end,
        req_query_end,
        chunk_hit_mask,
        top_k_indices,
    ):
        # for multi req batch, maybe we should update compute_mask in batch level rather than in req level
        chunks = self.compute_mask[req_query_start:req_chunk_end]
        chunks = chunks.reshape(len(chunk_hit_mask), -1)

        # for chunk block cache miss part, just recompute
        chunks.masked_fill_(~chunk_hit_mask.unsqueeze(1), True)

        flat = chunks.view(-1)
        # for chunk block cache hit part, just recompute HKVD(highest KV deviation) tokens
        flat[top_k_indices] = True

        # for question part, default
        self.compute_mask[req_chunk_end:req_query_end].fill_(True)


class Blend(UcmSparseBase):
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role)
        self.blend_config = vllm_config.kv_transfer_config.kv_connector_extra_config[
            "ucm_sparse_config"
        ]["Blend"]

        max_model_len = vllm_config.model_config.max_model_len
        self.block_size = vllm_config.cache_config.block_size

        self.device = vllm_config.device_config.device
        self.forward_mask = torch.zeros(max_model_len, device=self.device).bool()
        self.mask_idx = torch.arange(
            round_up(max_model_len, self.block_size), device=self.device
        )
        self.mask_idx = self.mask_idx.reshape(-1, self.block_size)

        # for multi batch, ignore the decode-stage req at the beginning
        self.blend_start_req_idx = 0

        self.compute_meta = self.blend_config["compute_meta"]
        for config in self.compute_meta.values():
            if config.get("selection_metric", "k") not in ("k", "v", "kv"):
                raise ValueError("selection_metric must be 'k', 'v', or 'kv'")
        self.selection_diagnostics = []
        self.blend_req_metas: BlendMetaData = BlendMetaData(
            need_re_index=False,
            chunk_blks_hit_mask=torch.zeros(
                round_up(max_model_len, self.block_size), device=self.device
            ).bool(),
        )
        self.attn_metadata = None

    def build_sparse_meta(
        self, scheduler_output, requests, input_batch, attn_metadata
    ) -> UcmSparseMetadata:

        if isinstance(attn_metadata, dict):
            attn_metadata = next(iter(attn_metadata.values()))
        self.attn_metadata = attn_metadata

        self.blend_req_metas.reset_blend_meta(
            self.forward_mask, attn_metadata, scheduler_output
        )

        blend_conn_request_meta = scheduler_output.kv_connector_metadata.request_meta
        for idx, request in enumerate(scheduler_output.scheduled_new_reqs):
            req_id = request.req_id
            self.blend_req_metas.add_request(
                idx,
                blend_conn_request_meta[req_id],
                attn_metadata.seq_lens,
                self.block_size,
            )
            if self.blend_req_metas.requests and blend_conn_request_meta[req_id].chunks_meta:
                self.blend_req_metas.requests[-1].request_id = req_id

        return self.blend_req_metas

    def _update_attn_metadata(self):
        # update attn_metadata, cause we sparse the prefill tokens
        # golden kv caches are available in current blend layer, so maybe we should cache all of them
        # so maybe we should modify slot_mapping at the beginning of next layer/attn
        self.attn_metadata.slot_mapping = self.attn_metadata.slot_mapping[
            self.blend_req_metas.compute_mask
        ]
        self.attn_metadata.query_start_loc[1:] = torch.cumsum(
            self.blend_req_metas.query_lens, dim=0
        )
        self.attn_metadata.max_query_len = self.blend_req_metas.query_lens.max().item()
        self.attn_metadata.num_actual_tokens = (
            self.blend_req_metas.query_lens.sum().item()
        )

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        """
        This is called by "Scheduler->schedule" function to estimate the number of required blocks.
        """
        return INVALID_SLOT

    def request_begin(self, request_id: Union[int, str], prompt_token_ids: List[int]):
        pass

    def ffn_begin(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states is equal to attn out, which is contiguous
        if self.blend_req_metas.need_re_index and len(
            self.blend_req_metas.compute_mask
        ) == len(residual):
            logger.info(
                f"[blend-ffn] after cache blend, reduce ffn tokens from {len(self.blend_req_metas.compute_mask)} "
                f"to {self.blend_req_metas.compute_mask.sum().item()}"
            )
            return hidden_states[
                : self.attn_metadata.num_actual_tokens
            ], self._index_tensor(residual)
        return hidden_states, residual

    def layer_begin(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(positions) != len(hidden_states):
            logger.info(
                f"[blend-layer] after cache blend, reduce layer tokens from {len(self.blend_req_metas.compute_mask)} "
                f"to {self.blend_req_metas.compute_mask.sum().item()}"
            )
            return self._index_tensor(positions), hidden_states, residual
        return positions, hidden_states, residual

    def execute_finished(self, logits_indices: torch.Tensor):
        if self.blend_req_metas.need_re_index:
            modified_logits_indices = self.attn_metadata.query_start_loc[1:] - 1
            logger.info(
                f"[blend-model] modify logits_indices from {logits_indices} "
                f"to {modified_logits_indices}"
            )
            return modified_logits_indices
        return logits_indices

    def _index_tensor(self, tensor: torch.Tensor):
        if self.blend_req_metas.need_re_index:
            return tensor[self.blend_req_metas.compute_mask]
        return tensor
