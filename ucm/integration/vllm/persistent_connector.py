"""Private connector: request-scoped hashes and explicit transfer retirement."""
import json
import os
from copy import deepcopy
from pathlib import Path
from ucm.integration.vllm.blend_connector import (UCMBlendConnector,
    UCMBlendConnectorMetadata, BlendRequestDispatchMeta, ChunkMetaData, BlendStage)
from ucm.sparse.prophetkv.lifecycle import namespace_from_id, seed_value, TrackedStore


class PersistentBlendConnector(UCMBlendConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A prepared prompt can contain a final context chunk below 1024 tokens.
        # Full-hit validation is performed by ProphetKV, so any cached block
        # is sufficient to enter its selection path.
        self.min_blend_threshold = 1
        self.persistent_setup = self.launch_config.get('persistent_setup')
        self.temporary_layouts = self.launch_config.get("temporary_layouts")
        self.temporary_readonly = False
        self.setup_boundaries = None
        self.prompt_lengths = {}
        self.dispatches = {}
        self.worker_request_id = None

    def setup_model(self, model):
        if not getattr(self, 'prophet_delta_normalized', False):
            super().setup_model(model)

    def _create_store(self, *args, **kwargs):
        store = super()._create_store(*args, **kwargs)
        config = self.launch_config.get('persistent_setup')
        if config:
            from runner.setup_store import SetupStore
            return SetupStore(store, config, self.global_rank)
        if self.launch_config.get("temporary_layouts"):
            from runner.temporary import TemporaryStore
            return TemporaryStore(store)
        return TrackedStore(store)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        namespace = namespace_from_id(request.request_id)
        if request.request_id in self.requests_blend_meta:
            # Allocation may have failed after lookup on the previous tick.
            meta = self.requests_blend_meta[request.request_id]
            return min(meta.pc_hit_block_num * self.block_size, request.num_tokens - 1), False
        if self.requests_blend_meta:
            raise RuntimeError('Previous scheduler request not retired')
        config = getattr(self, 'persistent_setup', None)
        if config:
            from runner.identity import setup_seed
            self._seed = self.request_hasher(setup_seed(config['fingerprint']))
            if config['readonly']:
                tag = request.request_id.split(':')[1]
                self.setup_boundaries = config['layouts'].get(tag)
                if self.setup_boundaries is None or self.setup_boundaries[-1] != request.num_prompt_tokens:
                    raise RuntimeError('Population requests are forbidden in cached experiments; rerun setup')
        else:
            self._seed = self.request_hasher(seed_value(namespace))
        if getattr(self, "temporary_layouts", None):
            from runner.temporary import request_phase
            phase, bounds = request_phase(request.request_id, self.temporary_layouts)
            self.temporary_readonly = phase == "read"
            self.setup_boundaries = bounds if self.temporary_readonly else None
            if self.temporary_readonly and bounds[-1] != request.num_prompt_tokens:
                raise RuntimeError("Temporary prompt length changed")
        self.prompt_lengths[request.request_id] = request.num_prompt_tokens
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if (config and config['readonly']) or getattr(self, 'temporary_readonly', False):
            meta = self.requests_blend_meta[request.request_id]
            if (not meta.blend_stage.is_blend_cache()
                    or result[0] != self.setup_boundaries[1]
                    or not all(hit for chunk in meta.chunks_meta for hit in chunk.store_hits)):
                if getattr(self, 'temporary_readonly', False):
                    raise RuntimeError('Temporary context KV is incomplete; fail this prompt sweep')
                raise RuntimeError('Persistent context KV is incomplete; rerun setup')
        return result

    def _process_req(self, tokens):
        if not ((getattr(self, 'persistent_setup', None) and self.persistent_setup['readonly'])
                or getattr(self, 'temporary_readonly', False)):
            return super()._process_req(tokens)
        # Frozen boundaries avoid delimiter heuristics and accidental prefix matches
        # across independently constructed (possibly duplicate) chunks.
        chunks, keys = [], []
        for start, end in zip(self.setup_boundaries[:-2], self.setup_boundaries[1:-1]):
            hashes = self.generate_hash(self.block_size, tokens[start:end], self._seed)
            chunks.append(ChunkMetaData(start_token_dix=start, chunk_tokens_len=end-start,
                start_blk_idx=start//64, chunk_blks_len=(end-start)//64,
                chunk_blks_hash=hashes, cached_start_position=0))
            keys.extend(hashes)
        prefix = keys[:self.setup_boundaries[1]//64]
        return BlendStage.CACHE_BLEND, prefix, chunks, keys

    def reservation_tokens(self, request, scheduled_and_external):
        meta = self.requests_blend_meta.get(request.request_id)
        if meta is not None and meta.blend_stage.is_blend_cache():
            return request.num_prompt_tokens
        return scheduled_and_external

    def build_connector_meta(self, scheduler_output):
        dispatch = {}
        for request in scheduler_output.scheduled_new_reqs:
            rid = request.req_id
            meta = self.requests_blend_meta.get(rid)
            if meta is None:
                raise RuntimeError('Missing external-cache lookup')
            item = self._generate_blend_dispatch_meta(meta,
                scheduler_output.num_scheduled_tokens[rid], request.block_ids[0])
            if ((getattr(self, 'persistent_setup', None) and self.persistent_setup['readonly'])
                    or getattr(self, 'temporary_readonly', False)) and item.dump_block_ids[0]:
                raise RuntimeError('Read-only setup forbids cache reconstruction; rerun setup')
            item.full_block_ids = tuple(request.block_ids[0])
            if len(item.load_block_ids[0]) != len(item.load_block_ids[1]):
                raise RuntimeError('Full prompt KV reservation is missing')
            self.dispatches[rid] = deepcopy(item)
            dispatch[rid] = item
        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            if cached.resumed_from_preemption[i]:
                raise RuntimeError('ProphetKV preemption is unsupported')
            # A final one-token prefill is not decode. Keep layout metadata,
            # but never load stale KV or rotate keys again on a continuation.
            if cached.num_computed_tokens[i] < self.prompt_lengths[rid]:
                original = self.dispatches[rid]
                dispatch[rid] = BlendRequestDispatchMeta(([], []), ([], []), [],
                    full_block_ids=original.full_block_ids, continuing=True)
        return UCMBlendConnectorMetadata(dispatch)

    def bind_connector_metadata(self, metadata):
        first = [rid for rid, item in metadata.request_meta.items()
                 if not getattr(item, 'continuing', False)]
        if first:
            if len(first) != 1:
                raise RuntimeError('One request at a time is supported')
            if self.worker_request_id is not None:
                raise RuntimeError('Cache load attempted before retiring the previous request')
            if getattr(self, "temporary_layouts", None):
                from runner.temporary import request_phase
                phase, _ = request_phase(first[0], self.temporary_layouts)
                self.store.writable = phase == "populate"
            self.worker_request_id = first[0]
            if hasattr(self, 'prophet_aligned'):
                self.prophet_aligned.clear()
        elif any(rid != self.worker_request_id for rid in metadata.request_meta):
            raise RuntimeError('Continuation has no loaded request')
        super().bind_connector_metadata(metadata)

    def start_load_kv(self, *args, **kwargs):
        super().start_load_kv(*args, **kwargs)
        if self._invalid_block_ids:
            raise RuntimeError('Incomplete external cache load')

    def retire_worker_request(self, request_id):
        if self.worker_request_id == request_id:
            self.store.drain()
            self.clear_connector_metadata()
            self.prophet_aligned.clear()
            self.worker_request_id = None
            if getattr(self, "temporary_layouts", None):
                self.store.writable = False

    def request_finished(self, request, block_ids):
        result = super().request_finished(request, block_ids)
        if result[0]:
            raise RuntimeError('Unexpected deferred request completion')
        self.requests_blend_meta.pop(request.request_id, None)
        self.requests_meta.pop(request.request_id, None)
        self.req2rag_load_chunks.pop(request.request_id, None)
        self.prompt_lengths.pop(request.request_id, None)
        self.dispatches.pop(request.request_id, None)
        path = Path(os.environ['PROPHETKV_SCHEDULER_RECEIPT'])
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(request_id=request.request_id,
            requests_blend_meta=len(self.requests_blend_meta), requests_meta=len(self.requests_meta))))
        tmp.replace(path)
        return result
