"""Namespace and transfer lifecycle, shared by worker and private connector."""
import re


def namespace_from_id(request_id):
    namespace, sep, purpose = request_id.partition(':')
    if not sep or not purpose or not re.fullmatch('[a-f0-9]{32}', namespace):
        raise ValueError('Every request requires an attempt UUID namespace')
    return namespace


def seed_value(namespace):
    namespace_from_id(namespace + ':seed')
    return ('UCM_HASH_SEED', 'prophetkv-persistent-v1', namespace)


class TrackedStore:
    """Retain every asynchronous operation until its backend wait succeeds."""
    def __init__(self, store):
        self.store = store
        self.pending = {}
        self.completed = 0

    def __getattr__(self, name):
        return getattr(self.store, name)

    def _submit(self, operation, *args, **kwargs):
        task = getattr(self.store, operation)(*args, **kwargs)
        self.pending[id(task)] = task
        return task

    def load_data(self, *args, **kwargs):
        return self._submit('load_data', *args, **kwargs)

    def dump_data(self, *args, **kwargs):
        return self._submit('dump_data', *args, **kwargs)

    def wait(self, task):
        result = self.store.wait(task)
        if self.pending.pop(id(task), None) is not None:
            self.completed += 1
        return result

    def drain(self):
        for task in list(self.pending.values()):
            self.wait(task)
        return dict(pending=len(self.pending), completed=self.completed)


def retire(worker):
    """RPC barrier after engine completion, before deleting owned block files."""
    import gc
    import torch
    from ucm.sparse.state import get_ucm_sparse, has_ucm_sparse
    if has_ucm_sparse():
        sparse = get_ucm_sparse()
        if sparse.request_state.pending is not None:
            raise RuntimeError('Unconsumed request metadata at retirement')
        connector = sparse.connector
        transfers = connector.store.drain()
        if connector.rope_store is not None:
            connector.rope_store.drain()
        if connector._invalid_block_ids:
            raise RuntimeError('Invalid cache blocks at retirement')
        connector.clear_connector_metadata()
        connector.prophet_aligned.clear()
        connector.delta_rope_vllm_ids = connector.delta_rope_positions = None
        connector.requests_meta.clear()
        connector.requests_blend_meta.clear()
        connector.req2rag_load_chunks.clear()
        sparse.selection_diagnostics.clear()
        sparse.active = False
        sparse.pending_audit = None
        sparse.layer_index = 0
        # Release per-request tensors but retain model, preallocated masks and hooks.
        for name in ('request','req','full_slots','prefix_blocks','current_positions',
                     'fusion_positions','fusion_slots','fusion_zero','fusion_q_len',
                     'fusion_k_len','skipped_slots','attn_metadata'):
            if hasattr(sparse, name): setattr(sparse, name, None)
        from ucm.sparse.blend.blend import BlendMetaData
        sparse.blend_req_metas = BlendMetaData()
    else:
        transfers = dict(pending=0, completed=0)
    runner = worker.model_runner
    # vLLM ordinarily removes the last finished request on the next forward.
    # The driver has confirmed engine idleness before this RPC.
    for request_id in list(runner.requests):
        runner.ucm_sparse_request_finished_in_worker(request_id)
        runner.requests.pop(request_id, None)
        runner.encoder_cache.pop(request_id, None)
        runner.input_batch.remove_request(request_id)
    bookkeeping = len(runner.requests) + len(runner.input_batch.req_id_to_index)
    if bookkeeping: raise RuntimeError('Worker request bookkeeping remains')
    gc.collect()
    torch.cuda.synchronize()
    return dict(quiescent=True, transfers=transfers,
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                request_bookkeeping=bookkeeping)


def delete_retired_files(cache_dir, owned_files):
    """Delete precisely the quiescent namespace's files; keep backend .temp."""
    from pathlib import Path
    cache_dir = Path(cache_dir)
    actual = {str(p.relative_to(cache_dir)) for p in cache_dir.rglob('*') if p.is_file()}
    if actual != set(owned_files):
        raise RuntimeError('Unexpected files at namespace retirement')
    for name in owned_files:
        relative = Path(name)
        if len(relative.parts) != 3 or relative.parts[0] != 'kv' or not re.fullmatch('[a-f0-9]{32}', relative.name) or relative.parts[1] != relative.name[:8]:
            raise ValueError('Refusing to delete non-block file')
    for name in owned_files:
        path = cache_dir / name
        path.unlink()
        if not any(path.parent.iterdir()): path.parent.rmdir()
