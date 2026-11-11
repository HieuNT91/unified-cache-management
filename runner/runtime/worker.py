"""Process-local patching and Qwen3 YaRN validation."""
from ucm.integration.vllm.patch.apply_patch import apply_all_patches
apply_all_patches()

from vllm.v1.worker.gpu_worker import Worker as BaseWorker


class Worker(BaseWorker):
    def load_model(self):
        super().load_model()
        from runner.runtime.rope_window import ensure_rope_window
        model = self.model_runner.model.model
        if model.__class__.__name__ != 'Qwen3Model' or len(model.layers) != 64:
            raise RuntimeError('Expected Qwen3-32B with 64 decoder layers')
        tile = getattr(self.vllm_config.model_config.hf_config, 'ucm_activation_tile', None)
        if tile is not None:
            if tile != 4096:
                raise ValueError('Router activation tile must be 4096')
            from runner.runtime.tiles import install_tiles
            install_tiles(model, tile)
        self.rope_audits = [ensure_rope_window(layer.self_attn.rotary_emb, 131072)
                            for layer in model.layers]


def setup(worker):
    from vllm.distributed import get_tensor_model_parallel_rank
    from ucm.sparse.prophetkv.runtime import setup as install
    return dict(rank=get_tensor_model_parallel_rank(), sparse=install(worker), rope=worker.rope_audits,
                kv_tokens=worker.vllm_config.model_config.max_model_len,
                kv_blocks=worker.vllm_config.cache_config.num_gpu_blocks,
                block_size=worker.vllm_config.cache_config.block_size)


def arm(worker, **kwargs):
    from ucm.sparse.prophetkv.runtime import set_request
    return set_request(worker, **kwargs)


def drain(worker):
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank
    from ucm.sparse.state import has_ucm_sparse, get_ucm_sparse
    entries = get_ucm_sparse().selection_diagnostics if has_ucm_sparse() else getattr(worker.model_runner, 'prefill_diagnostics', [])
    entries = list(entries)
    if has_ucm_sparse():
        entries.extend(getattr(worker.model_runner, 'prefill_diagnostics', []))
        get_ucm_sparse().selection_diagnostics.clear()
    result = [{k: v.detach().cpu().tolist() if torch.is_tensor(v) else v
               for k, v in entry.items()} for entry in entries]
    entries.clear()
    if hasattr(worker.model_runner, "prefill_diagnostics"):
        worker.model_runner.prefill_diagnostics.clear()
    return dict(rank=get_tensor_model_parallel_rank(), diagnostics=result)


def retire(worker):
    from ucm.sparse.prophetkv.lifecycle import retire as finish
    return finish(worker)


def configure(worker, adatree_chunk_size):
    from runner.cache.temporary import configure_sparse
    from ucm.sparse.state import get_ucm_sparse
    from vllm.distributed import get_tensor_model_parallel_rank
    if worker.model_runner.requests or worker.model_runner.input_batch.req_id_to_index:
        raise RuntimeError('Worker request bookkeeping remains before policy change')
    return dict(rank=get_tensor_model_parallel_rank(),
                **configure_sparse(get_ucm_sparse(), 'adatree-v1', None, adatree_chunk_size=adatree_chunk_size))
