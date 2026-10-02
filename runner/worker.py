"""Process-local patching and Qwen3 YaRN validation."""
from ucm.integration.vllm.patch.apply_patch import apply_all_patches
apply_all_patches()

from vllm.v1.worker.gpu_worker import Worker as BaseWorker


class Worker(BaseWorker):
    def load_model(self):
        # Readers retain their shared lock even if the driver dies. Build workers
        # publish process identities so a killed builder cannot overlap a resume.
        transfer = self.vllm_config.kv_transfer_config
        config = transfer.kv_connector_extra_config.get('persistent_setup') if transfer else None
        if config:
            import fcntl
            import os
            from pathlib import Path
            from runner.setups import atomic_json
            root = Path(config['setup_root'])
            if config['readonly']:
                self.setup_read_lock = (root / '.lock').open('rb')
                fcntl.flock(self.setup_read_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            else:
                pid = os.getpid()
                starttime = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
                atomic_json(root / 'owners' / f'{pid}.json', dict(pid=pid, starttime=starttime))
        super().load_model()
        from runner.rope_window import ensure_rope_window
        model = self.model_runner.model.model
        if model.__class__.__name__ != 'Qwen3Model' or len(model.layers) != 64:
            raise RuntimeError('Expected Qwen3-32B with 64 decoder layers')
        tile = getattr(self.vllm_config.model_config.hf_config, 'ucm_activation_tile', None)
        if tile is None and self.vllm_config.cache_config.num_gpu_blocks_override == 1031:
            tile = 4096  # preserve the frozen router profile
        if tile is not None:
            if tile != 4096:
                raise ValueError('Router activation tile must be 4096')
            from runner.router_tiles import install_tiles
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


def configure(worker, method, ratio, layers=None):
    from runner.temporary import configure_sparse
    from ucm.sparse.state import get_ucm_sparse
    from vllm.distributed import get_tensor_model_parallel_rank
    if worker.model_runner.requests or worker.model_runner.input_batch.req_id_to_index:
        raise RuntimeError('Worker request bookkeeping remains before policy change')
    return dict(rank=get_tensor_model_parallel_rank(),
                **configure_sparse(get_ucm_sparse(), method, ratio, layers))


def router_mode(worker, request_id, action, capture=False):
    from runner.router_policy import RATIOS
    from runner.router_guard import is_dense_request
    from ucm.sparse.state import get_ucm_sparse
    from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
    if action not in RATIOS or is_dense_request(request_id) != (action == 'baseline'):
        raise ValueError('Invalid router request action/tag')
    if capture and action != 'prophetkv-all64-1':
        raise ValueError('Router probe must use all64/1%')
    configure(worker, 'prophetkv', RATIOS[action] or .01)
    sparse = get_ucm_sparse()
    sparse.router_capture = capture
    sparse.router_arrays = None
    sparse.router_native_layers = set()
    sparse.connector.router_dense_id = request_id if action == 'baseline' else None
    sparse.connector.store.router_dense = action == 'baseline'
    sparse.router_store_before = dict(sparse.connector.store.operations)
    # Install the native-forward observer on every layer, including dense-first requests.
    from ucm.sparse.prophetkv.attention import install
    for layer in sparse.model.layers:
        install(layer.self_attn.attn, sparse)
    get_tp_group().barrier()
    return dict(rank=get_tensor_model_parallel_rank(),request_id=request_id,action=action,capture=capture)


def router_export(worker, output):
    import numpy as np
    from pathlib import Path
    from ucm.sparse.state import get_ucm_sparse
    from vllm.distributed import get_tensor_model_parallel_rank
    from runner.setups import file_hash
    sparse = get_ucm_sparse()
    if sparse.router_arrays is None:
        raise RuntimeError('Missing native attention capture')
    layers,scores,local_mean = sparse.router_arrays
    rank = get_tensor_model_parallel_rank()
    path = Path(f'{output}.rank{rank}.npz')
    np.savez(path, layers=layers.cpu().numpy(), scores=scores.cpu().numpy(), local_mean=local_mean.cpu().numpy())
    sparse.router_arrays = None
    sparse.router_capture = False
    return dict(rank=rank,path=str(path),sha256=file_hash(path))


def router_dense_receipt(worker):
    from ucm.sparse.state import get_ucm_sparse
    from vllm.distributed import get_tensor_model_parallel_rank
    sparse = get_ucm_sparse()
    if not sparse.connector.router_dense_id or sparse.connector.store.pending:
        raise RuntimeError('Dense fallback was not armed or has pending transfers')
    layers = sorted(sparse.router_native_layers)
    expected = sorted(f'model.layers.{i}.self_attn.attn' for i in range(64))
    if layers != expected:
        raise RuntimeError('Dense fallback did not use all 64 native attention layers')
    operations={k:v-sparse.router_store_before[k] for k,v in sparse.connector.store.operations.items()}
    if any(operations.values()):
        raise RuntimeError('Dense fallback accessed the UCM store')
    return dict(rank=get_tensor_model_parallel_rank(),request_id=sparse.connector.router_dense_id,
                native_layers=layers,store_operations=operations)
