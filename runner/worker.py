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
        self.rope_audits = [ensure_rope_window(layer.self_attn.rotary_emb, 131072)
                            for layer in model.layers]


def setup(worker):
    from ucm.sparse.prophetkv.runtime import setup as install
    return dict(sparse=install(worker), rope=worker.rope_audits)


def arm(worker, **kwargs):
    from ucm.sparse.prophetkv.runtime import set_request
    return set_request(worker, **kwargs)


def drain(worker):
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank
    from ucm.sparse.state import has_ucm_sparse, get_ucm_sparse
    entries = get_ucm_sparse().selection_diagnostics if has_ucm_sparse() else getattr(worker.model_runner, 'prefill_diagnostics', [])
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
