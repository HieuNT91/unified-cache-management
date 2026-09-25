"""Temporary observer installed only for ordinary, unmeasured priming."""
def begin_capture(worker):
    from ucm.sparse.prophetkv import runtime
    if hasattr(worker, '_prime_importance'):
        raise RuntimeError('Priming observer already installed')
    worker._prime_importance = runtime.context_importance
    worker._prime_layers = []
    def capture(*args, **kwargs):
        value = worker._prime_importance(*args, **kwargs)
        worker._prime_layers.append(value.clone())
        return value
    runtime.context_importance = capture
    return {'installed': True}


def end_capture(worker, prior_stem, output):
    import numpy as np
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank
    from ucm.sparse.prophetkv import runtime
    from ucm.sparse.state import get_ucm_sparse
    runtime.context_importance = worker._prime_importance
    del worker._prime_importance
    rank = get_tensor_model_parallel_rank()
    a = torch.stack(worker._prime_layers).cpu().numpy()
    del worker._prime_layers
    prior = np.load(prior_stem + f'.rank{rank}.npz')['layers'][[45,48,50,56,58]]
    assert a.shape == prior.shape and a.dtype == np.float32
    assert np.isfinite(a).all() and (a >= 0).all()
    assert np.allclose(a.sum(1, dtype=np.float64), 1, rtol=2e-4, atol=2e-4)
    close = np.allclose(a, prior, rtol=2e-5, atol=2e-7)
    path = output + f'.rank{rank}.npz'
    native = next(d for d in get_ucm_sparse().selection_diagnostics if d['kind']=='prophetkv_selection')
    np.savez_compressed(path, layers=a, native_scores=native['scores'].cpu().numpy(),
                        selected_positions=native['selected_positions'].cpu().numpy())
    if not close:
        raise RuntimeError('Selected-layer attention differs from prior capture: ' + path)
    return dict(rank=rank, path=path, layers=[45,48,50,56,58], prior_close=bool(close),
                bitwise_equal=bool(np.array_equal(a, prior)),
                max_abs_difference=float(np.max(np.abs(a-prior))), observer_removed=True)
