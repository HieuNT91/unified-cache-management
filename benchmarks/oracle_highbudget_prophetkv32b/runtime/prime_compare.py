"""Temporary capture during ordinary priming; never installed for timed inference."""
def begin_capture(worker):
    from ucm.sparse.prophetkv import runtime
    if hasattr(worker,'_prime_importance'):raise RuntimeError('Observer already installed')
    worker._prime_importance=runtime.context_importance;worker._prime_layers=[]
    def capture(*args,**kwargs):
        value=worker._prime_importance(*args,**kwargs);worker._prime_layers.append(value.clone());return value
    runtime.context_importance=capture
    return {'installed':True}

def end_capture(worker,prior_stem,output,selection_mode,layer_scope):
    import numpy as np
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank
    from ucm.sparse.prophetkv import runtime
    from ucm.sparse.state import get_ucm_sparse
    from query_config import LAYERS
    runtime.context_importance=worker._prime_importance;del worker._prime_importance
    rank=get_tensor_model_parallel_rank();native=next(d for d in get_ucm_sparse().selection_diagnostics if d['kind']=='prophetkv_selection')
    layers=[] if selection_mode=='target_only' else list(range(64)) if layer_scope=='all64' else list(LAYERS)
    a=torch.stack(worker._prime_layers).cpu().numpy() if layers else np.empty((0,native['prefix_tokens']+native['eligible_count']),dtype=np.float32)
    del worker._prime_layers
    assert len(a)==len(layers) and np.isfinite(a).all() and (a>=0).all()
    close=None;difference=None
    if layers:
        assert np.allclose(a.sum(1,dtype=np.float64),1,rtol=2e-4,atol=2e-4)
        if prior_stem:
            with np.load(prior_stem+f'.rank{rank}.npz') as z:prior=z['layers'][layers]
            close=bool(np.allclose(a,prior,rtol=2e-5,atol=2e-7));difference=float(np.max(np.abs(a-prior)))
            assert close,'Full-question priming changed from historical capture'
    path=output+f'.rank{rank}.npz'
    np.savez_compressed(path,layers=a,native_scores=native['scores'].cpu().numpy(),
        selected_positions=native['selected_positions'].cpu().numpy(),base_selected_positions=native['base_selected_positions'].cpu().numpy(),
        query_positions=native['question_positions'],scoring_layers=layers)
    return dict(rank=rank,path=path,layers=layers,selection_mode=selection_mode,layer_scope=layer_scope,
        prior_close=close,historical_match_required=bool(prior_stem and layers),max_abs_difference=difference,observer_removed=True)
