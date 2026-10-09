"""Explicit zero-repair control, separate from learned router inventories."""
ACTION = dict(id='naive-reuse', method='naive_reuse', ratio=0.0)


def inventory(protocol):
    if protocol.get('naive_reuse') is True:
        if (protocol.get('actions') != [ACTION] or protocol.get('scheduled_actions') != [ACTION['id']]
                or protocol.get('answer_validation') != 'native-answer-diagnostics-v1'):
            raise ValueError('Invalid naive reuse control protocol')
        return {ACTION['id']: dict(ACTION)}
    from runner.tree_policy import actions
    return actions(protocol['actions'])


def select_without_scoring(sparse, device):
    """Load/align all layers once; never project suffix queries or rank context."""
    import torch
    if sparse.ratio != 0 or sparse.router_capture or getattr(sparse, 'router_head_layers', ()):
        raise ValueError('Naive reuse requires ratio zero without capture')
    for i in range(len(sparse.model.layers)):
        sparse.connector.wait_for_layer_load(f'model.layers.{i}.self_attn.attn')
    if set(sparse.connector.prophet_aligned) != {f'model.layers.{i}.self_attn.attn' for i in range(64)}:
        raise RuntimeError('Naive reuse requires all-layer cache alignment')
    selected = torch.empty(0, dtype=torch.long, device=device)
    sparse.selection_diagnostics.append(dict(kind='naive_reuse_selection',
        request_id=sparse.request.request_id, selected_positions=selected,
        ratio=0.0, scoring_layers=[], probe_layers=0, suffix_forward_layers=0,
        alignment_count=64))
    return selected
