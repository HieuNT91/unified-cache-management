"""Per-request CacheBlend selection scores (inputs contain eligible tokens only)."""

import torch


def selection_scores(his_k, golden_k, his_v=None, golden_v=None, selection_metric="k"):
    """Return scores and available L1 channels without changing legacy K arithmetic.

    Inputs have shape [eligible_tokens, kv_heads * head_dim]. K has already
    received the connector's positional alignment. V is indexed identically,
    but does not receive RoPE. Only normalization of the combined score is FP32.
    """
    if selection_metric not in ("k", "v", "kv"):
        raise ValueError(f"Unknown CacheBlend selection_metric: {selection_metric!r}")
    diff_k = diff_v = None
    if selection_metric in ("k", "kv"):
        diff_k = torch.sum((his_k - golden_k).abs(), dim=[1])
    if selection_metric in ("v", "kv"):
        diff_v = torch.sum((his_v - golden_v).abs(), dim=[1])
    if selection_metric == "k":
        return diff_k, diff_k, diff_v
    if selection_metric == "v":
        return diff_v, diff_k, diff_v
    k, v = diff_k.float(), diff_v.float()
    if k.numel() == 0:
        return k, diff_k, diff_v
    scores = 0.5 * (k / k.mean().clamp_min(1e-8) + v / v.mean().clamp_min(1e-8))
    return scores, diff_k, diff_v
