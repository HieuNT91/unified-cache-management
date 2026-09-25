"""Extend only YaRN's position table, preserving its factor and original entries."""
import torch


def ensure_rope_window(rope, required):
    if rope.scaling_factor != 2.0 or rope.max_position_embeddings != 32768:
        raise ValueError('Expected YaRN factor 2 and original context 32768')
    if hasattr(rope, '_yarn2_window_audit'):
        if len(rope.cos_sin_cache) < required:
            raise ValueError('Incompatible repeated RoPE window request')
        return dict(rope._yarn2_window_audit)
    original = rope.cos_sin_cache
    old_length = len(original)
    if old_length != 65536 or required != 65920:
        raise ValueError('Unexpected original or requested position table length')
    # Use the exact installed vLLM YaRN formula on the same device. Do not
    # alter max_position_embeddings, correction ranges, factor, or magnitude.
    with torch.device(original.device):
        inv_freq = rope._compute_inv_freq(rope.scaling_factor)
        positions = torch.arange(required, dtype=torch.float32)
        freqs = torch.einsum('i,j -> ij', positions, inv_freq)
        reference = torch.cat((freqs.cos() * rope.mscale,
                               freqs.sin() * rope.mscale), dim=-1).to(original.dtype)
    if not torch.equal(reference[:old_length], original):
        raise RuntimeError('Recomputed YaRN table differs within its original window')
    extended = torch.cat((original, reference[old_length:]), dim=0)
    if not torch.equal(extended, reference):
        raise RuntimeError('Extended YaRN table differs from unchanged-frequency formula')
    rope.cos_sin_cache = extended
    audit = dict(factor=rope.scaling_factor, original_max_position_embeddings=32768,
                 original_table_positions=old_length, table_positions=len(extended),
                 added_positions=required-old_length, original_entries_bitwise_equal=True,
                 extended_formula_bitwise_equal=True, mscale=rope.mscale)
    rope._yarn2_window_audit = audit
    return dict(audit)
