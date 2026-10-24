"""Verify native factor-4 YaRN table, with no factor-2 extension or shared edits."""
import torch

def ensure_rope_window(rope,required):
    if rope.scaling_factor!=4.0 or rope.max_position_embeddings!=32768 or required!=131072:
        raise ValueError('Expected YaRN4, original 32768, total 131072')
    if hasattr(rope,'_yarn4_window_audit'):
        return dict(rope._yarn4_window_audit)
    table=rope.cos_sin_cache
    if len(table)!=required: raise ValueError('Factor-4 table does not cover full window')
    with torch.device(table.device):
        inv=rope._compute_inv_freq(rope.scaling_factor)
        freq=torch.einsum('i,j -> ij',torch.arange(required,dtype=torch.float32),inv)
        reference=torch.cat((freq.cos()*rope.mscale,freq.sin()*rope.mscale),dim=-1).to(table.dtype)
    if not torch.equal(reference,table): raise ValueError('YaRN4 table formula mismatch')
    audit=dict(factor=4.0,original_max_position_embeddings=32768,table_positions=required,
               formula_bitwise_equal=True,added_positions=0,mscale=rope.mscale)
    rope._yarn4_window_audit=audit
    return dict(audit)


def ensure_native_rope(rope,positions):
    """Verify an unscaled RoPE table (Qwen3-4B-Instruct-2507 profile)."""
    if hasattr(rope,'_native_window_audit'):
        return dict(rope._native_window_audit)
    if type(rope).__name__!='RotaryEmbedding' or rope.max_position_embeddings!=positions:
        raise ValueError('Expected an unscaled RoPE table over the native window')
    table=rope.cos_sin_cache
    if len(table)!=positions: raise ValueError('Native RoPE table does not cover the native window')
    with torch.device(table.device):
        inv=rope._compute_inv_freq(rope.base)
        freq=torch.einsum('i,j -> ij',torch.arange(positions,dtype=torch.float32),inv)
        reference=torch.cat((freq.cos(),freq.sin()),dim=-1).to(table.dtype)
    if not torch.equal(reference,table): raise ValueError('Native RoPE table formula mismatch')
    audit=dict(factor=None,table_positions=positions,formula_bitwise_equal=True,added_positions=0,mscale=1.0,
               base=rope.base)
    rope._native_window_audit=audit
    return dict(audit)
