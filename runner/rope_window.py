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
