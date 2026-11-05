"""Bound activation temporaries without changing context positions or RoPE."""
def install_tiles(model, tile=4096):
    import torch
    def tiled(module, pair):
        original=module.forward
        def forward(x):
            if len(x) <= tile:
                return original(x)
            parts=[original(x[i:i+tile]) for i in range(0,len(x),tile)]
            if pair:
                biases=[p[1] for p in parts]
                if any(b is not None for b in biases):
                    raise RuntimeError('Unexpected output-projection bias')
                return torch.cat([p[0] for p in parts]),None
            return torch.cat(parts)
        module.forward=forward
    for layer in model.layers:
        tiled(layer.mlp,False)
        tiled(layer.self_attn.o_proj,True)
