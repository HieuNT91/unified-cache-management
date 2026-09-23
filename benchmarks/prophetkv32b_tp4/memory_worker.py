"""Private memory-bounded Qwen3 execution; no shared package edits."""
import inspect
import json
import os
from pathlib import Path
import textwrap
import types

import torch
from vllm.v1.worker.gpu_worker import Worker as BaseWorker

TILE = 4096


def tiled_mlp(self, x):
    if len(x) <= TILE:
        return self._untiled_forward(x)
    output = torch.empty_like(x)
    for start in range(0, len(x), TILE):
        output[start:start + TILE].copy_(self._untiled_forward(x[start:start + TILE]))
    return output


def tiled_output_projection(self, x):
    if len(x) <= TILE:
        return self._untiled_forward(x)
    output = torch.empty((len(x), self.output_size), device=x.device, dtype=x.dtype)
    for start in range(0, len(x), TILE):
        part, bias = self._untiled_forward(x[start:start + TILE])
        if bias is not None:
            raise RuntimeError('Qwen3 output projection must have no bias')
        output[start:start + TILE].copy_(part)
    return output, None


class Worker(BaseWorker):
    @torch.inference_mode()
    def load_model(self):
        super().load_model()
        model = self.model_runner.model
        from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group
        rank = get_tensor_model_parallel_rank()
        # Save the exact loaded per-rank weights once to avoid rereading all
        # 61 GiB on every TP rank for every qualification/measurement engine.
        from common import settings, dump
        directory = Path(settings()['cache']) / 'checkpoint-tp4'
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / 'ready.json').exists():
            from vllm.model_executor.model_loader import ShardedStateLoader
            from safetensors import safe_open
            ShardedStateLoader.save_model(model, str(directory), max_size=2**30)
            state = ShardedStateLoader._filter_subtensors(model.state_dict())
            count = 0
            for path in directory.glob(f'model-rank-{rank}-part-*.safetensors'):
                with safe_open(path, framework='pt') as saved:
                    for name in saved.keys():
                        if not torch.equal(saved.get_tensor(name), state[name].detach().cpu()):
                            raise RuntimeError('Saved TP checkpoint differs from loaded original weights')
                        count += 1
            if count != len(state):
                raise RuntimeError('Incomplete TP checkpoint export')
            dump(directory / f'rank-{rank}.json', dict(rank=rank, tensors=count, bitwise_equal=True))
            get_tp_group().barrier()
            if rank == 0:
                dump(directory / 'ready.json', dict(complete=True, tensor_parallel_size=4,
                    model=settings()['model'], ranks=[json.loads((directory / f'rank-{r}.json').read_text()) for r in range(4)]))
            get_tp_group().barrier()
            print(f'CHECKPOINT_EXPORTED rank={rank} bitwise_equal_tensors={count}', flush=True)
        audits = []
        for index, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            original = mlp.forward
            mlp._untiled_forward = original
            mlp.forward = types.MethodType(tiled_mlp, mlp)
            # All-layer, actual-size numerical check before the KV allocation.
            generator = torch.Generator(device=self.device).manual_seed(20260921 + index)
            x = torch.randn((TILE + 17, 5120), generator=generator,
                            device=self.device, dtype=torch.bfloat16)
            reference = original(x)
            actual = mlp(x)
            error = (reference.float() - actual.float()).abs()
            relative_l2 = float(error.norm() / reference.float().norm())
            normalized_max = float(error.max() / reference.abs().max())
            # Different BF16 GEMM row shapes can differ by one output ULP;
            # a near-zero elementwise denominator is not a sound norm test.
            # Bound both the full tensor's L2 error and its largest-scale error.
            passed = relative_l2 <= .005 and normalized_max <= .01
            audits.append(dict(layer=index, max_abs_error=float(error.max()),
                relative_l2=relative_l2, normalized_max=normalized_max, passed=passed))
            if not passed:
                print(f'MLP_DIAGNOSTIC rank={rank} layer={index} relative_l2={relative_l2} normalized_max={normalized_max}', flush=True)
                raise RuntimeError(f'Tiled MLP numerical mismatch at layer {index}')
            del x, reference, actual, error
            # Drop dead QKV/norm views before allocating the output projection.
            # Preserve the original forward's exact operations and UCM hooks.
            attention = layer.self_attn
            projection = attention.o_proj
            projection._untiled_forward = projection.forward
            projection.forward = types.MethodType(tiled_output_projection, projection)
            x = torch.randn((TILE + 17, attention.q_size), generator=generator,
                            device=self.device, dtype=torch.bfloat16)
            reference, _ = projection._untiled_forward(x)
            actual, _ = projection(x)
            error = (reference.float() - actual.float()).abs()
            relative_l2 = float(error.norm() / reference.float().norm())
            normalized_max = float(error.max() / reference.abs().max())
            if relative_l2 > .005 or normalized_max > .01:
                raise RuntimeError(f'Tiled attention output projection mismatch at layer {index}')
            audits[-1]['output_projection'] = dict(relative_l2=relative_l2,
                normalized_max=normalized_max, passed=True)
            del x, reference, actual, error

            source = textwrap.dedent(inspect.getsource(type(attention).forward))
            source = source.replace('output, _ = self.o_proj(attn_output)',
                'del qkv, q, k, v, q_by_head, k_by_head\n    output, _ = self.o_proj(attn_output)')
            scope = dict(torch=torch)
            exec(compile(source, __file__, 'exec'), scope)
            attention.forward = types.MethodType(scope['forward'], attention)
        output = os.environ.get('TP4_MEMORY_AUDIT')
        if output:
            dump(Path(output + f'.rank{rank}.json'), dict(rank=rank, tile_tokens=TILE, layers=audits))
        torch.cuda.empty_cache()
        print(f'MEMORY_ADAPTATION_VERIFIED rank={rank} layers={len(audits)} tile={TILE}', flush=True)


def loader_options():
    from common import settings
    directory = Path(settings()['cache']) / 'checkpoint-tp4'
    options = dict(worker_cls='memory_worker.Worker')
    # vLLM 0.9.2 V1 does not support sharded_state; use original safetensors.
    return options
