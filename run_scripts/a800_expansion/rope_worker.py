"""Extend the YaRN table before vLLM profiling or any forward pass."""
from vllm.v1.worker.gpu_worker import Worker as BaseWorker
from rope_window import ensure_rope_window


class Worker(BaseWorker):
    def load_model(self):
        super().load_model()
        for layer in self.model_runner.model.model.layers:
            ensure_rope_window(layer.self_attn.rotary_emb, self.model_config.max_model_len)
