"""Private connector: request-scoped hashes and explicit transfer retirement."""
import json
import os
from pathlib import Path
from ucm.integration.vllm.blend_connector import UCMBlendConnector
from ucm.sparse.prophetkv.lifecycle import namespace_from_id, seed_value, TrackedStore


class PersistentBlendConnector(UCMBlendConnector):
    def _create_store(self, *args, **kwargs):
        return TrackedStore(super()._create_store(*args, **kwargs))

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        namespace = namespace_from_id(request.request_id)
        if self.requests_blend_meta:
            raise RuntimeError('Previous scheduler request not retired')
        self._seed = self.request_hasher(seed_value(namespace))
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def request_finished(self, request, block_ids):
        result = super().request_finished(request, block_ids)
        if result[0]:
            raise RuntimeError('Unexpected deferred request completion')
        self.requests_blend_meta.pop(request.request_id, None)
        self.requests_meta.pop(request.request_id, None)
        self.req2rag_load_chunks.pop(request.request_id, None)
        path = Path(os.environ['PROPHETKV_SCHEDULER_RECEIPT'])
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(request_id=request.request_id,
            requests_blend_meta=len(self.requests_blend_meta), requests_meta=len(self.requests_meta))))
        tmp.replace(path)
        return result
