"""Independent native-feature references and chunk-specific routing evidence."""
import copy
import itertools
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

from ucm.sparse.prophetkv.adatree import (
    compute_features, decide, feature_names, protocol_for, route_features,
)


def offline_reference(raw, scores, boundaries):
    """Use the original reductions: per-chunk direct sums and ranked coverage."""
    p = raw.astype(np.float64)
    p /= p.sum(1, keepdims=True)
    n = p.shape[1]
    order = np.argsort(-scores, kind='stable')
    chunks = np.repeat(np.arange(len(boundaries)-3), np.diff(boundaries[1:-1]))
    spread = {}
    for rate in (.20, .60):
        mask = np.ones(n)
        mask[order[:int(np.floor(rate*n))]] = 0
        masses = np.array([np.bincount(chunks, weights=row*mask,
                           minlength=len(boundaries)-3) for row in p])
        spread[rate] = masses.sum(1)**2 / (masses.shape[1] * (masses**2).sum(1))
    coverage = np.cumsum(p[:, order], axis=1)
    r1 = 1-coverage[:, int(np.floor(.01*n))-1]
    r5 = 1-coverage[:, int(np.floor(.05*n))-1]
    return [np.quantile(spread[.60], .90),
            np.mean(r5/np.maximum(r1, 1e-15)), spread[.20].mean()]


class Chunk2048Tests(unittest.TestCase):
    def test_native_features_match_independent_offline_reductions(self):
        rng = np.random.default_rng(42)
        for layers, tied in itertools.product((1, 8, 64), (False, True)):
            # Unequal chunks include a partial last context chunk; prefix and
            # suffix have no place in the chunk denominator or normalization.
            bounds = [0, 64, 2112, 4160, 4992, 5057]
            raw = rng.random((layers, bounds[-2]-bounds[1]), dtype=np.float32)
            scores = (rng.integers(0, 8, raw.shape[1]).astype(np.float32)
                      if tied else rng.random(raw.shape[1], dtype=np.float32))
            a, s = torch.from_numpy(raw), torch.from_numpy(scores)
            before = a.clone(), s.clone()
            result = compute_features(a, s, bounds, chunk_size=2048)
            np.testing.assert_allclose(result.numpy(), offline_reference(raw, scores, bounds), atol=1e-12, rtol=0)
            self.assertEqual(result.dtype, torch.float64)
            self.assertTrue(torch.equal(a, before[0]) and torch.equal(s, before[1]))
            scaled = compute_features(a*torch.arange(1, layers+1)[:, None], s, bounds, 2048)
            np.testing.assert_allclose(scaled.numpy(), result.numpy(), atol=1e-8, rtol=0)

    def test_inclusive_thresholds_and_adjacent_fp64_values(self):
        choices = [[0., edge, np.nextafter(edge, 1.), 1.] for edge in (.734, .737, .693)]
        for d60, retention, d20 in itertools.product(*choices):
            expected = (5 if retention <= .737 else 20) if d60 <= .734 else (20 if d20 <= .693 else 60)
            values = torch.tensor([d60, retention, d20], dtype=torch.float64)
            self.assertEqual(int(decide(values, 2048)), expected)
            self.assertEqual(route_features(*values.tolist(), chunk_size=2048), f'tree-prophetkv-{expected}')
            named = dict(zip(feature_names(2048), values.tolist()))
            self.assertEqual(route_features(**named, chunk_size=2048), f'tree-prophetkv-{expected}')

    def test_invalid_features_and_chunk_schema_fail_closed(self):
        for size in (None, True, 1024, 2048., '2048'):
            with self.subTest(size=size), self.assertRaises(ValueError):
                feature_names(size)
        self.assertNotEqual(protocol_for(2048), protocol_for(4096))
        for value in (float('nan'), float('inf'), -.01, 1.01):
            for index in range(3):
                values = [.5]*3
                values[index] = value
                with self.assertRaises(ValueError):
                    decide(torch.tensor(values, dtype=torch.float64), 2048)
                with self.assertRaises(ValueError):
                    route_features(*values, chunk_size=2048)
        with self.assertRaises(ValueError):
            route_features(G5_mean=.2, S60_p90=.7, chunk_size=2048)
        raw = torch.zeros(64, 200)
        raw[:, :120] = 1
        with self.assertRaisesRegex(ValueError, 'Zero residual'):
            compute_features(raw, torch.ones(200), [0, 64, 164, 264, 300], 2048)

    def test_explicit_chunk_size_reaches_engine_and_retired_worker(self):
        from runner.inputs.config import engine_config
        from runner.cache.temporary import configure_sparse
        cfg = engine_config('/model', 'adatree-v1', cache_dir='/fresh', adatree_chunk_size=2048)
        self.assertEqual(cfg['kv_transfer_config']['kv_connector_extra_config']['adatree_chunk_size'], 2048)
        sparse = NS(request_state=NS(pending=None), active=False, tree_f3_config={},
                    connector=NS(worker_request_id=None, store=NS(pending=[])))
        configure_sparse(sparse, 'adatree-v1', None, adatree_chunk_size=2048)
        self.assertEqual(sparse.adatree_chunk_size, 2048)
        configure_sparse(sparse, 'adatree-v1', None)
        self.assertEqual(sparse.adatree_chunk_size, 2048)
        with self.assertRaises(ValueError):
            configure_sparse(sparse, 'adatree-v1', None, adatree_chunk_size=1024)

    def test_integrated_tp_packet_names_and_schema_use_explicit_chunk(self):
        for tp in (1, 2):
            with self.subTest(tp=tp):
                self.check_tp_packet(tp)

    def check_tp_packet(self, tp):
        from ucm.sparse.prophetkv.tree_runtime import route_adatree
        bounds = [0, 64, 2112, 4160, 4224]
        layers = [torch.ones(bounds[-2]) for _ in range(64)]
        sparse = NS(model=NS(layers=[None]*64), scoring_layers=tuple(range(64)),
                    adatree_chunk_size=2048, selection_diagnostics=[], request=NS(request_id='read'))
        group = NS(world_size=tp, broadcast=lambda packet, src: packet)
        with patch.dict('sys.modules', {'vllm.distributed': NS(tensor_model_parallel_all_reduce=lambda a: a*tp)}):
            route_adatree(sparse, layers, torch.ones(4096), bounds, group)
        event, = sparse.selection_diagnostics
        self.assertEqual(event['chunk_size'], 2048)
        self.assertEqual(event['schema'], protocol_for(2048))
        self.assertEqual(tuple(event['features']), feature_names(2048))
        self.assertEqual(event['action'], route_features(**event['features'], chunk_size=2048))
        self.assertEqual(event['independent_probes'], 0)
        self.assertTrue(event['feature_mask_only'])

    def test_routing_receipt_rejects_missing_or_mismatched_chunk_and_features(self):
        from runner.router.adatree import validate_routing
        event = dict(kind='adatree_v1', schema=protocol_for(2048), chunk_size=2048,
                     request_id='read', features=dict(D60_p90=.8, T5_1_mean=.8, D20_mean=.8),
                     action='tree-prophetkv-60', ratio=.6, independent_probes=0, feature_mask_only=True)
        workers = [dict(rank=i, diagnostics=[copy.deepcopy(event),
                    dict(kind='prophetkv_selection', request_id='read')]) for i in range(2)]
        with patch('runner.runtime.generation.verify_diagnostics') as verify:
            result = validate_routing(workers, {}, 'read', 2, chunk_size=2048)
            self.assertEqual(result['chunk_size'], 2048)
            self.assertEqual(result['ratio'], .6)
            verify.assert_called_once()
        for change in (lambda e: e.pop('chunk_size'), lambda e: e.update(chunk_size=4096),
                       lambda e: e.update(schema=protocol_for(4096)),
                       lambda e: e.update(features=dict(G5_mean=.2, S60_p90=.7))):
            bad = copy.deepcopy(workers)
            change(bad[0]['diagnostics'][0])
            with self.assertRaises(RuntimeError):
                validate_routing(bad, {}, 'read', 2, chunk_size=2048)

    def test_resume_and_generation_keep_chunk_and_original_budget_provenance(self):
        from runner.router.adatree import answer_generation, generation_policy, validate_record
        from runner.runtime.output_budget import make_policy
        from test_ruler_thinking import Tokenizer
        protocol = dict(kv_chunk_size=2048, adaptive_routing=protocol_for(2048),
                        answer_generation=answer_generation())
        prepared = make_policy(Tokenizer())
        before = copy.deepcopy(prepared)
        effective = generation_policy(prepared, protocol)
        self.assertEqual((effective['thinking_cap'], effective['output_reserve']), (8192, 10000))
        self.assertEqual(prepared, before)
        routing = dict(schema=protocol_for(2048), chunk_size=2048,
                       features=dict(D60_p90=.734, T5_1_mean=.737, D20_mean=.8),
                       action='tree-prophetkv-5', ratio=.05, independent_probes=0,
                       feature_mask_only=True, included_in_native_probe=True, request_id='read')
        record = dict(routing=routing, executed_action='tree-prophetkv-5', independent_probes=0)
        with patch('runner.tree.collection.inventory', return_value={'adatree-v1': {}}):
            validate_record(protocol, record)
            record['routing']['chunk_size'] = 4096
            with self.assertRaises(ValueError):
                validate_record(protocol, record)


if __name__ == '__main__':
    unittest.main()
