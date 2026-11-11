"""CPU integration checks for the fixed native-probe AdaTreeV1 method."""
import copy
from types import SimpleNamespace as NS
import unittest

from runner.inputs.config import engine_config
from runner.router.adatree import SCHEMA, validate_routing


class ConfigurationTests(unittest.TestCase):
    def test_adaptive_engine_uses_tree_f3_without_fixed_ratio_override(self):
        config = engine_config('/model', 'adatree-v1', cache_dir='/fresh')
        extra = config['kv_transfer_config']['kv_connector_extra_config']
        self.assertTrue(extra['adatree_v1'])
        self.assertEqual(extra['ucm_sparse_config']['ProphetKV']['method'], 'tree-prophetkv')
        self.assertEqual(extra['ucm_sparse_config']['ProphetKV']['ratio'], .05)
        self.assertFalse(extra['tree_f3']['readonly'])
        with self.assertRaises(ValueError):
            engine_config('/model', 'adatree-v1', cache_dir='/fresh', ratio=.2)
        from runner.runtime.generation import verify_diagnostics
        with self.assertRaisesRegex(ValueError, 'validate_routing'):
            verify_diagnostics([], {}, 'adatree-v1', .05, 4)

    def test_switching_back_to_fixed_action_clears_routing(self):
        from runner.cache.temporary import configure_sparse
        sparse = NS(request_state=NS(pending=None), request=None, active=False,
                    connector=NS(worker_request_id=None, store=NS(pending={})), tree_f3_config={})
        result = configure_sparse(sparse, 'adatree-v1', None)
        self.assertEqual(result['method'], 'adatree-v1')
        self.assertEqual(sparse.method, 'tree-prophetkv')
        self.assertTrue(sparse.adatree_v1)
        self.assertEqual(sparse.ratio, .05)
        configure_sparse(sparse, 'tree-prophetkv', .4)
        self.assertFalse(sparse.adatree_v1)
        with self.assertRaises(ValueError):
            configure_sparse(sparse, 'adatree-v1', .4)
        sparse.request_state.pending = object()
        with self.assertRaisesRegex(RuntimeError, 'Retire'):
            configure_sparse(sparse, 'adatree-v1', None)







class RoutingEvidenceTests(unittest.TestCase):
    @staticmethod
    def fixture(*, features=None):
        import numpy as np
        from runner.tree.prophetkv import METHOD, PROTOCOL, budget
        from ucm.sparse.prophetkv.tree_selection import select_tree
        from ucm.sparse.prophetkv.adatree import route_features
        sample = dict(boundaries=[0,64,192,448], question_positions=[300])
        values = features or dict(G5_mean=.2, S60_p90=.73)
        action = route_features(**values)
        ratio = int(action.rsplit('-', 1)[1])/100
        count, forced = budget(sample['boundaries'], ratio)
        selected, _ = select_tree(np.ones(128), np.ones(128), sample['boundaries'], ratio)
        positions = selected + list(range(192,448))
        routing = dict(kind='adatree_v1', schema=SCHEMA, request_id='answer',
                       features=values, action=action, ratio=ratio,
                       independent_probes=0, feature_mask_only=True)
        selection = dict(kind='prophetkv_selection', request_id='answer', selection_method=METHOD,
            tree_protocol=PROTOCOL, scores=[1.]*128, selected_positions=selected,
            selected_count=count, eligible_count=128, ratio=ratio, unit_budget=count,
            forced=forced, f3_links=[1.]*128, f3_layers=list(range(16)),
            question_positions=sample['question_positions'], probe_question_positions=list(range(300,448)),
            recompute_mask='prefix-fresh-earlier-own-chunk-causal', suffix_mask='prefix-fresh-causal-suffix',
            decode_mask='native-unmasked', scoring_layers=list(range(64)),
            fusion='mean_layers_fp32', alignment_count=64)
        step = dict(kind='prefill_step', start=64, end=448, scheduled_tokens=384,
                    recomputed_tokens=len(positions), no_forward=False, prefill_complete=True)
        layers = [dict(kind='layer_counts', layer=f'model.layers.{i}.self_attn.attn', start=64, end=448,
                       selected_positions=positions, selected_set_verified=True, projection_tokens=len(positions),
                       attention_tokens=len(positions), ffn_tokens=len(positions)) for i in range(64)]
        workers = [dict(rank=i, diagnostics=copy.deepcopy([routing, selection, step, *layers])) for i in range(2)]
        return sample, workers

    def test_all_leaves_validate_native_tree_budget_without_routing_timing(self):
        for g, s, expected in ((.9,.73,.05), (.22,.8,.2), (.23,.8,.4)):
            sample, workers = self.fixture(features=dict(G5_mean=g, S60_p90=s))
            result = validate_routing(workers, sample, 'answer', 2)
            self.assertEqual(result['ratio'], expected)
            self.assertEqual(set(result), {'schema', 'features', 'action', 'ratio', 'request_id',
                'independent_probes', 'feature_mask_only', 'included_in_native_probe'})

    def test_corrupt_routing_or_native_evidence_fails(self):
        sample, original = self.fixture()
        mutations = (
            lambda w: w[0]['diagnostics'][0].update(ratio=.4),
            lambda w: w[1]['diagnostics'][0].update(features=dict(G5_mean=.21,S60_p90=.73)),
            lambda w: w[0]['diagnostics'][0].update(independent_probes=1),
            lambda w: w[0]['diagnostics'][0].update(feature_mask_only=False),
            lambda w: w[0]['diagnostics'][0].update(request_id='other'),
            lambda w: w[0]['diagnostics'].append(copy.deepcopy(w[0]['diagnostics'][0])),
            lambda w: w[1]['diagnostics'].pop(),
            lambda w: w[1].update(rank=0),
        )
        for change in mutations:
            workers = copy.deepcopy(original)
            change(workers)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_routing(workers, sample, 'answer', 2)


if __name__ == '__main__':
    unittest.main()
