"""CPU integration checks for the fixed native-probe AdaTreeV1 method."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from runner.inputs.config import engine_config
from runner.inputs.layout import stamp_sample
from runner.router.adatree import SCHEMA, validate_routing


class ConfigurationTests(unittest.TestCase):
    def test_adaptive_engine_uses_tree_f3_without_fixed_ratio_override(self):
        config = engine_config('/model', 'adatree-v1', cache_dir='/fresh')
        extra = config['kv_transfer_config']['kv_connector_extra_config']
        self.assertTrue(extra['adatree_v1'])
        self.assertEqual(extra['ucm_sparse_config']['ProphetKV']['method'], 'tree-prophetkv')
        self.assertEqual(extra['ucm_sparse_config']['ProphetKV']['ratio'], .05)
        self.assertFalse(extra['tree_f3']['readonly'])
        for override in ({'ratio': .2}, {'layers': [63]}, {'count': 1},
                         {'router_policy': '/policy'}, {'router_profile': True},
                         {'persistent': {'readonly': True}}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                engine_config('/model', 'adatree-v1', cache_dir='/fresh', **override)
        from runner.runtime.generation import verify_diagnostics
        with self.assertRaisesRegex(ValueError, 'validate_routing'):
            verify_diagnostics([], {}, 'adatree-v1', .05, 4, range(64))

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

    def test_cli_reuses_input_without_loading_gpu_libraries_or_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root/'model'
            model.mkdir()
            config = model/'config.json'
            config.write_text(json.dumps(dict(model_type='qwen3', num_hidden_layers=64, hidden_size=5120)))
            sample = root/'input.json'
            sample.write_text(json.dumps(stamp_sample(dict(token_ids=[1]*448,
                boundaries=[0,64,192,448], question_positions=[440], max_output_tokens=1, thinking=False,
                model_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()))))
            before = sample.read_bytes()
            args = [sys.executable, 'run.py', 'run', '--model', str(model), '--input', str(sample),
                    '--output', str(root/'output'), '--method', 'adatree-v1', '--dry-run']
            result = subprocess.run(args, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                    check=True, capture_output=True, text=True)
            extra = json.loads(result.stdout)['kv_transfer_config']['kv_connector_extra_config']
            self.assertTrue(extra['adatree_v1'])
            self.assertEqual(sample.read_bytes(), before)
            self.assertFalse((root/'output').exists())
            rejected = subprocess.run(args + ['--ratio', '.05'], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('rejects manual ratio', rejected.stderr)


class SingleRequestLifecycleTests(unittest.TestCase):
    def test_prepared_input_runs_once_and_retires_owned_tree_cache(self):
        from runner.runtime import single
        from runner.inputs.config import VERSIONS
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root/'model'
            model.mkdir()
            config = model/'config.json'
            config.write_text(json.dumps(dict(model_type='qwen3', num_hidden_layers=64, hidden_size=5120)))
            sample = stamp_sample(dict(token_ids=list(range(448)), boundaries=[0,64,192,448],
                question_positions=[440], max_output_tokens=8, thinking=False,
                model_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()))
            prepared = root/'input.json'
            prepared.write_text(json.dumps(sample))
            original = prepared.read_bytes()
            output = root/'output'
            args = NS(model=model, input=prepared, output=output, max_output_tokens=None,
                      method='adatree-v1', ratio=None, layers=None, num_layers=None, tp=1,
                      memory=.9, router_policy=None, router_id='router1', dry_run=False)
            calls = []
            worker = NS(setup=Mock(), arm=Mock(), drain=Mock(), retire=Mock())
            retirement = [dict(quiescent=True, transfers=dict(pending=0), request_bookkeeping=0)]
            def rpc(function, **kwargs):
                if function is worker.drain:
                    return []
                if function is worker.retire:
                    calls.append('retire')
                    return retirement
                return [dict(rank=0)]
            llm = NS(collective_rpc=rpc, llm_engine=NS(engine_core=NS(shutdown=Mock())), get_tokenizer=Mock())
            native_output = NS(outputs=[NS(token_ids=[42], text='answer', finish_reason='stop')], num_cached_tokens=64)
            def generate(engine, tokens, budget, request_id, *unused, **kwargs):
                calls.append(('generate', list(tokens), request_id, kwargs.get('phase')))
                (output/'scheduler.json').write_text(json.dumps(dict(request_id=request_id,
                    requests_blend_meta=0, requests_meta=0)))
                return native_output, 1., 2.
            prompt_cache = Mock(chunks=[tuple(sample['token_ids'][:64]), tuple(sample['token_ids'][64:192])])
            prompt_cache.ready.side_effect = lambda: calls.append('ready') or {'verified_shards':3}
            prompt_cache.delete.side_effect = lambda: calls.append('delete') or {}
            routing = dict(ratio=.2, action='tree-prophetkv-20')
            modules = {'vllm': NS(LLM=lambda **kwargs: llm), 'vllm.config': NS(KVTransferConfig=lambda **kwargs: kwargs),
                       'runner.runtime.worker': worker,
                       'ucm.integration.vllm.patch.apply_patch': NS(apply_all_patches=lambda: None)}
            with patch.dict(sys.modules, modules), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'GPU-test'}), \
                    patch('importlib.metadata.version', side_effect=lambda name: VERSIONS[name]), \
                    patch.object(single, 'generate', side_effect=generate), \
                    patch('runner.tree.collection.TreePromptCache', return_value=prompt_cache), \
                    patch('runner.tree.collection.retire_warmup') as retire_warmup, \
                    patch('runner.cache.lifecycle.wait_for_cache', return_value={}), \
                    patch('ucm.sparse.prophetkv.lifecycle.delete_retired_files') as delete_warmup, \
                    patch('runner.tree.prophetkv.snapshot_f3', return_value={}), \
                    patch('runner.router.adatree.validate_routing', return_value=routing) as validate, \
                    patch('runner.reporting.aggregate.OutputAnalyzer', return_value=NS(analyze=lambda *a: {})), \
                    patch('runner.reporting.aggregate.evaluation_metadata', return_value={}), \
                    patch('runner.reporting.aggregate.scoring_text', return_value='answer'), \
                    patch('runner.reporting.aggregate.score_answer', return_value=1.):
                single.run(args)
            measured = [entry for entry in calls if isinstance(entry, tuple) and entry[1] == sample['token_ids']]
            self.assertEqual(len(measured), 1)
            self.assertIsNone(measured[0][3])
            self.assertLess(calls.index('ready'), calls.index(measured[0]))
            self.assertEqual(calls[-2:], ['retire', 'delete'])
            self.assertNotEqual(calls[0][2].split(':')[0], measured[0][2].split(':')[0])
            self.assertEqual(prepared.read_bytes(), original)
            record = json.loads((output/'result.json').read_text())
            self.assertEqual(record['ratio'], .2)
            self.assertEqual(record['timings']['priming_seconds'], 0.)
            self.assertEqual(record['routing'], routing)
            validate.assert_called_once()
            delete_warmup.assert_called_once()
            retire_warmup.assert_called_once()
            llm.llm_engine.engine_core.shutdown.assert_called_once()


class ThinkingAnswerTests(unittest.TestCase):
    class Tokenizer:
        all_special_ids = [3, 4]
        eos_token_id = 3
        def convert_tokens_to_ids(self, text): return {'<think>':1, '</think>':2}[text]
        def convert_ids_to_tokens(self, token): return {1:'<think>', 2:'</think>'}[token]
        def encode(self, text, **kwargs): return list(map(ord, text))
        def decode(self, tokens, **kwargs): return ''.join(chr(token) for token in tokens)

    def policies(self):
        from runner.runtime.thinking_budget import make_policy
        from runner.runtime.output_budget import make_policy as make_total
        tokenizer = self.Tokenizer()
        return [make_policy(tokenizer, ' Answer:', 32), make_total(tokenizer, ' Answer:'), make_total(tokenizer)]

    def fixture(self, policy, capped=False):
        from runner.runtime.thinking_budget import KEY, BudgetState
        tokens = [65,66,policy['close_id']] + policy['transition_ids'] + policy['prefix_ids']
        if capped:
            tokens += [67]*policy.get('answer_cap', policy['output_reserve']-len(tokens))
        else:
            tokens += list(map(ord, 'TARGET')) + [policy['eos_ids'][0]]
        state = BudgetState(policy).consume(tokens)
        # Prompt contains a closed think block: policy state must override this
        # historical template when categorizing the new generated thinking.
        sample = dict(token_ids=[100,policy['close_id']], thinking=True,
                      max_output_tokens=policy['output_reserve'], **{KEY:policy})
        result = NS(outputs=[NS(token_ids=tokens)], ruler_thinking_state=state, first_answer_content_seconds=.2)
        return sample, result

    def test_answer_channel_and_counts_for_ruler_and_longbench_budgets(self):
        from runner.router.adatree import answer_details
        from runner.reporting.aggregate import scoring_text, score_answer
        for policy in self.policies():
            sample, result = self.fixture(policy)
            details = answer_details(result, sample, self.Tokenizer(), .1)
            self.assertEqual(details['metrics']['thinking_tokens'], 2)
            self.assertEqual(details['metrics']['generated_answer_tokens'], 6)
            self.assertEqual(details['metrics']['scored_text'], 'TARGET')
            self.assertEqual(details['timings']['first_answer_content_seconds'], .2)
            self.assertFalse(details['output_cap_reached'])
            evaluation = dict(scoring='ruler_all', references=['TARGET'])
            self.assertEqual(score_answer(scoring_text('thinking and forced prefix', details['metrics'], evaluation), evaluation), 1.)

    def test_caps_and_missing_corrupt_state_are_validated(self):
        from runner.router.adatree import answer_details
        for policy in self.policies():
            sample, result = self.fixture(policy, capped=True)
            details = answer_details(result, sample, self.Tokenizer(), .1)
            self.assertTrue(details['output_cap_reached'])
            self.assertTrue(details['metrics']['answer_cap_reached'])
            result.first_answer_content_seconds = .05
            with self.assertRaises(ValueError):
                answer_details(result, sample, self.Tokenizer(), .1)
            result.first_answer_content_seconds = .2
            result.ruler_thinking_state.phase = 'thinking'
            with self.assertRaises(RuntimeError):
                answer_details(result, sample, self.Tokenizer(), .1)
            del result.ruler_thinking_state
            with self.assertRaises(RuntimeError):
                answer_details(result, sample, self.Tokenizer(), .1)


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
