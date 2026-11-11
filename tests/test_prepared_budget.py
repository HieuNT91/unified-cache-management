"""Prepared metadata never selects runtime budgets or sampling seeds."""
import copy
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from runner.inputs import prepared_policy
from runner.runtime import output_budget
from runner.router.adatree import generation_sample, answer_generation, SCHEMA
from test_ruler_thinking import Tokenizer, raw_row
from scripts.data.ruler import format_thinking_sample


class BudgetTests(unittest.TestCase):
    def test_prepared_policy_is_read_only_and_cannot_execute_as_generation(self):
        sample = format_thinking_sample(Tokenizer(), raw_row(), 'qa_1')
        before = copy.deepcopy(sample)
        self.assertEqual(prepared_policy.validate_sample_policy(sample)['thinking_cap'], 15000)
        with self.assertRaises(ValueError):
            output_budget.BudgetState(sample[prepared_policy.KEY])
        effective = generation_sample(sample, dict(adaptive_routing=SCHEMA, answer_generation=answer_generation()))
        self.assertEqual(output_budget.validate_sample_policy(effective)['thinking_cap'], 8192)
        self.assertEqual(effective['max_output_tokens'], 10000)
        self.assertEqual(sample, before)
        self.assertIs(effective['token_ids'], sample['token_ids'])
        self.assertIs(effective['boundaries'], sample['boundaries'])

    def test_sampling_seed_submission_and_population_exclusion(self):
        from runner.runtime.generation import generate
        from runner.runtime.thinking_loop import KEY as LOOP_KEY, ADATREE_CONFIG
        original = format_thinking_sample(Tokenizer(), raw_row(), 'qa_1')
        sample = generation_sample(original, dict(adaptive_routing=SCHEMA, answer_generation=answer_generation()))
        p = sample[prepared_policy.KEY]
        class Engine:
            active = False
            def has_unfinished_requests(self): return self.active
            def add_request(self, rid, prompt, params): self.rid, self.params, self.active = rid, params, True
            def step(self):
                self.active = False
                tokens = [2]+p['transition_ids']+p['prefix_ids']+[65, 3] if self.params.max_tokens > 1 else [65]
                return [NS(request_id=self.rid, outputs=[NS(token_ids=tokens)], finished=True)]
        with patch.dict(sys.modules, {'vllm': NS(SamplingParams=lambda **kw: NS(**kw)), 'vllm.inputs': NS(TokensPrompt=lambda **kw: kw)}):
            engine = Engine()
            with self.assertRaises(ValueError):
                generate(engine, original['token_ids'], 'answer', sample=original)
            generate(engine, sample['token_ids'], 'answer', sample=sample, seed=7)
            self.assertEqual((engine.params.seed, engine.params.max_tokens), (7, 10000))
            self.assertEqual(engine.params.extra_args[LOOP_KEY], ADATREE_CONFIG)
            self.assertEqual(engine.params.extra_args[prepared_policy.KEY], p)
            for key, value in p['sampling'].items():
                self.assertEqual(getattr(engine.params, key), value)
            generate(engine, sample['token_ids'][:64], 'populate', phase='populate')
            self.assertEqual((engine.params.max_tokens, engine.params.temperature, engine.params.top_k), (1, 0., 32))
            self.assertNotIn(prepared_policy.KEY, engine.params.extra_args)
            self.assertNotIn(LOOP_KEY, engine.params.extra_args)


if __name__ == '__main__': unittest.main()
