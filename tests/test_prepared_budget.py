"""Original prepared total-output budget and runtime accounting regressions."""
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
from runner.runtime.thinking_budget import KEY, BudgetState, validate_thinking_outcome
from runner.runtime import output_budget as budget
from test_ruler_thinking import Tokenizer, raw_row
from scripts.data.ruler import format_thinking_sample

class BudgetTests(unittest.TestCase):
    def test_natural_forced_closure_total_cap_and_prefix_scoring(self):
        from runner.reporting.aggregate import OutputAnalyzer
        tokenizer = Tokenizer()
        for prefix in ('', ' Answer:'):
            p = budget.make_policy(tokenizer, prefix)
            for count in (0, 10, 15000):
                first = [65]*count + [p['close_id']] + p['transition_ids'] + p['prefix_ids']
                for answer in ([66, 3], [66]*(16384-len(first))):
                    tokens = first+answer; state = BudgetState(p).consume(tokens)
                    self.assertEqual(state.phase, 'done'); self.assertEqual(state.closure, 'forced' if count == 15000 else 'natural')
                    self.assertEqual(state.metrics(tokenizer)['scored_text'], 'B'*(len(answer)-(answer[-1] == 3)))
                    metrics = dict(OutputAnalyzer(tokenizer).analyze(tokens, [], True), **state.metrics(tokenizer))
                    metrics.update(output_cap_reached=metrics['answer_cap_reached'], ttft_seconds=1., first_answer_content_seconds=2.)
                    validate_thinking_outcome(metrics, p)
                    self.assertEqual(metrics['answer_cap_reached'], len(tokens) == 16384)
            with self.assertRaisesRegex(ValueError, 'forced token'):
                BudgetState(p).consume([65]*15000+[66])
        s = format_thinking_sample(tokenizer, raw_row(), 'qa_1', collection_v3=True)
        self.assertEqual(s['max_output_tokens'], 16384); self.assertNotIn('answer_cap', s[KEY])
        legacy = format_thinking_sample(tokenizer, raw_row(), 'qa_1')
        self.assertEqual(s['token_ids'], legacy['token_ids']); self.assertEqual(legacy[KEY]['answer_cap'], 32)

    def test_logits_forced_close_and_scheduler_total_budget(self):
        import torch
        from runner.runtime.thinking_budget import enforce_logits, stopping_hook
        p = budget.make_policy(Tokenizer())
        tokens = [65]*15000
        request = NS(sampling_params=NS(extra_args={KEY: p}), output_token_ids=tokens,
                     num_tokens=20000, num_output_tokens=len(tokens), max_tokens=16384)
        runner = NS(requests={'r': request}, input_batch=NS(req_id_to_index={'r': 0}), speculative_config=None)
        logits = torch.zeros((1, 256)); enforce_logits(runner, logits)
        self.assertEqual(logits[0, 2], 0); self.assertEqual(torch.isfinite(logits).sum(), 1)
        first = tokens+[2]+p['transition_ids']
        request.output_token_ids = first+[66]*(16384-len(first)); request.num_output_tokens = 16384
        stop = stopping_hook(Mock(), NS(FINISHED_STOPPED='eos', FINISHED_LENGTH_CAPPED='cap'))
        self.assertTrue(stop(request, 82304)); self.assertEqual(request.status, 'cap')
        request = NS(sampling_params=NS(extra_args={KEY: p}), output_token_ids=[2]+p['transition_ids']+[3],
                     num_tokens=10, num_output_tokens=4, max_tokens=16384)
        self.assertTrue(stop(request, 82304)); self.assertEqual(request.status, 'eos')

    def test_seed_submission_and_probe_populate_exclusion(self):
        from runner.runtime.generation import generate
        s = format_thinking_sample(Tokenizer(), raw_row(), 'qa_1', collection_v3=True)
        p = s[KEY]
        class Engine:
            active = False
            def has_unfinished_requests(self): return self.active
            def add_request(self, rid, prompt, params): self.rid, self.params, self.active = rid, params, True
            def step(self):
                self.active = False
                tokens = [2]+p['transition_ids']+p['prefix_ids']+[65, 3] if self.params.max_tokens > 1 else [65]
                return [NS(request_id=self.rid, outputs=[NS(token_ids=tokens)], finished=True)]
        with patch.dict(sys.modules, {'vllm': NS(SamplingParams=lambda **kw: NS(**kw)), 'vllm.inputs': NS(TokensPrompt=lambda **kw: kw)}):
            engine = Engine(); generate(engine, s['token_ids'], 16384, 'answer', True, sample=s, seed=3)
            self.assertEqual(engine.params.seed, 3); self.assertEqual(engine.params.extra_args[KEY], p)
            generate(engine, s['token_ids'], 1, 'probe', False, sample=s)
            self.assertNotIn(KEY, engine.params.extra_args)
            generate(engine, s['token_ids'][:64], 1, 'populate', phase='populate')
            self.assertNotIn(KEY, engine.params.extra_args)


if __name__ == "__main__": unittest.main()
