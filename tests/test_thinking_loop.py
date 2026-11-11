"""CPU tests for exact repetition -> Qwen3 thinking closure -> answer completion."""
import copy
import random
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from runner.runtime import output_budget
from runner.runtime.thinking_budget import (KEY, BudgetState, make_policy, request_state,
                                    enforce_logits, stopping_hook, validate_thinking_outcome)
from runner.runtime.thinking_loop import CONFIG, KEY as LOOP_KEY, ThinkingLoop, repetition_pattern_size
from runner.reporting.aggregate import OutputAnalyzer
from test_ruler_thinking import Tokenizer, sample


class DetectorTests(unittest.TestCase):
    def test_boundaries_exact_adjacent_tail_and_shorter_periods(self):
        for size in (8, 9, 127, 128):
            block = list(range(100, 100 + size))
            self.assertIsNone(repetition_pattern_size(block * 2 + block[:-1]))
            self.assertEqual(repetition_pattern_size([999] + block * 3), size)
            altered = block * 3
            altered[size + 2] = 999
            self.assertIsNone(repetition_pattern_size(altered))
            self.assertIsNone(repetition_pattern_size(block * 3 + [999]))
            self.assertIsNone(repetition_pattern_size(block + [999] + block * 2))
        self.assertIsNone(repetition_pattern_size(list(range(100, 229)) * 3))
        # vLLM tests periods, not the primitive/minimal period: a single token
        # repeated 24 times also consists of three identical 8-token blocks.
        self.assertIsNone(repetition_pattern_size([100] * 23))
        self.assertEqual(repetition_pattern_size([100] * 24), 8)

    def test_matches_independent_block_oracle(self):
        rng = random.Random(42)
        for _ in range(250):
            tokens = [rng.randrange(5) for _ in range(rng.randrange(500))]
            if rng.randrange(2):
                block = [rng.randrange(5) for _ in range(rng.randrange(1, 140))]
                tokens += block * rng.randrange(2, 5)
            expected = next((n for n in range(8, 129) if len(tokens) >= 3*n
                             and tokens[-n:] == tokens[-2*n:-n] == tokens[-3*n:-2*n]), None)
            self.assertEqual(repetition_pattern_size(tokens), expected)

    def test_bounded_history_and_invalid_configuration(self):
        loop = ThinkingLoop(CONFIG)
        for i in range(2000):
            loop.observe(100 + i, i + 1)
        self.assertFalse(loop.broken)
        self.assertEqual(len(loop.tail), 384)
        for i, token in enumerate(list(range(100, 228)) * 3, 2001):
            loop.observe(token, i)
        self.assertTrue(loop.broken)
        self.assertEqual(loop.pattern_size, 128)
        self.assertEqual(loop.token_index, 2384)
        with self.assertRaises(ValueError):
            ThinkingLoop(dict(CONFIG, min_count=2))


class ThinkingLoopTests(unittest.TestCase):
    def policies(self):
        return [make_policy(Tokenizer(), ' Answer:', 32),
                output_budget.make_policy(Tokenizer(), ' Answer:'),
                output_budget.make_policy(Tokenizer())]

    def outcome(self, state, tokens):
        values = OutputAnalyzer(Tokenizer()).analyze(tokens, [], True)
        values.update(state.metrics(Tokenizer()), ttft_seconds=.1,
                      first_answer_content_seconds=.2 if state.answer else None)
        values['output_cap_reached'] = (values['answer_cap_reached'] if state.policy['protocol'] == output_budget.PROTOCOL
                                       else values['thinking_cap_reached'] or values['answer_cap_reached'])
        return values

    def test_break_then_answer_and_validate_accounting(self):
        for policy in self.policies():
            for opening in (False, True):
                p = dict(policy, opening_in_prompt=opening)
                if p['protocol'] != output_budget.PROTOCOL and opening:
                    p['output_reserve'] -= 1
                before = copy.deepcopy(p)
                tokens = ([] if opening else [p['open_id']]) + list(range(100, 108)) * 3
                state = BudgetState(p, CONFIG).consume(tokens)
                self.assertEqual(state.phase, 'thinking')
                self.assertEqual(state.forced_next(), p['close_id'])
                self.assertEqual(state.loop.token_index, 24)
                tokens += [p['close_id']] + p['transition_ids'] + p['prefix_ids']
                state.consume(tokens)
                self.assertEqual(state.phase, 'answer')
                # A repeating answer must remain untouched and complete on EOS.
                answer = [65] * 24
                tokens += answer + [p['eos_ids'][0]]
                state.consume(tokens)
                self.assertEqual(state.phase, 'done')
                self.assertEqual(state.answer_ids, answer)
                self.assertEqual(state.closure, 'repetition')
                values = self.outcome(state, tokens)
                self.assertEqual(values['stop_reason'], 'thinking_broken')
                self.assertFalse(values['unfinished_thinking'])
                self.assertFalse(values['thinking_cap_reached'])
                validate_thinking_outcome(values, p)
                self.assertEqual(p, before)
                for field, bad in [('thinking_broken', False), ('thinking_break_pattern_size', 129),
                                   ('thinking_break_token_index', 25), ('forced_tokens', 0),
                                   ('stop_reason', None), ('thinking_closure', 'natural')]:
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        validate_thinking_outcome(dict(values, **{field: bad}), p)
                del values[LOOP_KEY]
                with self.assertRaises(ValueError):
                    validate_thinking_outcome(values, p)

    def test_natural_closure_and_legacy_records_remain_valid(self):
        for p in self.policies():
            for config in (None, CONFIG):
                tokens = [65] * 23 + [p['close_id']] + p['transition_ids'] + p['prefix_ids'] + [66, 3]
                state = BudgetState(p, config).consume(tokens)
                self.assertEqual(state.closure, 'natural')
                self.assertFalse(state.loop.broken)
                validate_thinking_outcome(self.outcome(state, tokens), p)
            # Old state replay does not retroactively reinterpret repetition.
            tokens = [65] * 100 + [p['close_id']] + p['transition_ids'] + p['prefix_ids'] + [3]
            self.assertEqual(BudgetState(p).consume(tokens).closure, 'natural')

    def test_cap_still_forces_close_and_answer_limit_still_applies(self):
        for p in self.policies():
            tokens = list(range(100, 100 + p['thinking_cap']))
            state = BudgetState(p, CONFIG).consume(tokens)
            self.assertEqual(state.forced_next(), p['close_id'])
            tokens += [p['close_id']] + p['transition_ids'] + p['prefix_ids']
            count = p.get('answer_cap', p['output_reserve'] - len(tokens))
            tokens += [65] * count
            state.consume(tokens)
            self.assertEqual(state.closure, 'forced')
            self.assertFalse(state.loop.broken)
            self.assertEqual(state.phase, 'done')
            validate_thinking_outcome(self.outcome(state, tokens), p)

    def test_scheduler_worker_and_generation_continue_same_request(self):
        import torch
        from runner.runtime.generation import generate
        s = sample(); p = s[KEY]
        statuses = NS(FINISHED_STOPPED='stop', FINISHED_LENGTH_CAPPED='length')
        stop = stopping_hook(Mock(side_effect=AssertionError('unmanaged request')), statuses)

        class Engine:
            def has_unfinished_requests(self): return getattr(self, 'active', False)
            def add_request(self, rid, prompt, params):
                self.params = params
                self.prompt = prompt
                self.active = True
                self.rid = rid
                self.tokens = []
                self.worker = NS(sampling_params=params, output_token_ids=self.tokens)
                self.request = NS(sampling_params=params, output_token_ids=self.tokens,
                                  num_tokens=len(s['token_ids']), num_output_tokens=0,
                                  max_tokens=params.max_tokens, stop_reason=None)
                self.runner = NS(requests={rid: self.worker},
                                 input_batch=NS(req_id_to_index={rid: 0}), speculative_config=None)
            def step(self):
                state = request_state(self.worker)
                suggested = (100 + len(self.tokens) % 8 if state.phase == 'thinking'
                             else 65 if state.answer < 2 else 3)
                logits = torch.zeros((1, 256)); logits[0, suggested] = 10.
                enforce_logits(self.runner, logits)
                self.tokens.append(int(logits.argmax()))
                self.request.num_tokens += 1; self.request.num_output_tokens += 1
                done = stop(self.request, 82304)
                self.active = not done
                if not done:
                    assert self.request.stop_reason is None
                return [NS(request_id=self.rid, finished=done, outputs=[NS(token_ids=list(self.tokens),
                           finish_reason=getattr(self.request, 'status', None), stop_reason=self.request.stop_reason)])]

        with patch.dict(sys.modules, {'vllm': NS(SamplingParams=lambda **kw: NS(**kw)),
                                     'vllm.inputs': NS(TokensPrompt=lambda **kw: kw)}):
            engine = Engine()
            result, ttft, _ = generate(engine, s['token_ids'], p['output_reserve'], 'answer', True, sample=s)
        self.assertEqual(engine.prompt['prompt_token_ids'], s['token_ids'])
        self.assertEqual(engine.params.extra_args[KEY], p)
        self.assertEqual(engine.params.extra_args[LOOP_KEY], CONFIG)
        self.assertEqual(result.outputs[0].finish_reason, 'stop')
        self.assertEqual(result.outputs[0].stop_reason, 'thinking_broken')
        self.assertEqual(result.ruler_thinking_state.answer_ids, [65, 65])
        self.assertGreaterEqual(result.first_answer_content_seconds, ttft)
        for state in (request_state(engine.worker), request_state(engine.request), result.ruler_thinking_state):
            self.assertEqual(state.closure, 'repetition')
            self.assertEqual(state.phase, 'done')

    def test_request_config_is_immutable_and_state_does_not_leak(self):
        p = self.policies()[0]
        req = NS(sampling_params=NS(extra_args={KEY: p, LOOP_KEY: dict(CONFIG)}),
                 output_token_ids=[65] * 24)
        self.assertTrue(request_state(req).loop.broken)
        req.sampling_params.extra_args[LOOP_KEY]['min_count'] = 4
        with self.assertRaisesRegex(ValueError, 'changed'):
            request_state(req)
        fresh = NS(sampling_params=NS(extra_args={KEY: p, LOOP_KEY: CONFIG}), output_token_ids=[])
        self.assertFalse(request_state(fresh).loop.broken)
        with self.assertRaisesRegex(ValueError, 'forced token'):
            BudgetState(p, CONFIG).consume([65] * 25)

    def test_real_vllm_wire_and_scheduler_stop_reason(self):
        from vllm import SamplingParams
        from vllm.v1.request import Request, RequestStatus
        from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
        p = self.policies()[0]
        params = SamplingParams(max_tokens=p['output_reserve'], ignore_eos=True,
                                extra_args={KEY: p, LOOP_KEY: CONFIG})
        wire = MsgpackDecoder(SamplingParams).decode(MsgpackEncoder().encode(params))
        self.assertEqual(wire.extra_args[LOOP_KEY], CONFIG)
        stop = stopping_hook(Mock(side_effect=AssertionError('unmanaged request')), RequestStatus)
        for answer in ([65, 3], [65] * p['answer_cap'], [3]):
            req = Request('loop', [65] * 100, None, None, None, wire, None, 3)
            tokens = [65] * 24 + [p['close_id']] + p['transition_ids'] + p['prefix_ids'] + answer
            for i, token in enumerate(tokens):
                req.append_output_token_ids(token)
                self.assertEqual(stop(req, 82304), i == len(tokens) - 1)
                if i < len(tokens) - 1:
                    self.assertIsNone(req.stop_reason)
            self.assertEqual(req.stop_reason, 'thinking_broken')
            self.assertEqual(req.status, RequestStatus.FINISHED_STOPPED if answer[-1] == 3
                             else RequestStatus.FINISHED_LENGTH_CAPPED)
            state = request_state(req)
            validate_thinking_outcome(self.outcome(state, tokens), p)

    def test_repetition_at_thinking_cap_records_both_causes(self):
        for p in self.policies():
            tokens = list(range(1000, 1000 + p['thinking_cap'] - 24)) + [65] * 24
            tokens += [p['close_id']] + p['transition_ids'] + p['prefix_ids'] + [66, 3]
            state = BudgetState(p, CONFIG).consume(tokens)
            values = self.outcome(state, tokens)
            self.assertTrue(values['thinking_cap_reached'])
            self.assertTrue(values['thinking_broken'])
            validate_thinking_outcome(values, p)

    def test_report_keeps_successful_broken_answers(self):
        from scripts.collection.collection_v3 import answer_metrics
        from runner.reporting.aggregate import metrics
        p = output_budget.make_policy(Tokenizer())
        tokens = [65] * 24 + [p['close_id']] + p['transition_ids'] + [66, 3]
        state = BudgetState(p, CONFIG).consume(tokens)
        record = self.outcome(state, tokens)
        record.update(accuracy=1., finish_reason='stop', gpu_uuids=['a', 'b'],
                      timings=dict(ttft_seconds=.1, first_answer_content_seconds=.2))
        summary = metrics([answer_metrics(record)], 1)
        self.assertEqual(summary['completed'], 1)
        self.assertEqual(summary['thinking_broken'], 1)
        self.assertEqual(summary['unfinished_thinking'], 0)
        self.assertEqual(summary['thinking_cap_reached'], 0)
        self.assertEqual(summary['accuracy_percent'], 100.)


if __name__ == '__main__':
    unittest.main()
