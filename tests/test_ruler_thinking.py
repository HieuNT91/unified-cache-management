"""Native RULER preparation, control tokens and answer accounting."""
import copy
import gc
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

from runner.runtime.thinking_budget import (KEY, PROTOCOL, THINKING_CAP, WINDOW,
    BudgetState, make_policy, request_state)
from scripts.data.ruler import TASKS, format_thinking_sample, generator_command


class Tokenizer:
    chat_template = 'native-thinking-test'
    all_special_ids = [3, 4]  # Qwen3 thinking delimiters are not tokenizer specials.
    eos_token_id = 3
    def convert_tokens_to_ids(self, text): return {'<think>':1, '</think>':2}[text]
    def convert_ids_to_tokens(self, token): return {1:'<think>', 2:'</think>'}[token]
    def apply_chat_template(self, messages, **kwargs):
        tail = '' if kwargs['enable_thinking'] else '<think>\n\n</think>\n\n'
        return '[USER]' + messages[0]['content'] + '[END]assistant\n' + tail
    def encode(self, text, **kwargs): return list(map(ord, text))
    def __call__(self, text, **kwargs):
        return dict(input_ids=self.encode(text), offset_mapping=[(i,i+1) for i in range(len(text))])
    def decode(self, tokens, **kwargs): return ''.join(chr(t) for t in tokens)


def raw_row(task='qa_1', context=9000):
    q = 'What is KEY?' if task.startswith('niah_') else 'Question: What is KEY?'
    return dict(index=0, input=Tokenizer().apply_chat_template([dict(content='x'*context+'\n'+q)], enable_thinking=False),
                answer_prefix=' Answer:', outputs=['TARGET'])


def sample(task='qa_1'):
    return format_thinking_sample(Tokenizer(),raw_row(task),task)


def closed(p, thinking=None, answer=None):
    return (thinking or []) + [p['close_id']] + p['transition_ids'] + p['prefix_ids'] + (answer or [])


class ThinkingTests(unittest.TestCase):


    def test_early_eos_empty_answer_illegal_controls_and_cleanup(self):
        p=make_policy(Tokenizer(),' Answer:',32)
        for answer in ([3], [32,65,3]):
            state=BudgetState(p).consume(closed(p,answer=answer))
            self.assertEqual(state.phase,'done');self.assertTrue(state.answer_eos)
            self.assertEqual(state.answer,len(answer)-1)
        for tokens in ([3], [1,1], closed(p,answer=[1]), closed(p,answer=[2])):
            with self.assertRaises(ValueError):BudgetState(p).consume(tokens)
        with self.assertRaises(ValueError):BudgetState(p).consume([2,65])
        req=NS(sampling_params=NS(extra_args={KEY:p}),output_token_ids=[])
        state=request_state(req); ref=weakref.ref(state)
        self.assertIs(request_state(req),state)
        del state,req;gc.collect();self.assertIsNone(ref())
        self.assertIsNone(request_state(NS(sampling_params=NS(extra_args={}),output_token_ids=[])))


    def test_native_preparation_preserves_source_question_and_defers_prefix(self):
        tokenizer=Tokenizer()
        for task in TASKS:
            row=raw_row(task);before=copy.deepcopy(row)
            s=format_thinking_sample(tokenizer,row,task)
            self.assertEqual(row,before)
            self.assertEqual(s['evaluation_protocol'],PROTOCOL)
            self.assertTrue(s['thinking']);self.assertFalse(s['truncated'])
            self.assertNotIn(row['answer_prefix'],s['formatted_text'])
            self.assertEqual(tokenizer.decode([s['token_ids'][i] for i in s['question_positions']]),s['question'])
            self.assertTrue(all(i>=s['boundaries'][-2] for i in s['question_positions']))
            self.assertEqual(tokenizer.decode(s[KEY]['prefix_ids']),row['answer_prefix'])
        with self.assertRaisesRegex(ValueError,'boundaries'):
            format_thinking_sample(tokenizer,dict(row,input='bad'),task)
        with self.assertRaises(ValueError):format_thinking_sample(tokenizer,raw_row(context=WINDOW),'qa_1')

    def test_generator_budget_seed_and_single_30_row_batch_unchanged(self):
        args=NS(ruler=Path('/r'),model=Path('/m'),samples=30,seed=42,thinking=True)
        defs={'qa_1':dict(task='qa',args={'dataset':'squad'})}
        const={'qa':dict(template='{context}\nQuestion: {query}',answer_prefix=' Answer:',tokens_to_generate=32)}
        with patch('scripts.data.ruler.definitions',return_value=(defs,const)):
            thinking=generator_command(args,'qa_1',Path('/raw'),Tokenizer())
            args.thinking=False
            self.assertEqual(thinking,generator_command(args,'qa_1',Path('/raw'),Tokenizer()))
        for flag,value in [('--num_samples','30'),('--max_seq_length','65536'),('--tokens_to_generate','32'),('--random_seed','42')]:
            self.assertEqual(thinking[thinking.index(flag)+1],value)
        self.assertEqual(thinking[0],sys.executable)
        self.assertEqual(thinking[thinking.index('--dataset')+1],'squad')


    def test_answer_only_credit_nulls_and_full_stream_accounting(self):
        from runner.reporting.aggregate import OutputAnalyzer,scoring_text,score_answer,metrics
        p=sample()[KEY];tok=Tokenizer();analysis=OutputAnalyzer(tok)
        tokens=closed(p,tok.encode('TARGET'),[3])
        state=BudgetState(p).consume(tokens)
        values=analysis.analyze(tokens,[1],True);values.update(state.metrics(tok))
        evaluation=dict(scoring='ruler_any',references=['TARGET'])
        self.assertEqual(score_answer(scoring_text('TARGET',values,evaluation),evaluation),0.)
        record=dict(values,prediction='TARGET',accuracy=0.,scoring='ruler_any',output_cap_reached=False,
                    timings=dict(ttft_seconds=1.,first_answer_content_seconds=None))
        report=metrics([record],1)
        self.assertEqual(report['ruler_nulls'],'1/1')
        self.assertIsNone(report['mean_first_answer_content_seconds'])
        self.assertEqual(values['thinking_tokens']+values['answer_tokens']+values['control_tokens'],len(tokens))
        self.assertEqual(report['natural_thinking_closures'],1)


    def test_initial_opener_is_control_and_no_second_block_can_start(self):
        p=make_policy(Tokenizer(),' Answer:',32)
        tokens=[p['open_id']]+closed(p,[32]*THINKING_CAP,[65]*32)
        state=BudgetState(p).consume(tokens)
        self.assertEqual(state.seen,p['output_reserve'])
        self.assertEqual(state.opening_tokens,1)
        self.assertEqual(state.thinking,THINKING_CAP)
        self.assertEqual(state.forced,1+len(p['transition_ids'])+len(p['prefix_ids']))
        with self.assertRaises(ValueError):BudgetState(p).consume([1,32,1])
        p=make_policy(Tokenizer(),' Answer:',32,opening_in_prompt=True)
        with self.assertRaises(ValueError):BudgetState(p).consume([1])
