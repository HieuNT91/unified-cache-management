"""CPU-only phase budgets, native preparation and runtime admission."""
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import weakref

from runner.runtime.thinking_budget import (KEY, PROFILE, PROTOCOL, THINKING_CAP, WINDOW,
    BudgetState, make_policy, enforce_logits, stopping_hook, request_state)
from scripts.data.ruler import TASKS, CAPS, format_thinking_sample, generator_command

ROOT = Path(__file__).resolve().parents[1]


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
    def test_natural_close_prefix_once_and_all_task_caps(self):
        for task in TASKS:
            p=make_policy(Tokenizer(),' Answer:',CAPS[task])
            tokens=closed(p,[32,32],[65]*CAPS[task])
            state=BudgetState(p).consume(tokens)
            self.assertEqual(state.phase,'done')
            self.assertEqual(state.closure,'natural')
            self.assertEqual(state.thinking,2)  # whitespace is generated content
            self.assertEqual(state.answer,CAPS[task])
            self.assertEqual(state.forced,len(p['transition_ids'])+len(p['prefix_ids']))
            self.assertEqual(state.metrics(Tokenizer())['scored_text'],'A'*CAPS[task])
            self.assertEqual(state.first_answer_index,3+state.forced)

    def test_exact_thinking_cap_forces_close_then_full_answer_budget(self):
        p=make_policy(Tokenizer(),' Answer:',30)
        state=BudgetState(p).consume([32]*THINKING_CAP)
        self.assertEqual(state.forced_next(),p['close_id'])
        tokens=closed(p,[32]*THINKING_CAP,[32]*30)
        state.consume(tokens)
        self.assertEqual(state.seen,p['output_reserve']-1)
        self.assertEqual(state.phase,'done')
        self.assertEqual(state.closure,'forced')
        self.assertTrue(state.metrics(Tokenizer())['thinking_cap_reached'])
        self.assertTrue(state.metrics(Tokenizer())['answer_cap_reached'])
        natural=BudgetState(p).consume(closed(p,[32]*(THINKING_CAP-1),[3]))
        self.assertEqual(natural.closure,'natural')
        self.assertEqual(natural.answer,0)
        with self.assertRaisesRegex(ValueError,'forced token'):
            BudgetState(p).consume([32]*THINKING_CAP+[65])

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

    def test_worker_logits_and_scheduler_for_baseline_sparse_and_probes(self):
        import torch
        p=make_policy(Tokenizer(),' Answer:',30)
        statuses=NS(FINISHED_STOPPED='eos',FINISHED_LENGTH_CAPPED='cap')
        original=Mock(return_value='legacy');stop=stopping_hook(original,statuses)
        for rid in ('baseline','sparse'):
            req=NS(sampling_params=NS(extra_args={KEY:p}),output_token_ids=[],num_tokens=9000,
                   num_output_tokens=0,max_tokens=p['output_reserve'])
            runner=NS(requests={rid:req},input_batch=NS(req_id_to_index={rid:0}),speculative_config=None)
            logits=torch.zeros((1,256));enforce_logits(runner,logits)
            self.assertEqual(logits[0,1],0.);self.assertTrue(torch.isneginf(logits[0,3]))
            self.assertEqual(logits[0,2],0.)
            req.output_token_ids=[32]*THINKING_CAP
            logits.zero_();enforce_logits(runner,logits)
            self.assertEqual(torch.isfinite(logits).sum(),1);self.assertEqual(logits[0,2],0.)
            self.assertFalse(stop(req,WINDOW))
            req.output_token_ids=closed(p,[32]*THINKING_CAP,[65]*30)
            self.assertTrue(stop(req,WINDOW));self.assertEqual(req.status,'cap')
        req=NS(sampling_params=NS(extra_args={KEY:p}),output_token_ids=closed(p,answer=[3]))
        self.assertTrue(stop(req,WINDOW));self.assertEqual(req.status,'eos')
        probe=NS(sampling_params=NS(extra_args={}),output_token_ids=[65])
        logits=torch.zeros((1,256))
        enforce_logits(NS(requests={'p':probe},input_batch=NS(req_id_to_index={'p':0})),logits)
        self.assertTrue(torch.equal(logits,torch.zeros_like(logits)))
        self.assertEqual(stop(probe,WINDOW),'legacy');original.assert_called_once()

    def test_generate_wire_serialization_latency_and_probe_exclusion(self):
        from runner.runtime import generation
        from vllm import SamplingParams
        from vllm.v1.serial_utils import MsgpackDecoder,MsgpackEncoder
        s=sample();p=s[KEY]
        for answer in ([65,3],[3]):
            class Engine:
                def has_unfinished_requests(self):return getattr(self,'active',False)
                def add_request(self,rid,prompt,params):
                    self.params=params;self.rid=rid;self.active=True;self.tokens=[]
                    self.planned=iter(closed(p,[84,65,82,71,69,84],answer))
                def step(self):
                    self.tokens.append(next(self.planned))
                    self.active=self.tokens[-1]!=3
                    return [NS(request_id=self.rid,outputs=[NS(token_ids=list(self.tokens))],finished=not self.active)]
            engine=Engine()
            out,ttft,_=generation.generate(engine,s['token_ids'],s['max_output_tokens'],'read',True,sample=s)
            wire=MsgpackDecoder(SamplingParams).decode(MsgpackEncoder().encode(engine.params))
            self.assertEqual(wire.extra_args[KEY],p)
            self.assertEqual(wire.temperature,.6);self.assertEqual(wire.top_k,20);self.assertTrue(wire.ignore_eos)
            self.assertEqual(out.ruler_thinking_state.answer,len(answer)-1)
            if len(answer)>1:self.assertGreaterEqual(out.first_answer_content_seconds,ttft)
            else:self.assertIsNone(out.first_answer_content_seconds)
        engine=Engine()
        def step():
            engine.active=False
            return [NS(request_id='probe',outputs=[NS(token_ids=[65])],finished=True)]
        engine.step=step
        generation.generate(engine,s['token_ids'],1,'probe',False,sample=s)
        self.assertNotIn(KEY,engine.params.extra_args);self.assertEqual(engine.params.max_tokens,1)

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

    def test_profile_full_capacity_equal_ranks_and_legacy(self):
        from runner.router.profiles import config,validate,validate_initialization
        validate(sample(),PROFILE)
        for cached in (False,True):
            cfg=config('/m',PROFILE,cached,'/c',hardware='l20-tp2',tp=2)
            self.assertEqual(cfg['max_model_len'],82304);self.assertEqual(cfg['gpu_memory_utilization'],.95)
            self.assertNotIn('num_gpu_blocks_override',cfg)
            self.assertEqual(cfg['rope_scaling']['factor'],4.)
        receipts=[dict(rank=r,kv_tokens=82304,kv_blocks=1286,block_size=64,
            rope=[dict(formula_bitwise_equal=True,table_positions=131072)]*64) for r in range(2)]
        validate_initialization(receipts,PROFILE,'l20-tp2',2)
        for blocks in (1285,1287):
            bad=copy.deepcopy(receipts);bad[1]['kv_blocks']=blocks
            with self.assertRaises(ValueError):validate_initialization(bad,PROFILE,'l20-tp2',2)
        self.assertEqual(config('/m','ruler',False,'/c',tp=2)['max_model_len'],65920)
        self.assertEqual(config('/m','longbench-v2',False,'/c',tp=2)['max_model_len'],131072)

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

    def test_real_scheduler_request_and_idempotent_hook_installation(self):
        from vllm import SamplingParams
        from vllm.v1.request import Request,RequestStatus
        from vllm.v1.core.sched import scheduler,utils
        from runner.runtime.thinking_budget import install_scheduler_hook
        p=make_policy(Tokenizer(),' Answer:',30)
        prior_utils,prior_scheduler=utils.check_stop,scheduler.check_stop
        try:
            install_scheduler_hook();hook=utils.check_stop
            install_scheduler_hook();self.assertIs(utils.check_stop,hook);self.assertIs(scheduler.check_stop,hook)
            for tail,status in (([3],RequestStatus.FINISHED_STOPPED),([65]*30,RequestStatus.FINISHED_LENGTH_CAPPED)):
                params=SamplingParams(max_tokens=p['output_reserve'],ignore_eos=True,extra_args={KEY:p})
                req=Request('thinking',[65]*100,None,None,None,params,None,3)
                ids=closed(p,[32],tail)
                for i,token in enumerate(ids):
                    req.append_output_token_ids(token)
                    self.assertEqual(scheduler.check_stop(req,WINDOW),i==len(ids)-1)
                self.assertEqual(req.status,status)
        finally:
            utils.check_stop=prior_utils;scheduler.check_stop=prior_scheduler

    def test_admission_checks_expanded_window_and_phase_overflow(self):
        from runner.runtime.thinking_budget import validate_sample_policy
        from runner.operations import check_hardware
        from runner.cache.setups import atomic_json
        s=sample();s['token_ids']=[65]*(WINDOW-s['max_output_tokens'])
        validate_sample_policy(s)
        s['token_ids'].append(65)
        with self.assertRaisesRegex(ValueError,'reserve'):validate_sample_policy(s)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic_json(root/'model.safetensors.index.json',dict(metadata=dict(total_size=61.2*2**30)))
            settings=dict(dataset='ruler',execution_profile=PROFILE,hardware_profile='l20-tp2',model=str(root),tp=2,groups=[['a','b']])
            raw='a,NVIDIA L20,49152,49152\nb,NVIDIA L20,49152,49152'
            with patch('runner.operations.subprocess.check_output',side_effect=[raw,'']):
                receipt=check_hardware(settings)
            self.assertAlmostEqual(receipt['required_mib_per_rank'],(30.6*2**30+82304*262144/2)/.95/2**20)
            self.assertAlmostEqual(receipt['estimated_required_mib_per_rank'],(30.6*2**30+82304*262144/2+3*2**30)/.95/2**20)
            self.assertEqual(receipt['workspace_validation'],'runtime-kv-capacity-audit')
