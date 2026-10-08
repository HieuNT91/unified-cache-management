"""CPU-only phase budgets, native preparation, TP2 handoff and portable training."""
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

from runner.thinking_budget import (KEY, PROFILE, PROTOCOL, THINKING_CAP, WINDOW,
    BudgetState, make_policy, enforce_logits, stopping_hook, request_state)
from scripts.ruler import TASKS, CAPS, format_thinking_sample, generator_command

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
        from runner import generation
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
        with patch('scripts.ruler.definitions',return_value=(defs,const)):
            thinking=generator_command(args,'qa_1',Path('/raw'),Tokenizer())
            args.thinking=False
            self.assertEqual(thinking,generator_command(args,'qa_1',Path('/raw'),Tokenizer()))
        for flag,value in [('--num_samples','30'),('--max_seq_length','65536'),('--tokens_to_generate','32'),('--random_seed','42')]:
            self.assertEqual(thinking[thinking.index(flag)+1],value)

    def test_profile_full_capacity_equal_ranks_and_legacy(self):
        from runner.tree_profiles import config,validate,validate_initialization
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
        from runner.reporting import OutputAnalyzer,scoring_text,score_answer,metrics
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

    def test_thinking_configure_prepare_scope_shards_and_pinned_policies(self):
        from scripts import longbench_a800_control as control, longbench_a800_data as data
        from runner.setups import atomic_json,file_hash
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prepared=root/'inputs'
            args=NS(root=root,role='ruler',model=Path('/m'),data=Path('/ruler'),prepared=prepared,
                    cache_root=root/'cache',tp=2,samples_per_task=30,seed=42,devices='unused',thinking=True)
            groups=[[str(i),str(i+1)] for i in range(0,10,2)]
            with patch.object(control,'resolve_groups',return_value=groups),patch.object(control,'code_hashes',return_value={}):
                control.configure(args)
                pins=file_hash(root/'settings.json')
                args.thinking=False;args.samples_per_task=100
                with self.assertRaisesRegex(ValueError,'Frozen metadata'):control.configure(args)
                self.assertEqual(pins,file_hash(root/'settings.json'))
            entries=[];files={}
            for task in TASKS:
                # Full adapter tested above; synthetic tiny prepared JSON avoids a large fixture.
                s=sample(task);s['token_ids']=[65]
                for i in range(30):
                    name=f'samples/{task}-{i}.json';atomic_json(prepared/name,s);files[name]=file_hash(prepared/name)
                    entries.append(dict(id=f'{task}-{i}',prepared=name,subtask=task))
            manifest=prepared/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in entries))
            files['manifest.jsonl']=file_hash(manifest);atomic_json(prepared/'preparation.json',dict(files=files))
            with patch('scripts.ruler.prepare') as prepare:
                plan=data.prepare(root)
            config=prepare.call_args.args[0]
            self.assertTrue(config.thinking);self.assertEqual((config.samples,config.seed),(30,42))
            self.assertEqual((plan['samples'],plan['answers'],plan['probes']),(390,4680,390))
            protocol,rows=data.stage(root,'ruler')
            self.assertEqual(protocol['execution_profile'],PROFILE)
            self.assertEqual(data.stage(root,'features')[0]['execution_profile'],PROFILE)
            for group in range(5):self.assertEqual(sum(r['ordinal']%5==group for r in rows),78)
            self.assertEqual(data.roles(data.read(root/'settings.json')),('ruler','features'))
            bad=copy.deepcopy(rows);bad[0]['evaluation_protocol']='legacy'
            with self.assertRaises(ValueError):data.validate_rows(data.read(root/'settings.json'),bad)

    def test_relocated_launcher_env_precedence_all_commands_and_fresh_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'checkout with spaces';launchers=root/'scripts/launcher'
            shutil.copytree(ROOT/'scripts/launcher',launchers)
            fake=Path(tmp)/'fake python'
            fake.write_text(f'#!{sys.executable}\nimport json,os,sys\nprint(json.dumps([sys.argv[1:],os.environ.get("CUDA_VISIBLE_DEVICES")]))\n')
            fake.chmod(0o755)
            (root/'.env.l20.thinking').write_text(f'PYTHON_BIN="{fake}"\nMODEL_PATH="/model from file"\n')
            (root/'.env.l20').write_text('PYTHON_BIN=/must-not-run\n')
            for command in ('configure','prepare','detach','resume','stop','status','status_same_count','report'):
                result=subprocess.run(['bash',str(launchers/'l20_ruler_thinking_data.sh'),command],cwd='/tmp',
                    env={'PATH':os.environ['PATH'],'MODEL_PATH':'/override'},capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                argv,cuda=json.loads(result.stdout)
                self.assertEqual(cuda,'');self.assertEqual(argv[1],command);self.assertIn('--thinking',argv)
                self.assertEqual(argv[argv.index('--model')+1],'/override')
                self.assertEqual(argv[argv.index('--samples-per-task')+1],'30')
                for flag in ('--root','--prepared','--cache-root'):
                    self.assertIn('thinking-30',argv[argv.index(flag)+1])

    def test_synthetic390_export_import_training_and_provenance(self):
        from scripts import router_dataset as portable,train_router as training
        from scripts.longbench_a800_data import ACTIONS
        from runner.setups import fingerprint
        from runner.tree_policy import load as load_tree
        from test_portable_training import synthetic_rows,SMALL_GRID,bundle
        rows=synthetic_rows('ruler',390)
        for row in rows:
            p=make_policy(Tokenizer(),' Answer:',CAPS[row['subtask']]);row[KEY]=p;row['evaluation_protocol']=PROTOCOL
            forced=len(p['transition_ids'])+len(p['prefix_ids'])
            for value in row['outcomes'].values():
                value.update(thinking_tokens=4,answer_tokens=forced+2,control_tokens=2,output_tokens=forced+8,
                    generated_answer_tokens=2,generated_thinking_open_tokens=0,forced_tokens=forced,thinking_cap_reached=False,
                    answer_cap_reached=False,thinking_closure='natural',first_answer_content_seconds=value['ttft_seconds']+1)
        provenance=dict(tp=2,seed=42,samples_per_task=30,execution_profile=PROFILE,evaluation_protocol=PROTOCOL)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'ruler-data.json'
            value=portable.save(path,'ruler',ACTIONS,rows,provenance)
            self.assertEqual(len(portable.load(path)['rows']),390)
            for mutate in (lambda d:d['provenance'].pop('execution_profile'),
                           lambda d:d['rows'][0].update(evaluation_protocol='legacy'),
                           lambda d:d['rows'][0]['outcomes']['nocache'].update(forced_tokens=0),
                           lambda d:d['rows'][0][KEY].update(answer_cap=999)):
                bad=copy.deepcopy(value);mutate(bad);bad['payload_sha256']=fingerprint({k:v for k,v in bad.items() if k!='payload_sha256'})
                with self.assertRaises(ValueError):portable.validate(bad)
            selected,inventory,meta=training.select_data('ruler',path)
            self.assertEqual(meta['dataset_provenance']['ruler'],provenance)
            with patch('runner.corpus_train.GRID',SMALL_GRID),patch.object(training,'GRID',SMALL_GRID):
                report=training.train(selected,inventory,meta,root/'synthetic-fit')
            self.assertEqual(report['samples'],390);self.assertEqual(len(report['policies']),3)
            tree=load_tree(root/'synthetic-fit/router1.json')
            self.assertEqual(tree['compatible_evaluation_protocols'],[PROTOCOL])
            self.assertEqual(len(training.select_data('both',path,bundle(root,'longbench-v2'))[0]),893)

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
        from runner.thinking_budget import install_scheduler_hook
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
        from runner.thinking_budget import validate_sample_policy
        from scripts.corpus_control import check_hardware
        from runner.setups import atomic_json
        s=sample();s['token_ids']=[65]*(WINDOW-s['max_output_tokens'])
        validate_sample_policy(s)
        s['token_ids'].append(65)
        with self.assertRaisesRegex(ValueError,'reserve'):validate_sample_policy(s)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic_json(root/'model.safetensors.index.json',dict(metadata=dict(total_size=61.2*2**30)))
            settings=dict(dataset='ruler',execution_profile=PROFILE,hardware_profile='l20-tp2',model=str(root),tp=2,groups=[['a','b']])
            raw='a,NVIDIA L20,49152,49152\nb,NVIDIA L20,49152,49152'
            with patch('scripts.corpus_control.subprocess.check_output',side_effect=[raw,'']):
                receipt=check_hardware(settings)
            self.assertAlmostEqual(receipt['required_mib_per_rank'],(30.6*2**30+82304*262144/2+3*2**30)/.95/2**20)

    def test_live_report_and_portable_example_preserve_phase_metrics(self):
        from scripts.longbench_a800_report import report
        from scripts.longbench_a800_data import ACTIONS
        from scripts.router_dataset import example
        from runner.reporting import OutputAnalyzer
        tok=Tokenizer();s=sample();p=s[KEY]
        ids=closed(p,tok.encode('TARGET'),[3])
        state=BudgetState(p).consume(ids)
        metrics=OutputAnalyzer(tok).analyze(ids,[],True);metrics.update(state.metrics(tok))
        row=dict(id='qa_1-0',subtask='qa_1',sha256='0'*64,evaluation_protocol=PROTOCOL,**{KEY:p})
        record=dict(metrics,prompt_id=row['id'],subtask='qa_1',prediction='TARGET',scoring='ruler_any',
            accuracy=0.,output_cap_reached=False,gpu_uuids=['gpu-a','gpu-b'],
            timings=dict(ttft_seconds=1.,first_answer_content_seconds=None))
        probe=dict(features={},timings=dict(routing_overhead_seconds=.1))
        exported=example(row,probe,{a['id']:record for a in ACTIONS},{'r':'0'*64},'ruler')
        self.assertEqual(exported[KEY],p)
        self.assertIsNone(exported['outcomes']['nocache']['first_answer_content_seconds'])
        settings=dict(dataset='ruler',execution_profile=PROFILE,tp=2)
        plan=dict(answers=4680,probes=390)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            with patch('scripts.longbench_a800_report.load',return_value=(settings,plan,[row])), \
                 patch('scripts.longbench_a800_report.records',return_value={a['id']:[record] for a in ACTIONS}):
                value=report(base,emit=False)
            overall=value['methods']['nocache']['overall']
            self.assertEqual(overall['ruler_nulls'],'1/1')
            self.assertEqual(overall['natural_thinking_closures'],1)
            self.assertIsNone(overall['mean_first_answer_content_seconds'])
            self.assertIn('First answer content s',(base/'status.md').read_text())
            self.assertIn('thinking_cap_reached',(base/'status.csv').read_text())
