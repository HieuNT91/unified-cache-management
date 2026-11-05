"""8192 thinking -> answer, 10000 total, and sequential connector-free baseline."""
import copy
import json
import sys
from types import SimpleNamespace as NS
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from runner.router.adatree import SCHEMA, answer_generation, generation_policy, generation_sample
from runner.runtime import output_budget
from runner.runtime.thinking_budget import KEY, BudgetState, validate_thinking_outcome
from runner.runtime.thinking_loop import ADATREE_CONFIG, ThinkingLoop
from runner.reporting.aggregate import OutputAnalyzer
from runner.cache.setups import atomic_json, file_hash
from runner.corpus.runtime import Engine
from scripts.collection import tree_prophetkv_data as collection
from test_tree_collection import fixture
from test_tree_chunk_size import enlarged
from test_ruler_thinking import Tokenizer


class AdaTreeGenerationTests(unittest.TestCase):
    def protocol(self):
        return dict(adaptive_routing=SCHEMA, answer_generation=answer_generation())

    def policy(self):
        return generation_policy(output_budget.make_policy(Tokenizer()), self.protocol())

    def validate(self, state, tokens):
        result=OutputAnalyzer(Tokenizer()).analyze(tokens,[],True)
        result.update(state.metrics(Tokenizer()),ttft_seconds=.1,
                      first_answer_content_seconds=.2 if state.answer else None)
        result['output_cap_reached']=result['answer_cap_reached']
        validate_thinking_outcome(result,state.policy)
        return result

    def test_cap_closes_thinking_then_answer_stops_at_total_10000(self):
        p=self.policy()
        for opening in (False,True):
            with self.subTest(opening=opening):
                policy=dict(p,opening_in_prompt=opening)
                tokens=([] if opening else [policy['open_id']])+list(range(100,8292))
                state=BudgetState(policy,ADATREE_CONFIG).consume(tokens)
                self.assertEqual(state.thinking,8192)
                self.assertEqual(state.phase,'thinking')
                self.assertEqual(state.forced_next(),policy['close_id'])
                tokens += [policy['close_id']]+policy['transition_ids']
                state.consume(tokens)
                self.assertEqual(state.phase,'answer')
                tokens += [65]*(10000-len(tokens))
                state.consume(tokens)
                self.assertEqual(state.phase,'done')
                self.assertFalse(state.loop.broken)
                outcome=self.validate(state,tokens)
                self.assertTrue(outcome['thinking_cap_reached'])
                self.assertTrue(outcome['answer_cap_reached'])
                self.assertEqual(outcome['output_tokens'],10000)
                self.assertGreater(outcome['generated_answer_tokens'],0)
                with self.assertRaises(ValueError): state.consume(tokens+[65])

    def test_new_repetition_requires_four_copies_and_preserves_answer(self):
        p=self.policy()
        for size in (32,128):
            block=list(range(100,100+size)); tokens=block*3+block[:-1]
            state=BudgetState(p,ADATREE_CONFIG).consume(tokens)
            self.assertFalse(state.loop.broken)
            tokens += [block[-1]];state.consume(tokens)
            self.assertEqual(state.loop.pattern_size,size)
            self.assertEqual(state.forced_next(),p['close_id'])
            tokens += [p['close_id']]+p['transition_ids']+[65]*600+[3]
            state.consume(tokens)
            self.assertEqual(state.phase,'done')
            self.assertEqual(state.answer,600)
            self.validate(state,tokens)
        loop=ThinkingLoop(ADATREE_CONFIG)
        for i in range(1000): loop.observe(100+i,i+1)
        self.assertEqual(len(loop.tail),512)

    def test_generation_view_never_changes_prepared_tokens_layout_or_policy(self):
        original=dict(token_ids=[100,101],boundaries=[0,1,2],question_positions=[1],
                      max_output_tokens=16384, **{KEY:output_budget.make_policy(Tokenizer())})
        before=copy.deepcopy(original)
        effective=generation_sample(original,self.protocol())
        self.assertEqual(original,before)
        for key in ('token_ids','boundaries','question_positions'):
            self.assertIs(effective[key],original[key])
        self.assertEqual(effective['max_output_tokens'],10000)
        self.assertEqual(effective[KEY]['thinking_cap'],8192)
        self.assertEqual(original[KEY]['thinking_cap'],15000)
        with self.assertRaises(ValueError):
            output_budget.validate_policy(dict(effective[KEY],output_reserve=8192))

    def adopt(self, root):
        args=fixture(root,'longbench-v2')
        args.adatree_v1=True;args.adatree_thinking_8192=True;args.followup_nocache=True
        args.percentages=[5,20,40];args.answer_seeds=[0,1]
        args.baseline_source=None
        receipt=collection.read(args.prepared/'preparation.json')
        for path in (args.prepared/'samples').glob('lb-*.json'):
            atomic_json(path,enlarged(collection.read(path)))
            receipt['files'][str(path.relative_to(args.prepared))]=file_hash(path)
        atomic_json(args.prepared/'preparation.json',receipt)
        with patch('scripts.router.router_control.code_hashes',return_value={}): collection.adopt(args)
        return args

    def test_plan_is_router_then_baseline_and_never_mixes_old_run(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.adopt(Path(temp))
            settings,plan,rows=collection.load(args.root)
            protocol,_=collection.stage(args.root,'tree')
            self.assertEqual(settings['phase_order'],['cached','baseline'])
            self.assertEqual((plan['answers'],plan['baseline_answers'],plan['probes']),(2012,1006,0))
            self.assertEqual(protocol['scheduled_actions'],['adatree-v1','nocache'])
            self.assertEqual(protocol['answer_generation'],answer_generation())
            self.assertNotIn('baseline_receipt',settings)
            self.assertEqual(rows[0][KEY]['thinking_cap'],15000)
            self.assertEqual(generation_sample(collection.read(args.prepared/rows[0]['prepared']),protocol)[KEY]['thinking_cap'],8192)
            args.adatree_thinking_8192=False;args.followup_nocache=False;args.baseline_source=Path(temp)/'old-baseline'
            with self.assertRaisesRegex(ValueError,'parameters differ'): collection.adopt(args)
            for fail in (False,True):
                with patch.object(collection.control,'execute_phase',side_effect=RuntimeError('cached failed') if fail else None) as execute, \
                        patch.object(collection,'report'):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'cached failed'): collection.supervise(args.root)
                    else:
                        collection.supervise(args.root)
                    self.assertEqual([c.args[2] for c in execute.call_args_list],['cached'] if fail else ['cached','baseline'])
            # Capture actual engine configuration before any model/GPU startup.
            def capture(cfg):
                self.assertNotIn('kv_transfer_config',cfg)
                self.assertFalse(cfg.get('enable_prefix_caching'))
                raise RuntimeError('before GPU')
            with patch.dict(sys.modules,{'runner.runtime.worker':NS(setup=None)}), \
                    patch('runner.corpus.runtime.start_engine',side_effect=capture), self.assertRaisesRegex(RuntimeError,'before GPU'):
                Engine(args.root/'tree',protocol,0,'baseline','capture',rows[:1])
            engine=Engine.__new__(Engine);engine.cached=False;engine.protocol=protocol
            with patch('runner.router.adatree.answer',return_value=('record','ds')) as answer:
                self.assertEqual(engine.answer(dict(id='nocache',method='baseline',ratio=None),'nocache',seed=1),('record','ds'))
                answer.assert_called_once_with(engine,1,baseline=True)

    def test_shared_answer_path_passes_new_budget_and_loop_for_both_methods(self):
        from runner.router.adatree import answer
        from runner.inputs.layout import stamp_sample
        sample=stamp_sample(dict(token_ids=list(range(576)),boundaries=[0,64,320,576],
            question_positions=[320],thinking=True,max_output_tokens=16384,
            evaluation_protocol=output_budget.LONGBENCH_PROTOCOL,
            **{KEY:output_budget.make_policy(Tokenizer())}))
        before=copy.deepcopy(sample)
        p=self.policy();tokens=list(range(100,132))*4+[p['close_id']]+p['transition_ids']+[65,3]
        state=BudgetState(p,ADATREE_CONFIG).consume(tokens)
        output=NS(outputs=[NS(token_ids=tokens,text='A',finish_reason='stop')],
                  ruler_thinking_state=state,first_answer_content_seconds=.2,num_cached_tokens=0)
        for baseline in (False,True):
            with self.subTest(baseline=baseline):
                engine=NS(sample=sample,protocol=self.protocol(),tp=2,scheduler='scheduler',
                    row=dict(id='p',subtask='t',references=['A'],scoring='longbench_v2'),
                    analyzer=OutputAnalyzer(Tokenizer()),pc=Mock(),llm=Mock(),rid=lambda suffix:'p:'+suffix)
                engine.common=lambda case,timings,retired:dict(method=case,timings=timings,**{KEY:sample[KEY]})
                def configure(*a,**kw): pass
                def arm(*a,**kw): pass
                def drain(*a,**kw): pass
                engine.llm.collective_rpc.side_effect=lambda fn,**kw: ([{'rank':0},{'rank':1}] if fn==configure else [])
                with patch.dict(sys.modules,{'runner.runtime.worker':NS(configure=configure,arm=arm,drain=drain)}), \
                        patch('runner.corpus.runtime.clean_capture'), \
                        patch('runner.runtime.generation.generate',return_value=(output,.1,1.)) as generate, \
                        patch('runner.runtime.generation.verify_diagnostics') as verify, \
                        patch('runner.router.adatree.validate_routing',return_value={'action':'tree-prophetkv-20'}) as routing, \
                        patch('runner.router.measure.verify_retired',return_value=[]):
                    record,_=answer(engine,1,baseline=baseline)
                self.assertIs(generate.call_args.args[1],sample['token_ids'])
                self.assertEqual(generate.call_args.args[2],10000)
                self.assertEqual(generate.call_args.kwargs['loop_breaker'],ADATREE_CONFIG)
                self.assertEqual(record[KEY],p)
                self.assertEqual(record['prepared_generation_policy'],sample[KEY])
                self.assertEqual(record['thinking_loop_breaker'],ADATREE_CONFIG)
                self.assertEqual(record['max_output_tokens'],10000)
                self.assertEqual(record['thinking_tokens'],128)
                self.assertEqual(record['generated_answer_tokens'],1)
                if baseline:
                    self.assertEqual(record['executed_action'],'nocache')
                    routing.assert_not_called();verify.assert_called_once()
                    self.assertEqual(engine.llm.collective_rpc.call_count,1)
                else:
                    routing.assert_called_once();verify.assert_not_called()
        self.assertEqual(sample,before)

    def test_new_baseline_records_and_report_enforce_same_policy(self):
        from test_baseline_breakdown import fixture as report_fixture, commit
        from scripts.reporting import experiment_task_breakdown as report
        from runner.tree.collection import collection_actions
        from runner.tree.prophetkv import PROTOCOL as TREE_PROTOCOL
        from runner.cache.setups import fingerprint
        from runner.runtime.thinking_loop import CONFIG as OLD_LOOP
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);target,old=report_fixture(base,'longbench-v2')
            commit(old,'nocache',ttft=999.)
            settings=report.read(base/'settings.json')
            settings.update(self.protocol(),followup_nocache=True,phase_order=['cached','baseline'],
                            percentages=[5,20,40],tree_protocol=TREE_PROTOCOL)
            atomic_json(base/'settings.json',settings)
            plan=report.read(base/'plan.json');plan['settings_sha256']=file_hash(base/'settings.json')
            atomic_json(base/'plan.json',plan)
            protocol=dict(target[1],**settings)
            protocol.update(actions=collection_actions(settings),scheduled_actions=['adatree-v1','nocache'],plan_sha256=fingerprint(plan))
            atomic_json(base/'tree/protocol.json',protocol)
            for method in ('adatree-v1','nocache'):
                commit((target[0],protocol,target[2]),method,ttft=2. if method=='adatree-v1' else 4.)
                folder=target[0]/'records'/method/target[2][0]['id']/'seed-0'
                record=report.read(folder/'result.json');original=record[KEY]
                effective=generation_policy(original,protocol)
                state=BudgetState(effective,ADATREE_CONFIG).consume(record['output_token_ids'])
                record.update(state.metrics(Tokenizer()),prepared_generation_policy=original,
                              max_output_tokens=10000,**{KEY:effective})
                if method=='adatree-v1':
                    record.update(executed_action='tree-prophetkv-5',independent_probes=0,
                        routing=dict(schema=SCHEMA,features={'S60_p90':.7,'G5_mean':.3},action='tree-prophetkv-5',
                            ratio=.05,independent_probes=0,feature_mask_only=True,included_in_native_probe=True,request_id='request'))
                atomic_json(folder/'result.json',record)
                receipt=report.read(folder/'validated.json');receipt['files']['result.json']=file_hash(folder/'result.json')
                atomic_json(folder/'validated.json',receipt)
            text,tables,matching=report.report(base)
            self.assertEqual(matching['matched_samples'],1)
            self.assertEqual(tables['overall']['nocache']['mean_ttft_seconds'],4.)
            self.assertIn('same new generation policy',text)
            self.assertNotIn('Baseline keeps its original',text)
            record['thinking_loop_breaker']=OLD_LOOP
            atomic_json(folder/'result.json',record)
            receipt['files']['result.json']=file_hash(folder/'result.json');atomic_json(folder/'validated.json',receipt)
            with self.assertRaisesRegex(ValueError,'loop breaker differs'): report.report(base)

    def test_vllm_wire_scheduler_keeps_answer_after_8192_and_stops_at_10000(self):
        from vllm import SamplingParams
        from vllm.v1.request import Request, RequestStatus
        from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
        from runner.runtime.thinking_loop import KEY as LOOP_KEY
        from runner.runtime.thinking_budget import request_state, stopping_hook
        p=self.policy()
        params=SamplingParams(max_tokens=10000,ignore_eos=True,extra_args={KEY:p,LOOP_KEY:ADATREE_CONFIG})
        wire=MsgpackDecoder(SamplingParams).decode(MsgpackEncoder().encode(params))
        self.assertEqual(wire.max_tokens,10000)
        self.assertEqual(wire.extra_args[LOOP_KEY],ADATREE_CONFIG)
        request=Request('new-budget',[65]*100,None,None,None,wire,None,3)
        stop=stopping_hook(Mock(side_effect=AssertionError('unmanaged request')),RequestStatus)
        tokens=list(range(100,8292))+[p['close_id']]+p['transition_ids']
        tokens += [65]*(10000-len(tokens))
        for i,token in enumerate(tokens):
            request.append_output_token_ids(token)
            self.assertEqual(stop(request,131072),i==9999)
            if i==8191:
                state=request_state(request)
                self.assertEqual(state.phase,'thinking')
                self.assertEqual(state.forced_next(),p['close_id'])
        self.assertEqual(request.status,RequestStatus.FINISHED_LENGTH_CAPPED)
        self.assertEqual(request_state(request).thinking,8192)
        self.assertGreater(request_state(request).answer,0)
