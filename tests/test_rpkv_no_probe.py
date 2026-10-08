"""Fixed controls must answer without a second request or attention archive."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch
from runner.corpus_records import (NATIVE_ANSWER_VALIDATION,PROBE_ANSWER_VALIDATION,
    accepted,publish,answer_validation)
from runner.setups import atomic_json,file_hash

ACTIONS=[dict(id='nocache',method='baseline',ratio=None),dict(id='prophetkv-1',method='prophetkv',ratio=.01),dict(id='prophetkv-5',method='prophetkv',ratio=.05)]


def diagnostics():
    selected=[64,65,66];positions=selected+list(range(128,384));count=len(positions)
    events=[dict(kind='prophetkv_selection',scores=[1.]*64,selected_positions=selected,
        scoring_layers=list(range(64)),fusion='mean_layers_fp32',alignment_count=64),
        dict(kind='prefill_step',start=64,end=384,scheduled_tokens=320,recomputed_tokens=count,no_forward=False,prefill_complete=True)]
    events += [dict(kind='layer_counts',layer=f'model.layers.{i}.self_attn.attn',start=64,end=384,
        selected_positions=positions,selected_set_verified=True,projection_tokens=count,attention_tokens=count,ffn_tokens=count) for i in range(64)]
    return [dict(rank=r,diagnostics=copy.deepcopy(events)) for r in range(4)]


class NoProbeTests(unittest.TestCase):
    def test_worker_resumes_missing_answers_without_probe(self):
        from scripts.rpkv_gpu_validation import worker
        protocol=dict(dataset='ruler',actions=ACTIONS,answer_validation=NATIVE_ANSWER_VALIDATION)
        row=dict(id='p');engine=Mock()
        engine.answer.return_value=(dict(thinking_tokens=0),[]);engine.end.return_value={}
        engine.probe.side_effect=AssertionError('Unexpected probe request')
        def existing(root,case,row,protocol):
            if case=='probe':raise AssertionError('Unexpected probe lookup')
            return {} if case=='prophetkv-1' else None
        with tempfile.TemporaryDirectory() as tmp,patch('scripts.rpkv_gpu_validation.load_run',return_value=(protocol,[row])),patch('runner.setups.check_environment'),patch('runner.corpus_runtime.Engine',return_value=engine),patch('scripts.rpkv_gpu_validation.accepted',side_effect=existing),patch('scripts.rpkv_gpu_validation.publish') as pub,patch('scripts.rpkv_gpu_validation.report'):
            worker(Path(tmp),'cached','attempt')
        engine.probe.assert_not_called();engine.answer.assert_called_once_with(ACTIONS[2],'prophetkv-5')
        engine.begin.assert_called_once_with(row);engine.end.assert_called_once();engine.close.assert_called_once();pub.assert_called_once()

    def test_engine_answer_keeps_diagnostics_retirement_and_cache_checks(self):
        from runner.corpus_runtime import Engine
        engine=Engine.__new__(Engine);engine.tp=4;engine.protocol=dict(actions=ACTIONS,answer_validation=NATIVE_ANSWER_VALIDATION)
        engine.sample=dict(token_ids=[1]*384,boundaries=[0,64,128,384],thinking=False,max_output_tokens=128)
        engine.cached=True;engine.pc=Mock();engine.rid=lambda _: 'r';engine.scheduler='scheduler';engine.llm=Mock()
        ds=diagnostics();engine.llm.collective_rpc.return_value=ds;engine.common=Mock(return_value={})
        engine.row=dict(subtask='niah_single_1',scoring='ruler_all',references=['x'])
        engine.analyzer=Mock();engine.analyzer.analyze.return_value=dict(thinking_tokens=0,answer_tokens=1)
        output=NS(outputs=[NS(token_ids=[3],text='x')],num_cached_tokens=64)
        with patch.dict('sys.modules',{'runner.worker':NS(drain=object(),router_dense_receipt=object())}),patch('runner.generation.generate',return_value=(output,.2,.3)) as generate,patch('runner.corpus_runtime.clean_capture'),patch('runner.corpus_runtime.arm_tree'),patch('runner.corpus_runtime.verify_retired',return_value=[]) as retire,patch('runner.corpus_runtime.match_answer') as replay:
            record,_=engine.answer(ACTIONS[2],'prophetkv-5')
            self.assertEqual(record['answer_validation'],NATIVE_ANSWER_VALIDATION)
            self.assertEqual(record['accuracy'],1);generate.assert_called_once();retire.assert_called_once();engine.pc.unchanged.assert_called_once();replay.assert_not_called()
            broken=copy.deepcopy(ds);broken[2]['diagnostics'][0]['scores'][0]=2.
            engine.llm.collective_rpc.return_value=broken
            with self.assertRaisesRegex(RuntimeError,'TP ranks disagree'):engine.answer(ACTIONS[2],'prophetkv-5')
            with self.assertRaisesRegex(ValueError,'cannot consume'):engine.answer(ACTIONS[2],'prophetkv-5',attention={})
        with self.assertRaisesRegex(ValueError,'disabled'):engine.probe(Path('/unused'))

    def test_full_receipt_validation_without_probe_still_rejects_corrupt_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prepared=root/'inputs';sample=dict(boundaries=[0,64,128,384]);atomic_json(prepared/'p.json',sample)
            row=dict(id='p',ordinal=0,prepared='p.json',sha256=file_hash(prepared/'p.json'))
            protocol=dict(actions=ACTIONS,prepared=str(prepared),groups=[['GPU-'+str(i) for i in range(4)]],answer_validation=NATIVE_ANSWER_VALIDATION)
            atomic_json(root/'init.json',dict(validated=True,engine_config={}))
            record=dict(prompt_id='p',method='prophetkv-5',executed_action='prophetkv-5',input_sha256=row['sha256'],group=0,
                gpu_uuids=protocol['groups'][0],cache_immutable=True,answer_validation=NATIVE_ANSWER_VALIDATION,
                retirement=[dict(rank=r,quiescent=True,transfers=dict(pending=0),request_bookkeeping=0) for r in range(4)],
                initialization='init.json',initialization_sha256=file_hash(root/'init.json'),accuracy=1.,timings=dict(ttft_seconds=.2))
            publish(root,'prophetkv-5',row,protocol,record,diagnostics())
            self.assertEqual(accepted(root,'prophetkv-5',row,protocol,full=True),record)
            self.assertFalse((root/'records/probe').exists())
            with self.assertRaisesRegex(ValueError,'do not use'):accepted(root,'probe',row,protocol)
            # A repinned bad native mask still fails semantic validation, not just file hashes.
            folder=root/'records/prophetkv-5/p';ds=diagnostics();ds[0]['diagnostics'][0]['selected_positions']=[65,66,67]
            atomic_json(folder/'diagnostics.json',ds);receipt=json.loads((folder/'validated.json').read_text());receipt['files']['diagnostics.json']=file_hash(folder/'diagnostics.json');atomic_json(folder/'validated.json',receipt)
            with self.assertRaisesRegex(RuntimeError,'Selection replay'):accepted(root,'prophetkv-5',row,protocol,full=True)

    def test_final_live_report_requires_answers_but_no_probes(self):
        from scripts.rpkv_gpu_validation import report
        protocol=dict(dataset='ruler',actions=ACTIONS,answer_validation=NATIVE_ANSWER_VALIDATION);rows=[dict(id='p',subtask='niah_single_1')]
        def outcome(root,case,row,protocol):
            if case=='probe':raise AssertionError('Unexpected probe lookup')
            return dict(subtask=row['subtask'],scoring='ruler_all',prediction='x',accuracy=1.,thinking_tokens=0,answer_tokens=1,control_tokens=0,output_tokens=1,unfinished_thinking=False,output_cap_reached=False,timings=dict(ttft_seconds=.2))
        with tempfile.TemporaryDirectory() as tmp,patch('scripts.rpkv_gpu_validation.load_run',return_value=(protocol,rows)),patch('scripts.rpkv_gpu_validation.accepted',side_effect=outcome):
            value=report(Path(tmp),True);self.assertEqual(value['probes'],0)
            self.assertTrue(value['final']);self.assertEqual(len(value['actions']),3)
            with patch('scripts.rpkv_gpu_validation.accepted',return_value=None):
                with self.assertRaisesRegex(ValueError,'incomplete answers'):report(Path(tmp),True)

    def test_old_protocols_keep_their_original_validation_identity(self):
        self.assertEqual(answer_validation({}),PROBE_ANSWER_VALIDATION)
        with self.assertRaises(ValueError):answer_validation(dict(answer_validation='unknown'))

    def test_final_report_uses_committed_results_without_model_or_replay(self):
        from scripts.rpkv_gpu_results import report
        from scripts.ruler import TASKS
        from runner.layout import token_hash
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);ruler=base/'upstream';(ruler/'scripts/eval/synthetic').mkdir(parents=True)
            (ruler/'scripts/eval/evaluate.py').write_text('def postprocess_pred(text, task):\n    return text.strip()\n')
            (ruler/'scripts/eval/synthetic/constants.py').write_text("TASKS={'fake': {'metric_fn': lambda predictions, references: 100.0}}\n")
            atomic_json(base/'experiment.json',dict(ruler_samples_per_task=1,longbench_samples=1,seed=42))
            for dataset,tasks in [('ruler',TASKS),('longbench-v2',['qa'])]:
                root=base/'results'/dataset;prepared=base/'prepared'/dataset
                sample=dict(token_ids=[1,2,3],token_sha256=token_hash([1,2,3]),thinking=False,max_output_tokens=128 if dataset=='ruler' else 16384,truncation=dict(truncated=False))
                atomic_json(prepared/'p.json',sample)
                rows=[dict(id=t,subtask=t,prepared='p.json') for t in tasks]
                atomic_json(root/'rows.json',rows);atomic_json(root/'complete.json',dict(answers=len(rows)*3,probes=0))
                atomic_json(root/'protocol.json',dict(model='fake',prepared=str(prepared),actions=ACTIONS,answer_validation=NATIVE_ANSWER_VALIDATION))
            analysis=dict(answer_text='The correct answer is (A)',thinking_tokens=0,answer_tokens=1,control_tokens=0,output_tokens=1,unfinished_thinking=False)
            def outcome(root,case,row,protocol,full=False):
                self.assertFalse(full)
                if case=='probe':raise AssertionError('Unexpected independent probe')
                cap=128 if root.name=='ruler' else 16384
                return dict(subtask=row['subtask'],output_token_ids=[9],prediction='The correct answer is (A)',references=['A'],scoring='ruler_all' if root.name=='ruler' else 'longbench_v2',
                    token_sha256=token_hash([1,2,3]),prompt_tokens=3,max_output_tokens=cap,accuracy=1.,output_cap_reached=False,
                    timings=dict(ttft_seconds=.2,answer_engine_ttft_seconds=.19),**analysis)
            with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':''}),patch('transformers.AutoTokenizer.from_pretrained',side_effect=AssertionError('Reporter must not load a model')),patch('scripts.rpkv_gpu_results.accepted',side_effect=outcome),patch('runner.generation.verify_diagnostics',side_effect=AssertionError('Reporter must not replay diagnostics')),patch('builtins.print'):
                report(base)
            receipt=json.loads((base/'final/completion.json').read_text())
            self.assertEqual((receipt['answers'],receipt['probes']),(42,0))
            self.assertTrue(receipt['complete']);self.assertFalse(receipt['audits']['ruler']['diagnostic_replay'])
            self.assertFalse(receipt['audits']['ruler']['independent_scoring_replay'])
            self.assertFalse((base/'final/validation.json').exists())
            self.assertIn('No independent diagnostic probes were run',(base/'final/report.md').read_text())


if __name__=='__main__':unittest.main()
