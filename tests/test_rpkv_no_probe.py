"""Fixed controls must answer without a second request or attention archive."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch
from runner.corpus.records import (NATIVE_ANSWER_VALIDATION,PROBE_ANSWER_VALIDATION,
    accepted,publish,answer_validation)
from runner.cache.setups import atomic_json,file_hash

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

    def test_engine_answer_keeps_diagnostics_retirement_and_cache_checks(self):
        from runner.corpus.runtime import Engine
        engine=Engine.__new__(Engine);engine.tp=4;engine.protocol=dict(actions=ACTIONS,answer_validation=NATIVE_ANSWER_VALIDATION)
        engine.sample=dict(token_ids=[1]*384,boundaries=[0,64,128,384],thinking=False,max_output_tokens=128)
        engine.cached=True;engine.pc=Mock();engine.rid=lambda _: 'r';engine.scheduler='scheduler';engine.llm=Mock()
        ds=diagnostics();engine.llm.collective_rpc.return_value=ds;engine.common=Mock(return_value={})
        engine.row=dict(subtask='niah_single_1',scoring='ruler_all',references=['x'])
        engine.analyzer=Mock();engine.analyzer.analyze.return_value=dict(thinking_tokens=0,answer_tokens=1)
        output=NS(outputs=[NS(token_ids=[3],text='x')],num_cached_tokens=64)
        with patch.dict('sys.modules',{'runner.runtime.worker':NS(drain=object(),router_dense_receipt=object())}),patch('runner.runtime.generation.generate',return_value=(output,.2,.3)) as generate,patch('runner.corpus.runtime.clean_capture'),patch('runner.corpus.runtime.arm_tree'),patch('runner.corpus.runtime.verify_retired',return_value=[]) as retire,patch('runner.corpus.runtime.match_answer') as replay:
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

    def test_old_protocols_keep_their_original_validation_identity(self):
        self.assertEqual(answer_validation({}),PROBE_ANSWER_VALIDATION)
        with self.assertRaises(ValueError):answer_validation(dict(answer_validation='unknown'))


if __name__=='__main__':unittest.main()
