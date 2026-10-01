"""Linked provenance, missing-only resume, six-action search/export and lifecycle tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from runner.corpus_extension import LinkedCorpus,check_pins,inventory,derive
from runner.corpus import load_attention
from runner.corpus_records import accepted,publish,record_dir,protocol_identity
from runner.setups import atomic_json,file_hash,fingerprint
from runner.tree_policy import export,load,decide
from runner.router_policy import FEATURES
from runner.extension_train import GRID,train,validate_matrix
from test_corpus import archives,tiny_sample,tree_value

BASE=[dict(id='nocache',method='baseline',ratio=None)]+[dict(id=f'prophetkv-{i}',method='prophetkv',ratio=i/100) for i in (1,20,50)]
ALL,EXTRA=inventory(BASE,[5,10])


class ExtensionTests(unittest.TestCase):
    def test_original_inventory_and_derived_ties(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);receipt=archives(folder,BASE);before={p.name:file_hash(p) for p in folder.iterdir()}
            attention=load_attention(folder,receipt,tiny_sample(),BASE)
            masks=derive(attention,tiny_sample(),ALL)
            self.assertEqual(masks['prophetkv-5'].tolist(),[64,65,66])
            self.assertEqual(masks['prophetkv-10'].tolist(),list(range(64,70)))
            self.assertEqual(before,{p.name:file_hash(p) for p in folder.iterdir()})
            with self.assertRaises(KeyError):load_attention(folder,receipt,tiny_sample(),ALL)
            with self.assertRaises(ValueError):inventory(BASE,[1])

    def test_source_changed_and_missing(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'source';p.write_text('original');pins={str(p):file_hash(p)};check_pins(pins)
            p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):check_pins(pins)
            p.unlink()
            with self.assertRaisesRegex(ValueError,'missing'):check_pins(pins)

    def fixture(self,root):
        base=root/'base';extension=root/'extension';base.mkdir();extension.mkdir()
        row=dict(id='sample',ordinal=0,sha256='a'*64,subtask='task')
        protocol=dict(groups=[['GPU-'+str(i) for i in range(4)]],actions=BASE)
        init=base/'init.json';atomic_json(init,dict(validated=True,engine_config={}))
        record=dict(prompt_id='sample',method='nocache',executed_action='nocache',input_sha256=row['sha256'],group=0,gpu_uuids=protocol['groups'][0],
            cache_immutable=True,retirement=[dict(rank=i,quiescent=True,transfers={'pending':0},request_bookkeeping=0) for i in range(4)],
            initialization='init.json',initialization_sha256=file_hash(init),accuracy=1.,num_cached_tokens=0,timings={'answer_engine_ttft_seconds':1.})
        publish(base,'nocache',row,protocol,record,[])
        class Base:
            def outcome(self,pid,action):return accepted(base,action,row,protocol)
        obj=LinkedCorpus.__new__(LinkedCorpus);obj.root=extension;obj.rows={'sample':row};obj.base=Base();obj.base.root=base;obj.base.actions={a['id']:a for a in BASE}
        obj.actions={a['id']:a for a in ALL};obj.protocol=dict(protocol,actions=ALL,added_actions=EXTRA)
        receipt=record_dir(base,'nocache','sample')/'validated.json';obj.sources={'files':{str(receipt):file_hash(receipt)}}
        return obj,row,record,protocol

    def test_linked_validation_uses_original_protocol_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            obj,row,record,protocol=self.fixture(Path(td))
            self.assertEqual(obj.outcome('sample','nocache'),record)
            with self.assertRaisesRegex(ValueError,'protocol'):accepted(obj.base.root,'nocache',row,obj.protocol)
            with self.assertRaises(FileExistsError):publish(obj.base.root,'nocache',row,protocol,record,[])
            path=record_dir(obj.base.root,'nocache','sample')/'result.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Corrupt'):obj.outcome('sample','nocache')

    def test_resume_only_missing_and_incomplete_matrix(self):
        with tempfile.TemporaryDirectory() as td:
            obj,row,record,protocol=self.fixture(Path(td))
            self.assertEqual(obj.pending(),[row])
            for action in EXTRA:
                r=dict(record,method=action,executed_action=action,derived_mask_provenance={},
                    alignment_audit=[dict(normalized=True,delta_amplitude=1.,aliases_native_table=False)],cache_deletion={'deleted_shards':8})
                atomic_json(obj.root/'init.json',dict(validated=True,engine_config={}))
                publish(obj.root,action,row,obj.protocol,r,[])
                with patch.object(obj,'provenance',return_value={}):
                    self.assertEqual(len(obj.pending()),0 if action==EXTRA[-1] else 1)
            with patch.object(obj,'verify_sources'),patch.object(obj,'inputs',return_value={}),patch.object(obj,'attention',return_value={}):
                # An absent original sparse answer must halt the combined matrix.
                with self.assertRaises(ValueError):obj.matrix()

    def matrix(self):
        return dict(ids=[f'p{i}' for i in range(260)],input_hashes=[fingerprint(i) for i in range(260)],tasks=[f't{i%13}' for i in range(260)],
            features=[dict.fromkeys(FEATURES,(i%20)/20) for i in range(260)],actions=ALL,
            scores=[[1.]*6 for i in range(260)],answer_ttft=[[30.,10.,5.,8.,15.,20.] for i in range(260)],
            probe_ttft=[2.]*260,output_caps=[[False]*6 for i in range(260)],protocol_sha256='c'*64)

    def test_six_action_search_export_and_independent_recompute(self):
        self.assertEqual(len(GRID),2592);self.assertEqual({h['depth'] for h in GRID},{1,2,3})
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);old=tree_value(BASE);old.update(setting={'depth':1},tree=dict(action='prophetkv-1',training_samples=260))
            export(root/'old.json',old);m=self.matrix();out=root/'training'
            train(out,m,root/'old.json',fingerprint(m),grid=[GRID[0],next(h for h in GRID if h['objective']=='minimize-time')])
            best=load(out/'min-ttft.json');self.assertEqual(len(best['actions']),6)
            self.assertEqual(decide(best,m['features'][0])['action'],'prophetkv-5')
            self.assertEqual(decide(best,{})['action'],'nocache')
            report=json.loads((out/'report.json').read_text());self.assertLess(report['policies'][1]['ttft_seconds'],5.001)
            self.assertEqual(report['policies'][1]['actions']['prophetkv-5'],260)
            self.assertEqual(json.loads((out/'independent-validation.json').read_text())['policy_decisions'],780)
            # Completed training is idempotent and immutable.
            train(out,m,root/'old.json',fingerprint(m),grid=[])
            with self.assertRaises(ValueError):train(out,m,root/'old.json','d'*64,grid=[])
            m['scores'].pop()
            with self.assertRaises(ValueError):validate_matrix(m)

    def test_previous_candidate_cannot_regress(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);m=self.matrix();old=tree_value(BASE)
            old.update(setting={'depth':1},tree=dict(action='prophetkv-1',training_samples=260));export(root/'old.json',old)
            train(root/'training',m,root/'old.json',fingerprint(m),grid=[])
            result=json.loads((root/'training/selection.json').read_text())
            self.assertEqual(result['min_ttft']['index'],-1)

    def test_export_real_splits_and_threshold_boundaries(self):
        from runner.extension_train import check_export,compile_node
        raw=dict(n=260,feature=0,threshold=.25,left=dict(n=60,action=2),
            right=dict(n=200,feature=1,threshold=.75,left=dict(n=140,action=3),right=dict(n=60,action=0)))
        names=[a['id'] for a in ALL]
        with tempfile.TemporaryDirectory() as td:
            data=tree_value(ALL);data['tree']=compile_node(raw,names)
            tree=export(Path(td)/'tree.json',data)
            self.assertGreater(check_export(tree,raw,names,self.matrix()['features']),280)

    def test_worker_reuses_one_cache_and_runs_only_missing_actions(self):
        from scripts.corpus_extend import collect
        from types import SimpleNamespace
        events=[]
        row=dict(id='second',ordinal=1)
        class Fake:
            protocol=dict(groups=[['gpu']*4],added_actions=EXTRA,prompt_ids=['first','second'],actions=ALL)
            actions={a['id']:a for a in ALL}
            def pending(self):return [row]
            def attention(self,pid,replay=False):
                events.append(('replay',replay));return 'saved-attention'
            def provenance(self,pid):return {'saved':True}
            def outcome(self,pid,action):return {} if action=='prophetkv-5' else None
        class Engine:
            def __init__(self,*args):events.append('engine')
            def begin(self,r):events.append(('begin',r['id']))
            def answer(self,definition,case,attention):
                events.append((case,attention));return {},[]
            def end(self):events.append('delete');return {'deleted_shards':8}
            def close(self):events.append('close')
        with tempfile.TemporaryDirectory() as td:
            with patch('scripts.corpus_extend.verify',return_value=Fake()),patch('scripts.corpus_extend.check_environment',return_value=['gpu']*4),patch('runner.corpus_runtime.Engine',Engine),patch('scripts.corpus_extend.publish') as published:
                collect(Path(td),'test')
                self.assertEqual(published.call_count,1)
                self.assertEqual(published.call_args.args[1],'prophetkv-10')
        self.assertEqual(events, ['engine',('replay',True),('begin','second'),('prophetkv-10','saved-attention'),'delete','close'])

    def test_finalization_requires_owned_engine_exit(self):
        from scripts.corpus_extend import finalize
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);atomic_json(root/'processes/worker.json',dict(pid=123,identity={'fake':True}))
            with patch('scripts.corpus_extend.verify'),patch('scripts.corpus_extend.alive',return_value=True):
                with self.assertRaisesRegex(ValueError,'engines'):finalize(root)

if __name__=='__main__':unittest.main()
