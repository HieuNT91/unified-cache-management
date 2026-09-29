"""Native router CPU acceptance: policies, arithmetic, lifecycle, relocation and full reports."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock,patch
import numpy as np

from runner.router_policy import (ACTIONS,FEATURES,DEFINITIONS,SCHEMA,digest,validate,decide,
                                  features_from_arrays,load_policy)
from runner.router_export import compile_policy,source_action
from runner.setups import atomic_json,file_hash,fingerprint
from runner.router_sweep import CASES,paired_check,accepted,record_dir
NS=types.SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]


def source_policy(rank=1,family='loss'):
    leaf=lambda loss,action:dict(loss=loss,action=action,n=30)
    return dict(version=1,rank=rank,primary=rank==1,family=family,feature_names=list(FEATURES),actions=list(ACTIONS),
        costs=[1.,2.,3.,10.],hyperparameters={'threshold':.02},
        tree=dict(n=195,loss=[.1,.05,.01],action=2,feature=0,threshold=.25,
            left=leaf([0.,.01,0.],0),right=dict(n=90,loss=[.4,.3,.2],action=3,feature=3,threshold=.75,
                left=leaf([.2,.02,0.],1),right=leaf([1.,1.,1.],3))))


def payload(action=None):
    rows=[compile_policy(source_policy(i,'cost' if i==2 else 'loss')) for i in (1,2,3)]
    if action is not None:
        for r in rows:r['tree']=dict(action=action,training_samples=195)
    data=dict(schema=SCHEMA,primary='router1',feature_names=list(FEATURES),feature_definitions=copy.deepcopy(DEFINITIONS),
        actions=list(ACTIONS),policies=rows,provenance=dict(training_protocol_sha256='a'*64,policy_lock_sha256='b'*64,cohort_sha256='c'*64))
    data['payload_sha256']=digest(data)
    return data


def write_policy(root,action=None):
    path=Path(root)/'trees.json';atomic_json(path,payload(action))
    Path(str(path)+'.sha256').write_text(file_hash(path)+'  '+path.name+'\n')
    return path


def sample():
    ids=[1]*384;ids[63]=ids[127]=99
    return dict(token_ids=ids,boundaries=[0,64,128,384],question_positions=[380,381],thinking=False,max_output_tokens=256)


class PolicyTests(unittest.TestCase):
    def test_exact_tree_paths_threshold_boundaries_all_families(self):
        data=payload()
        for rank in (1,2,3):
            p=source_policy(rank,'cost' if rank==2 else 'loss')
            for x in (.1,math.nextafter(.25,-math.inf),.25,math.nextafter(.25,math.inf),.9):
                for y in (.5,math.nextafter(.75,-math.inf),.75,math.nextafter(.75,math.inf),.9):
                    f=dict.fromkeys(FEATURES,.5);f.update(top1_mass=x,group_agreement=y)
                    self.assertEqual(decide(data,f'router{rank}',f)['action'],source_action(p,f))
        for value in (None,float('nan'),float('inf'),True,'0.1'):
            f=dict.fromkeys(FEATURES,.5);f['top1_mass']=value
            self.assertEqual(decide(data,'router1',f)['fallback_reason'],'missing_required_feature')
        self.assertEqual(decide(data,'router1',{})['action'],'baseline')

    def test_malformed_policy_is_never_dense_fallback(self):
        for mutation in (lambda d:d.update(primary='router2'),lambda d:d['feature_definitions'].update(version=2),
                         lambda d:d['policies'][0]['tree'].update(threshold=float('nan')),
                         lambda d:d['policies'][0].update(training_costs=[1.,-1.,2.,3.]),
                         lambda d:d['policies'].pop(),lambda d:d['policies'][0]['tree']['le'].update(action='invented')):
            data=payload();mutation(data)
            try:data['payload_sha256']=digest({k:v for k,v in data.items() if k!='payload_sha256'})
            except ValueError:pass
            with self.assertRaises(ValueError):decide(data,'router1',{})
        with tempfile.TemporaryDirectory() as td:
            path=write_policy(td);self.assertEqual(load_policy(path)['primary'],'router1')
            path.write_text(path.read_text()+' ')
            with self.assertRaises(ValueError):load_policy(path)

    def test_profile_cli_and_manual_overrides(self):
        from runner.config import engine_config
        with tempfile.TemporaryDirectory() as td:
            policy=write_policy(td)
            cfg=engine_config('/model','router',cache_dir='/cache',end_token=99,router_policy=policy)
            self.assertEqual((cfg['max_model_len'],cfg['num_gpu_blocks_override']),(65920,1031))
            self.assertEqual(cfg['max_num_batched_tokens'],16384)
            self.assertEqual(cfg['rope_scaling']['factor'],4.)
            for override in ({'ratio':.01},{'layers':[1]},{'count':5},{'tp':2}):
                with self.assertRaises(ValueError):engine_config('/model','router',cache_dir='/cache',end_token=99,router_policy=policy,**override)
            root=Path(td);model=root/'model';model.mkdir()
            atomic_json(model/'config.json',dict(model_type='qwen3',num_hidden_layers=64,hidden_size=5120))
            s=sample();s['model_config_sha256']=file_hash(model/'config.json');atomic_json(root/'sample.json',s)
            cmd=[sys.executable,str(ROOT/'run.py'),'run','--method','router','--router-policy',str(policy),
                '--model',str(model),'--input',str(root/'sample.json'),'--output',str(root/'never-created'),'--dry-run']
            proc=subprocess.run(cmd,capture_output=True,text=True,check=True,cwd='/tmp')
            self.assertEqual(json.loads(proc.stdout)['num_gpu_blocks_override'],1031)
            self.assertFalse((root/'never-created').exists())
            self.assertNotEqual(subprocess.run(cmd+['--ratio','.01'],capture_output=True).returncode,0)

    def test_export_refuses_unfinished_training(self):
        from runner.router_export import export
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):export(td,Path(td)/'never.json',ROOT/'scripts/router_cohort.json')
            self.assertFalse((Path(td)/'never.json').exists())


class FeaturesTests(unittest.TestCase):
    def test_attention_features_native_mass_and_stable_ties(self):
        layers=np.ones((64,128),dtype=np.float32)
        scores=np.ones(64,dtype=np.float32)
        f=features_from_arrays(layers,scores,64,128)
        self.assertEqual(f['top1_mass'],0.)
        self.assertEqual(f['top20_mass'],12/64)
        self.assertEqual(f['top50_mass'],.5)
        self.assertEqual(f['group_agreement'],1.)
        self.assertEqual(f['group_top20_jaccard'],1.)
        layers[32:,64:]=np.arange(64)
        f=features_from_arrays(layers,scores,64,128)
        self.assertEqual(f['group_top20_jaccard'],0.)
        self.assertEqual(features_from_arrays(layers,np.zeros(64),64,128),dict.fromkeys(FEATURES))
        for bad in (np.ones(63),np.full(64,np.nan),-scores):
            with self.assertRaises(ValueError):features_from_arrays(layers,bad,64,128)

    def test_exported_ranks_corruption_and_disagreement(self):
        from runner.router_measure import extract_features
        with tempfile.TemporaryDirectory() as td:
            arrays=[]
            for rank in range(4):
                p=Path(td)/f'{rank}.npz';np.savez(p,layers=np.ones((64,128),np.float32),scores=np.ones(64,np.float32),local_mean=np.ones(128,np.float32))
                arrays.append(dict(rank=rank,path=str(p),sha256=file_hash(p)))
            self.assertEqual(extract_features(arrays,sample(),4)['group_agreement'],1.)
            p=Path(arrays[-1]['path']);np.savez(p,layers=np.ones((64,128),np.float32),scores=np.arange(64),local_mean=np.ones(128,np.float32))
            with self.assertRaises(ValueError):extract_features(arrays,sample(),4)
            arrays[-1]['sha256']=file_hash(p)
            with self.assertRaises(RuntimeError):extract_features(arrays,sample(),4)


class LifecycleTests(unittest.TestCase):
    def test_real_scheduler_dense_bypasses_lookup_load_store_and_retires(self):
        from ucm.integration.vllm.persistent_connector import PersistentBlendConnector as Connector
        from ucm.sparse.prophetkv.lifecycle import TrackedStore
        rid='a'*32+':prompt-0:read:measured|router-dense'
        connector=object.__new__(Connector)
        connector.requests_blend_meta={};connector.requests_meta={};connector.req2rag_load_chunks={}
        connector.prompt_lengths={};connector.dispatches={}
        backend=Mock();connector.store=TrackedStore(backend);connector.store.router_dense=True
        request=NS(request_id=rid,num_prompt_tokens=384,num_tokens=384)
        self.assertEqual(connector.get_num_new_matched_tokens(request,0),(0,False))
        backend.lookup.assert_not_called()
        connector.router_dense_id=rid
        connector._connector_metadata=NS(request_meta={})
        connector.start_load_kv(None);connector.save_kv_layer(None);connector.wait_for_layer_load('layer0')
        backend.load_data.assert_not_called();backend.dump_data.assert_not_called()
        for method in ('lookup','load_data','dump_data'):
            with self.assertRaises(RuntimeError):getattr(connector.store,method)([])
        output=NS(scheduled_new_reqs=[NS(req_id=rid)],scheduled_cached_reqs=NS(req_ids=[rid],resumed_from_preemption=[False]))
        self.assertEqual(connector.build_connector_meta(output).request_meta,{})
        with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,PROPHETKV_SCHEDULER_RECEIPT=str(Path(td)/'scheduler.json')):
            self.assertEqual(connector.request_finished(request,[]),(False,None))
            self.assertEqual(json.loads((Path(td)/'scheduler.json').read_text()),dict(request_id=rid,requests_blend_meta=0,requests_meta=0))

    def test_routing_two_requests_retirement_export_sync_timing(self):
        from runner.router_measure import measure
        events=[];worker=NS(arm=Mock(),router_mode=Mock(),router_export=Mock(),drain=Mock(),retire=Mock())
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            def rpc(fn,kwargs=None):
                if fn is worker.router_mode:
                    events.append(('sync',kwargs['action']))
                    return [dict(rank=i,**kwargs) for i in range(4)]
                if fn is worker.arm:events.append(('arm',kwargs['request_id']));return []
                if fn is worker.drain:events.append(('drain',));return []
                if fn is worker.retire:
                    events.append(('retire',));return [dict(rank=i,quiescent=True,transfers={'pending':0},request_bookkeeping=0) for i in range(4)]
                if fn is worker.router_export:
                    events.append(('export',));rows=[]
                    for i in range(4):
                        path=Path(f'{kwargs["output"]}.rank{i}.npz');np.savez(path,layers=np.ones((64,128),np.float32),scores=np.ones(64,np.float32),local_mean=np.ones(128,np.float32))
                        rows.append(dict(rank=i,path=str(path),sha256=file_hash(path)))
                    return rows
                raise AssertionError('Unexpected RPC')
            llm=NS(collective_rpc=rpc,llm_engine=object())
            def generate(engine,tokens,budget,rid,thinking=False):
                events.append(('generate',budget,rid))
                return NS(outputs=[NS(token_ids=[7])]),.02,.1
            with patch.dict(sys.modules,{'runner.worker':worker}),patch('run.generate',side_effect=generate),patch('run.verify_diagnostics'):
                result,ttft,elapsed,routing=measure(llm,sample(),'a'*32+':p:read:measured',payload('baseline'),'router1',out,
                    unchanged=lambda:events.append(('immutable',)))
            self.assertEqual([e[1] for e in events if e[0]=='generate'],[1,256])
            self.assertLess(events.index(('export',)),events.index(('retire',)))
            self.assertLess(events.index(('retire',)),events.index(('sync','baseline')))
            self.assertGreater(ttft,.02)
            self.assertAlmostEqual(ttft,routing['routing_overhead_seconds']+.02)
            self.assertEqual(routing['internal_tokens'],1)
            self.assertTrue(routing['answer_request_id'].endswith('|router-dense'))
            self.assertEqual(len([e for e in events if e[0]=='arm']),1)

    def test_paired_action_requires_exact_outputs_scores_masks_group(self):
        diag=[dict(rank=i,diagnostics=[dict(kind='prophetkv_selection',scores=[.1,.2],selected_positions=[4],scoring_layers=list(range(64)))]) for i in range(4)]
        r=dict(prompt_id='a',group=0,gpu_uuids=['GPU-a'],input_sha256='x',output_token_ids=[1],routing={'decision':{'action':ACTIONS[0]}})
        fixed=dict(r,method=ACTIONS[0])
        self.assertTrue(paired_check(r,diag,fixed,diag)['output_tokens_equal'])
        for change in ({'output_token_ids':[2]},{'group':1},{'method':'baseline'}):
            with self.assertRaises(ValueError):paired_check(r,diag,dict(fixed,**change),diag)
        bad=copy.deepcopy(diag);bad[0]['diagnostics'][0]['scores'][0]=.3
        with self.assertRaises(ValueError):paired_check(r,diag,fixed,bad)


class WorkflowTests(unittest.TestCase):
    def test_env_precedence_relocation_and_commands(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fake=root/'python';fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n');fake.chmod(0o755)
            envfile=root/'.env';envfile.write_text(f'PYTHON_BIN="{fake}"\nEXPERIMENT_DIR="{root}/old"\nPREPARED_DIR="$EXPERIMENT_DIR/prepared"\n')
            for command in ('configure','prepare','verify','detach','resume','status'):
                proc=subprocess.run(['bash',str(ROOT/'scripts/a800_router.sh'),command],cwd='/tmp',
                    env=dict(os.environ,UCM_ENV_FILE=str(envfile),EXPERIMENT_DIR=str(root/'new')),capture_output=True,text=True,check=True)
                self.assertIn(str(root/'new/prepared'),proc.stdout)
                self.assertIn(command,proc.stdout)
            from scripts.router_control import groups
            g=[f'GPU-{i:08x}-0000-0000-0000-000000000000' for i in range(8)]
            self.assertEqual(groups(','.join(g[:4]),','.join(g[4:])),[g[:4],g[4:]])
            with self.assertRaises(ValueError):groups(','.join(g[:4]),','.join(g[:4]))

    def test_resume_hashes_identity_and_partial_records(self):
        from runner.router_sweep import accepted
        row=dict(id='a',ordinal=0,sha256='input');protocol=dict(groups=[['GPU-a'],['GPU-b']])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.assertIsNone(accepted(root,'baseline',row,protocol))
            folder=record_dir(root,'baseline','a');folder.mkdir(parents=True)
            record=dict(prompt_id='a',method='baseline',group=0,gpu_uuids=['GPU-a'],input_sha256='input',cache_immutable=True,
                retirement=[dict(quiescent=True,transfers={'pending':0},request_bookkeeping=0)]*4)
            atomic_json(folder/'result.json',record)
            self.assertIsNone(accepted(root,'baseline',row,protocol))
            atomic_json(folder/'validated.json',dict(complete=True,protocol_sha256=fingerprint(protocol),files={'result.json':file_hash(folder/'result.json')}))
            self.assertEqual(accepted(root,'baseline',row,protocol,full=False),record)
            (folder/'result.json').write_text('{}')
            with self.assertRaises(ValueError):accepted(root,'baseline',row,protocol,full=False)

    def test_synthetic_9100_record_reporting_and_plots(self):
        from runner.router_report import summarize,publish
        rows=[dict(id=f'{t}-{i}',subtask=f'task{t}',ordinal=i) for t in range(13) for i in range(100)]
        records=[]
        for case in CASES:
            for row in rows:
                routing=None
                if case.startswith('router'):
                    routing=dict(internal_tokens=1,decision={'action':ACTIONS[0]})
                records.append(dict(prompt_id=row['id'],subtask=row['subtask'],method=case,accuracy=.99 if routing else 1.,
                    output_cap_reached=False,routing=routing,timings=dict(ttft_seconds=2. if routing else 10.,
                    answer_engine_ttft_seconds=1. if routing else 10.,routing_overhead_seconds=1. if routing else 0.)))
        report=summarize(records,rows)
        self.assertEqual((report['answers'],report['router_probes']),(9100,3900))
        self.assertTrue(report['policies']['router1']['primary'])
        self.assertTrue(report['policies']['router1']['both_targets_met'])
        with self.assertRaises(ValueError):summarize(records[:-1],rows)
        with self.assertRaises(ValueError):summarize(records+[records[0]],rows)
        with tempfile.TemporaryDirectory() as td:
            publish(td,report,records,payload())
            files={p.name for p in Path(td).iterdir()}
            self.assertTrue({'results.txt','index.html','summary.json','raw-records.json','summary.csv','raw-records.csv','paired.csv'}<=files)
            self.assertTrue(all(f'tree-router{i}.{suffix}' in files for i in (1,2,3) for suffix in ('png','pdf')))



class NativeExecutionTests(unittest.TestCase):
    def test_seven_configurations_native_worker_resume_and_engine_exit(self):
        from runner import router_sweep as sweep
        events=[];worker=NS(setup=Mock(),arm=Mock(),drain=Mock(),retire=Mock(),router_mode=Mock(),
                            router_export=Mock(),router_dense_receipt=Mock())
        class Tokenizer:
            all_special_ids=[97,98,99]
            def convert_tokens_to_ids(self,t):return {'<think>':98,'</think>':97}[t]
            def convert_ids_to_tokens(self,t):return {98:'<think>',97:'</think>'}[t]
            def decode(self,t,**kw):return 'answer'
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prepared=root/'prepared';prepared.mkdir()
            s=sample();atomic_json(prepared/'input.json',s)
            row=dict(id='test-a-000',ordinal=0,prepared='input.json',sha256=file_hash(prepared/'input.json'),
                     subtask='a',references=['answer'],scoring='ruler_all')
            protocol=dict(prepared=str(prepared),model='/model',cache_root=str(root/'cache'),groups=[['GPU-a']*4,['GPU-b']*4])
            atomic_json(root/'protocol.json',protocol);write_policy(root,'baseline')
            def start(cfg):
                events.append(('start',cfg['num_gpu_blocks_override']))
                current={'cached':'kv_transfer_config' in cfg,'action':'prophetkv-all64-1'}
                llm=NS(llm_engine=NS(engine_core=NS(shutdown=lambda:events.append(('shutdown',)))),get_tokenizer=lambda:Tokenizer())
                llm.llm_engine.current=current
                def rpc(fn,kwargs=None):
                    if fn is worker.setup:return [dict(rope=[dict(formula_bitwise_equal=True,table_positions=131072)]*64,
                            kv_tokens=65920,kv_blocks=1031,block_size=64) for _ in range(4)]
                    if fn is worker.router_mode:
                        current.update(action=kwargs['action'],rid=kwargs['request_id'])
                        return [dict(rank=i,**kwargs) for i in range(4)]
                    if fn is worker.arm:return []
                    if fn is worker.retire:
                        events.append(('retire',));return [dict(rank=i,quiescent=True,transfers={'pending':0},request_bookkeeping=0) for i in range(4)]
                    if fn is worker.drain:return [dict(rank=i,diagnostics=[]) for i in range(4)]
                    if fn is worker.router_export:
                        items=[]
                        for i in range(4):
                            path=Path(f'{kwargs["output"]}.rank{i}.npz');np.savez(path,layers=np.ones((64,128),np.float32),
                                scores=np.ones(64,np.float32),local_mean=np.ones(128,np.float32))
                            items.append(dict(rank=i,path=str(path),sha256=file_hash(path)))
                        return items
                    if fn is worker.router_dense_receipt:
                        return [dict(rank=i,request_id=current['rid'],native_layers=sorted(f'model.layers.{j}.self_attn.attn' for j in range(64)),
                                     store_operations=dict(lookup=0,load=0,store=0)) for i in range(4)]
                    raise AssertionError('Unexpected worker call')
                llm.collective_rpc=rpc
                return llm
            class Cache:
                def __init__(self,*args):self.chunks=[[1]*64,[2]*64]
                def ready(self):events.append(('ready',));return {'complete':True}
                def unchanged(self):events.append(('immutable',))
                def delete(self):events.append(('delete',));return dict(deleted_shards=8)
            def generate(engine,tokens,budget,rid,thinking=False):
                events.append(('generate',budget,rid))
                atomic_json(Path(os.environ['PROPHETKV_SCHEDULER_RECEIPT']),dict(request_id=rid,requests_blend_meta=0,requests_meta=0))
                return NS(num_cached_tokens=0 if not engine.current['cached'] or rid.endswith('|router-dense') else 64,
                          outputs=[NS(token_ids=[7],text='answer',finish_reason='stop')]),.02,.04
            with patch.dict(sys.modules,{'runner.worker':worker}),patch.object(sweep,'start_engine',side_effect=start), \
                 patch.object(sweep,'check_environment',return_value=['GPU-a']*4),patch('scripts.router_inputs.verify_prepared',return_value=[row]), \
                 patch.object(sweep,'PromptCache',Cache),patch('runner.cache.wait_for_cache',return_value={'verified_shards':8}), \
                 patch('run.generate',side_effect=generate),patch('run.verify_diagnostics'):
                for case in CASES:sweep.run_group(root,0,case,'fixture')
                self.assertEqual(sum(e[0]=='start' for e in events),7)
                self.assertEqual(sum(e[0]=='shutdown' for e in events),7)
                for case in CASES:self.assertIsNotNone(sweep.accepted(root,case,row,protocol))
                previous=[file_hash(record_dir(root,c,row['id'])/'validated.json') for c in CASES]
                for case in CASES:sweep.run_group(root,0,case,'resume')
                self.assertEqual(sum(e[0]=='start' for e in events),7)
                self.assertEqual(previous,[file_hash(record_dir(root,c,row['id'])/'validated.json') for c in CASES])
            self.assertEqual(sum('measured-probe' in e[2] for e in events if e[0]=='generate'),3)
            self.assertEqual(sum(e[0]=='delete' for e in events),6)

    def test_supervisor_retries_operational_once_and_halts_validation(self):
        from scripts import router_control as control
        for failure in (75,76):
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);starts=[]
                class Child:
                    def __init__(self,code):self.pid=1000000+len(starts);self.code=code
                    def poll(self):return self.code
                    def wait(self,**kwargs):return self.code
                def launch(cmd,**kwargs):
                    starts.append(cmd)
                    return Child(failure if len(starts)==1 else 0)
                protocol=dict(groups=[['GPU-a']*4,['GPU-b']*4],cache_root=str(root/'cache'))
                rows=[dict(id='a',ordinal=0),dict(id='b',ordinal=1)]
                with patch.object(control,'verify',return_value=(protocol,rows)),patch.object(control,'accepted',return_value=None), \
                     patch.object(control.subprocess,'Popen',side_effect=launch),patch.object(control,'identity',return_value={'start':'1','cmd':'00'}), \
                     patch.object(control,'group_alive',return_value=False),patch.object(control,'terminate_owned'), \
                     patch.object(control.signal,'signal'),patch.object(control.time,'sleep'),patch('runner.router_report.finalize') as final:
                    if failure==75:
                        control.supervise(root)
                        self.assertEqual(len(starts),15);final.assert_called_once()
                        self.assertTrue(json.loads((root/'cleanup.json').read_text())['owned_engines_exited'])
                    else:
                        with self.assertRaises(RuntimeError):control.supervise(root)
                        final.assert_not_called();self.assertEqual(len(starts),2)

    def test_duplicate_supervisor_and_group_receipts_block_launch(self):
        from scripts.router_control import idle
        from runner.router_process import identity
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);atomic_json(root/'supervisor.json',dict(pid=os.getpid(),identity=identity(os.getpid())))
            with self.assertRaises(ValueError):idle(root)

    def test_readonly_persistent_router_configuration(self):
        from test_setups import SetupTests
        from runner import setups
        helper=SetupTests();helper.setUp()
        try:
            helper.args.tp=4
            helper.build()
            path=write_policy(helper.root)
            args=helper.run_args(method='router',ratio=None,router_policy=path,router_id='router1',max_output_tokens=256,dry_run=True)
            import io
            output=io.StringIO()
            with patch('sys.stdout',output):setups.run_collection(args)
            cfg=json.loads(output.getvalue())['engine']
            self.assertEqual(cfg['num_gpu_blocks_override'],1031)
            connector=cfg['kv_transfer_config']['kv_connector_extra_config']['persistent_setup']
            self.assertTrue(connector['readonly'])
            self.assertEqual(connector['tp'],4)
        finally:helper.doCleanups()

class ExportAndWatchdogTests(unittest.TestCase):
    def test_finalized_export_replays_all_saved_features_without_importing_training_code(self):
        from runner.router_export import export
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);atomic_json(root/'protocol.json',{'fixture':'synthetic finalized training data'})
            cohort=root/'cohort.json';atomic_json(cohort,dict(training_protocol_sha256=file_hash(root/'protocol.json')))
            ids=[f'tuning-fixture-{i:03d}' for i in range(195)];features=[];evidence={}
            for i,pid in enumerate(ids):
                f=dict.fromkeys(FEATURES,.1 if i%2 else .9);features.append(f)
                for case in (*ACTIONS,'tuning-probe'):
                    path=root/'records'/pid/(case+'.json')
                    atomic_json(path,dict(routing={'features':f}) if case=='tuning-probe' else {'fixture':True})
                    receipt=path.with_suffix('.validated.json');atomic_json(receipt,dict(complete=True,sha256={str(path):file_hash(path)}))
                    evidence[str(receipt)]=file_hash(receipt)
            for i in (1,2,3):atomic_json(root/'policies'/f'{i}.json',source_policy(i,'cost' if i==2 else 'loss'))
            atomic_json(root/'policies/training-matrices.json',dict(ids=ids,features=features))
            files={str(p):file_hash(p) for p in (root/'policies').glob('*.json')}
            atomic_json(root/'policies/lock.json',dict(complete=True,training_answers=780,training_probes=195,
                primary='router-1',sha256=files,training_evidence=evidence))
            atomic_json(root/'final-validation.json',dict(complete=True,validated_answers=780,tuning_probes=195,sha256=files))
            atomic_json(root/'cleanup.json',dict(complete=True))
            atomic_json(root/'supervisor.json',dict(state='complete',pid=999999999,identity={'start':'0','cmd':'00'}))
            output=root/'portable.json';result=export(root,output,cohort)
            self.assertEqual(load_policy(output),result)
            self.assertLess(output.stat().st_size,20000)
            self.assertEqual(json.loads(output.with_suffix('.verification.json').read_text())['training_rows'],195)
            self.assertNotIn('training_ids',output.read_text())
            export(root,output,cohort)  # idempotent one-shot, never a watcher
            atomic_json(root/'records'/ids[0]/'tuning-probe.json',{'corrupt':True})
            with self.assertRaises(ValueError):export(root,root/'never-published.json',cohort)
            self.assertFalse((root/'never-published.json').exists())

    def test_watchdog_allows_only_one_operational_retry(self):
        from scripts import router_control as control
        with tempfile.TemporaryDirectory() as td:
            starts=[]
            class Child:
                pid=999999999
                def poll(self):return None
                def wait(self,**kwargs):return 75
            def launch(*args,**kwargs):starts.append(args);return Child()
            ticks=iter(range(0,100000,1000))
            with patch.object(control,'verify',return_value=({'groups':[['GPU-a']*4,['GPU-b']*4],'cache_root':str(Path(td)/'cache')},[dict(id='a',ordinal=0)])), \
                 patch.object(control,'accepted',return_value=None),patch.object(control.subprocess,'Popen',side_effect=launch), \
                 patch.object(control,'identity',return_value={'start':'1','cmd':'00'}),patch.object(control,'group_alive',return_value=False), \
                 patch.object(control,'terminate_owned') as stop,patch.object(control.signal,'signal'), \
                 patch.object(control.time,'sleep'),patch.object(control.time,'monotonic',side_effect=lambda:next(ticks)), \
                 patch('runner.router_report.finalize') as final:
                with self.assertRaises(RuntimeError):control.supervise(Path(td))
                self.assertEqual(len(starts),2);self.assertEqual(stop.call_count,2);final.assert_not_called()
            state=json.loads((Path(td)/'supervisor.json').read_text())
            self.assertEqual(state['state'],'failed')

class CacheRecoveryTests(unittest.TestCase):
    def test_cleanup_requires_dead_owned_namespace_and_preserves_results(self):
        from scripts import router_control as control
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);protocol=dict(cache_root=str(root/'cache'))
            cache=root/'cache'/fingerprint(str(root))[:16]/'group0'/('a'*32)
            cache.mkdir(parents=True);(cache/'incomplete-kv').write_text('owned temporary bytes')
            owner=root/'sessions'/'attempt'/'ownership.json'
            atomic_json(owner,dict(pid=999999999,identity={'start':'1','cmd':'00'},cache=str(cache),group=0))
            result=root/'records'/'sentinel';result.parent.mkdir();result.write_text('accepted results')
            with patch.object(control,'alive',return_value=True),patch.object(control,'group_alive',return_value=False):
                self.assertEqual(control.cleanup_caches(root,protocol),[]);self.assertTrue(cache.exists())
            with patch.object(control,'alive',return_value=False),patch.object(control,'group_alive',return_value=False):
                self.assertEqual(control.cleanup_caches(root,protocol),[str(cache)])
                self.assertFalse(cache.exists());self.assertEqual(result.read_text(),'accepted results')
                outside=root/'unrelated';outside.mkdir();(outside/'sentinel').write_text('preserve')
                atomic_json(owner,dict(pid=999999999,identity={'start':'1','cmd':'00'},cache=str(outside),group=0))
                with self.assertRaises(ValueError):control.cleanup_caches(root,protocol)
                self.assertTrue((outside/'sentinel').exists())


if __name__=='__main__':unittest.main()
