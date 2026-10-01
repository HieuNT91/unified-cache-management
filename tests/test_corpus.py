from runner.layout import stamp_sample
"""CPU-only reusable corpus, pooled trainer, and shared live lifecycle contracts."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from runner.tree_policy import actions,DEFAULT_ACTIONS,FEATURE_DEFINITIONS,SCHEMA,export,load,decide
from runner.router_policy import FEATURES,digest,features_from_arrays
from runner.corpus import selection,load_attention,match_answer,Corpus
from runner.corpus_records import publish,accepted,record_dir,protocol_identity
from runner.corpus_train import split,fold_ids,search,GRID,LOSS_GRID,COST_GRID,train,rank
from runner.corpus_eval import replay,test_tree as evaluate,evaluation_ids
from runner.setups import atomic_json,file_hash
from scripts.corpus_inputs import TASKS,inventory,format_corpus,prepared_rows


def inventory_for(count):
    return [dict(id='nocache',method='baseline',ratio=None)]+[
        dict(id=f'prophetkv-{i}',method='prophetkv',ratio=i/100) for i in range(1,count)]


def tree_value(inventory=DEFAULT_ACTIONS):
    return dict(schema=SCHEMA,actions=inventory,feature_names=list(FEATURES),feature_definitions=FEATURE_DEFINITIONS,
        tree=dict(feature='top1_mass',threshold=.1,le=dict(action=inventory[-1]['id'],training_samples=5),
                  gt=dict(action='nocache',training_samples=5)),training_costs=[1.]*len(inventory),
        training_ids=['training'],heldout_ids=['heldout'],provenance=dict(snapshot_sha256='a'*64,
        training_run_sha256='b'*64,corpus_protocol_sha256='c'*64))


def signed(data):
    data=copy.deepcopy(data);data['payload_sha256']=digest({k:v for k,v in data.items() if k!='payload_sha256'});return data


def tiny_sample():
    return stamp_sample(dict(token_ids=[1]*260,boundaries=[0,64,128,260],question_positions=[255],original_to_formatted=list(range(260))))


def archives(folder,inventory=DEFAULT_ACTIONS,sample=None):
    sample=sample or tiny_sample();end=sample['boundaries'][-2];prefix=sample['boundaries'][1]
    layers=np.full((64,end),1/end,np.float32);mean=np.zeros(end,np.float32)
    for layer in layers:np.add(mean,layer,out=mean)
    mean/=np.float32(64);scores=mean[prefix:]
    artifacts=[]
    for rank in range(4):
        path=folder/f'attention.rank{rank}.npz'
        arrays=dict(layers=layers,local_mean=mean,scores=scores,context_positions=np.arange(end),
            boundaries=sample['boundaries'],question_positions=sample['question_positions'],original_to_formatted=sample['original_to_formatted'])
        arrays.update({name:selection(scores,prefix,d['ratio']) for name,d in actions(inventory).items() if name!='nocache'})
        np.savez_compressed(path,**arrays);artifacts.append(dict(rank=rank,path=path.name,sha256=file_hash(path)))
    return dict(artifacts=artifacts)


class FakeCorpus:
    """Synthetic complete measured data; no claims of GPU acceptance."""
    def __init__(self,root,per_task=50,inventory=DEFAULT_ACTIONS):
        self.root=Path(root);self.actions=actions(inventory)
        self.protocol=dict(kind='collection',dataset='ruler',actions=inventory,plan_sha256='a'*64,hardware={'synthetic':True})
        self.rows={f'{t}-{i:03d}':dict(id=f'{t}-{i:03d}',subtask=t,ordinal=i,sha256='d'*64) for i in range(per_task) for t in TASKS}
        self.pins={}
        for pid in self.rows:
            for c in ['probe',*self.actions]:
                path=record_dir(root,c,pid)/'validated.json';atomic_json(path,dict(synthetic=True,id=pid,case=c))
                self.pins[str(path.relative_to(root))]=file_hash(path)
    def snapshot(self):
        return dict(samples=[dict(id=r['id'],task=r['subtask'],ordinal=r['ordinal']) for r in self.rows.values()],
            protocol_sha256=protocol_identity(self.protocol),plan_sha256=self.protocol['plan_sha256'],actions=self.protocol['actions'],
            acceptance_hashes=dict(self.pins),counts=dict(complete=len(self.rows)))
    def features(self,pid):return dict.fromkeys(FEATURES,.5)
    def outcome(self,pid,action):
        if action not in self.actions:raise ValueError('Unknown corpus action')
        return dict(accuracy=1.,timings=dict(answer_engine_ttft_seconds=10. if action=='nocache' else 1.))
    def probe(self,pid):return dict(timings=dict(routing_overhead_seconds=.25))


class CorpusTests(unittest.TestCase):
    def test_generic_inventories_and_tree_boundaries(self):
        for count in (2,4,7):
            inv=inventory_for(count);tree=signed(tree_value(inv));self.assertEqual(len(actions(inv)),count)
            for value,expected in ((np.nextafter(.1,-np.inf),inv[-1]['id']),(.1,inv[-1]['id']),
                                   (np.nextafter(.1,np.inf),'nocache')):
                features=dict.fromkeys(FEATURES,.5);features['top1_mass']=float(value)
                self.assertEqual(decide(tree,features)['action'],expected)
            self.assertEqual(decide(tree,{})['action'],'nocache')
            self.assertEqual(decide(tree,dict.fromkeys(FEATURES,float('nan')))['action'],'nocache')
        for bad in (DEFAULT_ACTIONS+[dict(id='probe',method='prophetkv',ratio=.5)],DEFAULT_ACTIONS+[DEFAULT_ACTIONS[-1]],DEFAULT_ACTIONS[1:],[dict(id='probe',method='magic',ratio=.1)]):
            with self.assertRaises(ValueError):actions(bad)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'tree.json';export(path,tree_value());self.assertEqual(load(path)['actions'],DEFAULT_ACTIONS)
            with self.assertRaises(FileExistsError):export(path,tree_value())
            path.write_text(path.read_text()+' ')
            with self.assertRaises(ValueError):load(path)
        broken=signed(tree_value());broken['tree']['feature']='task_id';broken=signed(broken)
        with self.assertRaises(ValueError):decide(broken,{})

    def test_archives_lossless_arithmetic_rank_corruption_ties(self):
        for count in (2,4,7):
            with tempfile.TemporaryDirectory() as td:
                folder=Path(td);inv=inventory_for(count);receipt=archives(folder,inv)
                a=load_attention(folder,receipt,tiny_sample(),inv)
                self.assertEqual(a['layers'].dtype,np.float32)
                for name,d in actions(inv).items():
                    if name!='nocache':self.assertEqual(a['selections'][name].tolist(),list(range(64,64+math.floor(64*d['ratio']))))
                path=folder/receipt['artifacts'][0]['path']
                with np.load(path) as z:values={k:z[k] for k in z.files}
                values['local_mean'][0]+=.01;np.savez_compressed(path,**values)
                with self.assertRaisesRegex(ValueError,'Corrupt'):load_attention(folder,receipt,tiny_sample(),inv)
                receipt['artifacts'][0]['sha256']=file_hash(path)
                with self.assertRaisesRegex(ValueError,'replay'):load_attention(folder,receipt,tiny_sample(),inv)
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);receipt=archives(folder);path=folder/receipt['artifacts'][1]['path']
            with np.load(path) as z:values={k:z[k] for k in z.files}
            values['scores']*=np.float32(2)
            for a,d in actions().items():
                if a!='nocache':values[a]=selection(values['scores'],64,d['ratio'])
            np.savez_compressed(path,**values);receipt['artifacts'][1]['sha256']=file_hash(path)
            with self.assertRaisesRegex(ValueError,'TP score'):load_attention(folder,receipt,tiny_sample())
            receipt['artifacts'][1]['rank']=0
            with self.assertRaisesRegex(ValueError,'rank'):load_attention(folder,receipt,tiny_sample())

    def test_exact_total_split_all_tasks_and_folds(self):
        samples=[dict(id=f'{task}-{i}',task=task) for i in range(50) for task in TASKS]
        result=split(samples,390,42)
        self.assertEqual(len(result['train_ids']),390);self.assertEqual(len(result['heldout_ids']),260)
        self.assertEqual(set(result['allocation'].values()),{30});self.assertEqual(result,split(list(reversed(samples)),390,42))
        self.assertFalse(set(result['train_ids'])&set(result['heldout_ids']))
        for n in (13,14,251,637):self.assertEqual(len(split(samples,n,42)['train_ids']),n)
        for n in (12,638):
            with self.assertRaises(ValueError):split(samples,n,42)
        with self.assertRaises(ValueError):split([r for r in samples if r['task']!=TASKS[0]],13,42)
        byid={r['id']:r for r in samples};training=[byid[i] for i in result['train_ids']]
        folds=fold_ids(training,42)
        for task in TASKS:self.assertEqual([sum(f==j and r['task']==task for f,r in zip(folds,training)) for j in range(5)],[6]*5)
        self.assertEqual(set(fold_ids([dict(id=t,task=t) for t in TASKS],42)),set(range(5)))

    def test_all_settings_fold_local_and_generic_counts(self):
        self.assertEqual((len(LOSS_GRID),len(COST_GRID),len(GRID)),(90,126,216))
        samples=[dict(id=f'{t}-{i}',task=t) for i in range(3) for t in TASKS]
        features=[dict.fromkeys(FEATURES,.5) for _ in samples];features[0]={}
        for count in (2,4,6):
            inv=inventory_for(count);scores=np.ones((len(samples),count));times=np.ones_like(scores);times[:,0]=10
            policies,table,folds=search(samples,features,scores,times,np.ones(len(samples))*.1,inv)
            self.assertEqual(len(table),216);self.assertEqual(len(policies),3);self.assertTrue(policies[0]['primary'])
            self.assertTrue(policies[1]['duplicate_of']);self.assertEqual(table[0]['oof_actions'][0],0)
            stat=table[0]['fold_statistics'][0]
            self.assertFalse(set(stat['train_ids'])&set(stat['validation_ids']))
            self.assertEqual(len(stat['costs']),count)
            modified=times.copy();modified[np.array(folds)==0,0]=1000
            _,other,_=search(samples,features,scores,modified,np.ones(len(samples))*.1,inv)
            self.assertEqual(stat,other[0]['fold_statistics'][0])
            self.assertEqual([r['index'] for r in rank(table)], [p['oof']['index'] for p in policies])

    def test_partial_650_train390_offline_and_frozen_export(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);corpus=FakeCorpus(root/'corpus');output=root/'training'
            train(corpus,390,output)
            self.assertEqual(json.loads((output/'complete.json').read_text())['heldout'],260)
            tree=load(output/'router1.json');self.assertEqual(len(tree['training_ids']),390)
            report=json.loads((output/'router1-heldout/report.json').read_text())
            self.assertEqual(report['overall']['samples'],260);self.assertEqual(report['task_coverage'],13)
            self.assertEqual(report['overall']['estimated_router_ttft_seconds'],1.25)
            self.assertEqual(len(json.loads((output/'search.json').read_text())),216)
            self.assertEqual(evaluation_ids(corpus,tree),tree['heldout_ids'])
            before=file_hash(output/'router1.json');corpus.rows['later']=dict(id='later',subtask=TASKS[0],ordinal=99)
            self.assertEqual(file_hash(output/'router1.json'),before)
            with self.assertRaisesRegex(ValueError,'overlapping'):evaluation_ids(corpus,tree,corpus.snapshot())
            with self.assertRaises(ValueError):replay(corpus,[dict(sample_id=tree['heldout_ids'][0],action='uncollected')],root/'bad')
            path=next(iter(tree['heldout_hashes']));(corpus.root/path).write_text('corruption')
            with self.assertRaisesRegex(ValueError,'changed'):evaluation_ids(corpus,tree)

    def test_generation_plan_interleaving_qa_and_reference_separation(self):
        qa={t:[dict(source_index=i,question_sha256=digest([t,i]),question=f'{t} {i}') for i in range(200)] for t in ('qa_1','qa_2')}
        plan=inventory(qa);self.assertEqual(len(plan),2600);self.assertEqual({r['task'] for r in plan[:13]},set(TASKS))
        self.assertEqual(len({r.get('question_sha256') for r in plan if r['task'].startswith('qa_')}),400)
        self.assertEqual(plan,inventory(qa))
        class Tokenizer:
            chat_template="test native template"
            def apply_chat_template(self,messages,**kwargs):
                self.thinking=kwargs['enable_thinking'];return '[USER]'+messages[0]['content']+'[END]\n'
            def __call__(self,text,**kwargs):return dict(input_ids=list(map(ord,text)),offset_mapping=[(i,i+1) for i in range(len(text))])
            def encode(self,text,**kwargs):return list(map(ord,text))
            def convert_tokens_to_ids(self,text):return 999
        for task in TASKS:
            query='What are all the special magic numbers for KEY?' if task.startswith('niah') else 'Question: Return TARGET.'
            row=dict(index=0,input='[USER]Instructions\n'+'context '*700+'\n'+query+'[END]\n',answer_prefix=' Answer:',outputs=['SECRET'])
            tokenizer=Tokenizer();sample=format_corpus(tokenizer,row,task)
            self.assertEqual(sample['token_ids'],list(map(ord,row['input']+row['answer_prefix'])));self.assertFalse(tokenizer.thinking)
            from scripts.ruler_64000 import CAPS
            self.assertEqual(sample['max_output_tokens'],CAPS[task]);self.assertNotIn('references',sample);self.assertNotIn('outputs',sample)
            self.assertEqual(sample,format_corpus(tokenizer,row,task))
            self.assertEqual(''.join(chr(sample['token_ids'][p]) for p in sample['question_positions']),query)

    def test_atomic_receipts_interrupted_writes_relocation_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prepared=root/'inputs';prepared.mkdir();atomic_json(prepared/'sample.json',tiny_sample())
            row=dict(id='sample',ordinal=0,subtask=TASKS[0],prepared='sample.json',sha256=file_hash(prepared/'sample.json'))
            protocol=dict(actions=DEFAULT_ACTIONS,groups=[['GPU-'+str(i) for i in range(4)]],prepared=str(prepared))
            init=root/'session/init.json';atomic_json(init,dict(validated=True,engine_config={}))
            retired=[dict(rank=i,quiescent=True,transfers={'pending':0},request_bookkeeping=0) for i in range(4)]
            record=dict(prompt_id='sample',method='nocache',executed_action='nocache',input_sha256=row['sha256'],group=0,gpu_uuids=protocol['groups'][0],
                cache_immutable=True,retirement=retired,initialization='session/init.json',initialization_sha256=file_hash(init),
                accuracy=1.,num_cached_tokens=0,timings=dict(answer_engine_ttft_seconds=1.))
            folder=record_dir(root,'nocache','sample');folder.mkdir(parents=True);atomic_json(folder/'result.json',record)
            self.assertIsNone(accepted(root,'nocache',row,protocol))
            publish(root,'nocache',row,protocol,record,[])
            self.assertEqual(accepted(root,'nocache',row,dict(protocol,prepared='/relocated')),record)
            with self.assertRaises(FileExistsError):publish(root,'nocache',row,protocol,record,[])
            (folder/'result.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Corrupt'):accepted(root,'nocache',row,protocol)

    def test_profiles_uuid_groups_and_capacity(self):
        from runner.tree_profiles import config
        from scripts.corpus_control import groups,check_hardware
        uuids=[f'GPU-{i:08x}-1111-2222-3333-444444444444' for i in range(8)]
        self.assertEqual(len(groups(','.join(uuids[:4]))),1)
        self.assertEqual(len(groups(','.join(uuids[:4]),','.join(uuids[4:]))),2)
        with self.assertRaises(ValueError):groups('0,1,2,3')
        with self.assertRaises(ValueError):groups(','.join(uuids[:4]),','.join(uuids[:4]))
        for dataset,kv in [('ruler',65920),('longbench-v2',131072)]:
            dense=config('/model',dataset,False,'/cache',9);cached=config('/model',dataset,True,'/cache',9)
            self.assertNotIn('kv_transfer_config',dense);self.assertIn('kv_transfer_config',cached)
            self.assertEqual(cached['max_model_len'],kv);self.assertEqual(cached['hf_overrides']['ucm_activation_tile'],4096)
            self.assertEqual(cached['max_num_batched_tokens'],16384)
        with tempfile.TemporaryDirectory() as td:
            model=Path(td);atomic_json(model/'model.safetensors.index.json',dict(metadata=dict(total_size=64*2**30)))
            for name,capacity in [('NVIDIA A800',81920),('NVIDIA L20',49140)]:
                inventory='\n'.join(f'{u}, {name}, {capacity}, {capacity}' for u in uuids[:4])
                with patch('scripts.corpus_control.subprocess.check_output',side_effect=[inventory,'']):
                    hw=check_hardware(dict(model=td,dataset='longbench-v2',groups=[uuids[:4]]))
                self.assertEqual(hw['groups'][0][0]['name'],name)
            inventory='\n'.join(f'{u}, NVIDIA L20, 24000, 24000' for u in uuids[:4])
            with patch('scripts.corpus_control.subprocess.check_output',return_value=inventory):
                with self.assertRaisesRegex(ValueError,'cannot fit'):check_hardware(dict(model=td,dataset='longbench-v2',groups=[uuids[:4]]))

    def test_explicit_local_profile_preserves_precision_and_memory_gates(self):
        from runner.tree_profiles import config
        from scripts.corpus_control import check_hardware
        cfg=config('/model','ruler',True,'/cache',9,'rtx4500ada')
        self.assertEqual(cfg['gpu_memory_utilization'],.95)
        self.assertEqual((cfg['dtype'],cfg['tensor_parallel_size']),('bfloat16',4))
        self.assertEqual((cfg['max_model_len'],cfg['num_gpu_blocks_override']),(65920,1031))
        self.assertEqual(cfg['rope_scaling']['factor'],4.)
        self.assertIsNone(cfg['quantization']);self.assertEqual(cfg['cpu_offload_gb'],0)
        local_lb=config('/model','longbench-v2',False,'/cache',9,'rtx4500ada')
        self.assertEqual(local_lb['max_model_len'],65920)
        self.assertEqual(local_lb['num_gpu_blocks_override'],1031)
        self.assertNotIn('kv_transfer_config',local_lb)
        self.assertEqual(config('/model','longbench-v2',False,'/cache')['max_model_len'],131072)
        uuids=[f'GPU-{i:08x}-1111-2222-3333-444444444444' for i in range(4)]
        with tempfile.TemporaryDirectory() as td:
            atomic_json(Path(td)/'model.safetensors.index.json',dict(metadata=dict(total_size=61.2*2**30)))
            settings=dict(model=td,dataset='ruler',groups=[uuids],hardware_profile='rtx4500ada')
            inventory='\n'.join(f'{u}, NVIDIA RTX 4500 Ada Generation, 24570, 24087' for u in uuids)
            with patch('scripts.corpus_control.subprocess.check_output',side_effect=[inventory,'']):
                self.assertLess(check_hardware(settings)['required_mib_per_rank'],24087)
            with patch('scripts.corpus_control.subprocess.check_output',return_value=inventory):
                with self.assertRaisesRegex(ValueError,'explicit hardware'):check_hardware({k:v for k,v in settings.items() if k!='hardware_profile'})
            with patch('scripts.corpus_control.subprocess.check_output',return_value=inventory.replace('24087','19000')):
                with self.assertRaisesRegex(ValueError,'cannot fit'):check_hardware(settings)
            with patch('scripts.corpus_control.subprocess.check_output',side_effect=[inventory,uuids[0]]):
                with self.assertRaisesRegex(ValueError,'occupied'):check_hardware(settings)

    def test_collection_rejects_lost_yarn_delta_normalization(self):
        from runner.corpus_runtime import Engine
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);session=root/'session';session.mkdir()
            atomic_json(session/'initialization.json',dict(validated=True))
            engine=Engine.__new__(Engine);engine.sample={}
            engine.root=root;engine.session=session;engine.cached=True;engine.group=0
            engine.row=dict(id='sample',sha256='input');engine.construction={}
            engine.protocol=dict(groups=[['a','b','c','d']]);engine.llm=Mock()
            audits=[dict(rank=r,normalized=True,delta_amplitude=1.,aliases_native_table=False) for r in range(4)]
            engine.llm.collective_rpc.return_value=audits
            self.assertEqual(engine.common('probe',{},[])['alignment_audit'],audits)
            for key,value in [('normalized',False),('delta_amplitude',1.140625),('aliases_native_table',True)]:
                engine.llm.collective_rpc.return_value=[dict(a,**{key:value}) for a in audits]
                with self.assertRaisesRegex(ValueError,'YaRN delta'):engine.common('probe',{},[])
            engine.llm.collective_rpc.return_value=audits[:3]
            with self.assertRaisesRegex(ValueError,'YaRN delta'):engine.common('probe',{},[])

    def test_model_snapshot_symlinks_do_not_relax_corpus_containment(self):
        from scripts.corpus_control import model_artifact
        from runner.corpus import relative
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);snapshot=root/'snapshot';snapshot.mkdir()
            blob=root/'blob';blob.write_bytes(b'weights')
            (snapshot/'weights.safetensors').symlink_to(blob)
            self.assertTrue(model_artifact(snapshot,'weights.safetensors').is_file())
            for name in ('../blob',str(blob),''):
                with self.assertRaises(ValueError):model_artifact(snapshot,name)
            with self.assertRaisesRegex(ValueError,'escapes corpus'):relative(snapshot,'weights.safetensors')

    def test_fake_live_control_across_datasets_and_dense(self):
        from runner.corpus_runtime import infer_prompt
        class FakeEngine:
            def __init__(self,dataset):self.events=[];self.dataset=dataset;self.sample={}
            def probe(self,folder):
                self.events+=['probe','retire','features']
                return dict(features=dict.fromkeys(FEATURES,.05)),[],{}
            def answer(self,definition,case,attention,overhead=0,route_started=None):
                self.events.append(case)
                return dict(executed_action=definition['id'],prompt_id='p',group=0,gpu_uuids=['u'],input_sha256='s',output_token_ids=[1],
                            timings={'ttft_seconds':overhead+1}),[dict(rank=0,diagnostics=[])]
        for dataset in ('ruler','longbench-v2'):
            for dense in (True,False):
                with tempfile.TemporaryDirectory() as td:
                    engine=FakeEngine(dataset);tree=tree_value();tree['prompt_protocol']='rpkv-original-tokens-v1'
                    if dense:tree['tree']=dict(action='nocache',training_samples=10)
                    tree=signed(tree);baseline,ds=engine.answer(actions()['nocache'],'baseline',None);engine.events=[]
                    result,_=infer_prompt(engine,tree,Path(td),baseline,ds)
                    self.assertEqual(engine.events,['probe','retire','features','router']+([] if dense else ['control']))
                    self.assertTrue(result['paired_validation']['output_tokens_equal'])
                    self.assertGreater(result['timings']['ttft_seconds'],1)
        from runner.corpus_runtime import paired
        with self.assertRaisesRegex(ValueError,'output'):
            paired(dict(prompt_id='a'),[],dict(prompt_id='b'),[])

    def test_synthetic_full_and_partial_reporting(self):
        from scripts.corpus_control import snapshot_report
        for n in (50,200):
            rows=[dict(id=f'{t}-{i}',ordinal=i,subtask=t) for i in range(n) for t in TASKS]
            protocol=dict(kind='collection',dataset='ruler',limit_per_task=200,actions=DEFAULT_ACTIONS,hardware={'synthetic':True})
            def record(root,case,row,protocol):
                return dict(accuracy=1.,timings=dict(answer_engine_ttft_seconds=2.,routing_overhead_seconds=.2))
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);atomic_json(root/'cleanup.json',dict(owned_engines_exited=True))
                with patch('scripts.corpus_control.verify',return_value=(protocol,rows)),patch('scripts.corpus_control.accepted',side_effect=record):
                    result=snapshot_report(root,True)
                self.assertEqual(result['answers'],13*n*4);self.assertEqual(result['accepted_probes'],13*n)
                self.assertEqual(result['complete'],n==200);self.assertEqual(result['status'],'complete' if n==200 else 'partial-complete')


class CorpusOperationalTests(unittest.TestCase):
    def test_snapshot_freezes_receipts_while_new_records_publish(self):
        # The snapshot first freezes receipt visibility, then validates their bodies.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prepared=root/'prepared';prepared.mkdir()
            atomic_json(prepared/'p.json',dict(tokens=[]));(prepared/'raw.json').write_text('{}')
            row=dict(id='p',subtask=TASKS[0],ordinal=0,prepared='p.json',sha256=file_hash(prepared/'p.json'),
                     raw='raw.json',raw_sha256=file_hash(prepared/'raw.json'))
            corpus=object.__new__(Corpus);corpus.root=root;corpus.prepared=prepared;corpus.rows={'p':row}
            corpus.actions=actions();corpus.protocol=dict(actions=DEFAULT_ACTIONS,plan_sha256='a'*64)
            for case in ['probe','nocache','prophetkv-1','prophetkv-20']:
                atomic_json(record_dir(root,case,'p')/'validated.json',dict(case=case))
            def arriving_probe(pid):
                atomic_json(record_dir(root,'prophetkv-40','p')/'validated.json',dict(case='prophetkv-40'))
                return {'ok':True}
            corpus.probe=arriving_probe;corpus.outcome=lambda pid,a:{'ok':True}
            first=corpus.snapshot();self.assertEqual(first['counts']['complete'],0)
            second=corpus.snapshot();self.assertEqual(second['counts']['complete'],1)
            (prepared/'p.json').write_text('corrupted')
            with self.assertRaisesRegex(ValueError,'preparation'):corpus.snapshot()

    def test_duplicate_publication_is_serialized(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);row={'id':'p'};protocol={'version':1}
            def write(i):
                try:publish(root,'nocache',row,protocol,{'writer':i},[]);return 'ok'
                except FileExistsError:return 'duplicate'
            with ThreadPoolExecutor(2) as pool:results=list(pool.map(write,range(2)))
            self.assertEqual(sorted(results),['duplicate','ok'])
            receipt=json.loads((record_dir(root,'nocache','p')/'validated.json').read_text())
            self.assertEqual(receipt['files']['result.json'],file_hash(record_dir(root,'nocache','p')/'result.json'))

    def test_capture_cleanup_on_export_error_and_missing_capture(self):
        import torch
        from runner.corpus_worker import export as export_capture
        with tempfile.TemporaryDirectory() as td:
            sample=tiny_sample();layers=torch.ones((64,128),dtype=torch.float32)
            sparse=NS(router_capture=True,router_arrays=(layers,torch.ones(64),torch.ones(128)))
            with patch('ucm.sparse.state.get_ucm_sparse',return_value=sparse),patch('vllm.distributed.get_tensor_model_parallel_rank',return_value=0),patch('torch.cuda.synchronize'),patch('numpy.savez_compressed',side_effect=RuntimeError('write failed')):
                with self.assertRaisesRegex(RuntimeError,'write failed'):export_capture(None,td,sample)
            self.assertFalse(sparse.router_capture);self.assertIsNone(sparse.router_arrays)
            self.assertFalse(list(Path(td).glob('*.tmp')))
            sparse.router_capture=True
            with patch('ucm.sparse.state.get_ucm_sparse',return_value=sparse),patch('vllm.distributed.get_tensor_model_parallel_rank',return_value=0):
                with self.assertRaisesRegex(RuntimeError,'Missing'):export_capture(None,td,sample)
            self.assertFalse(sparse.router_capture)

    def test_missing_only_resume_uses_existing_probe(self):
        from runner.corpus_collect import run_group
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);rows=[dict(id='a',ordinal=0),dict(id='b',ordinal=1)]
            protocol=dict(kind='collection',groups=[['uuid']*4],actions=DEFAULT_ACTIONS)
            accepted_cases={(r['id'],case) for r in rows for case in ['nocache','probe',*actions()]}
            accepted_cases.remove(('b','prophetkv-40'))
            engine=Mock();engine.sample={};engine.answer.return_value=({'value':1},[]);engine.end.return_value={'deleted':True}
            def present(root,case,row,protocol):return {'existing':True} if (row['id'],case) in accepted_cases else None
            with patch('scripts.corpus_control.verify',return_value=(protocol,rows)),patch('runner.corpus_collect.check_environment',return_value=protocol['groups'][0]),patch('runner.corpus_collect.accepted',side_effect=present),patch('runner.corpus_collect.Engine',return_value=engine) as constructor,patch('runner.corpus_collect.load_attention',return_value={'saved':True}) as attention,patch('runner.corpus_collect.publish') as publication:
                run_group(root,0,'cached','resume')
                self.assertEqual(constructor.call_args.args[-1],[rows[1]])
                engine.probe.assert_not_called();engine.begin.assert_called_once_with(rows[1])
                self.assertEqual(engine.answer.call_args.args[1],'prophetkv-40')
                attention.assert_called_once();publication.assert_called_once();engine.close.assert_called_once()

    def test_env_precedence_cli_and_explicit_tree(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);settings=root/'config.env'
            uuids=','.join(f'GPU-{i:08x}-1111-2222-3333-444444444444' for i in range(4))
            settings.write_text(f'PYTHON_BIN={sys.executable}\nEXPERIMENT_DIR={root}/ignored\nPREPARED_DIR={root}/prepared\nMODEL_PATH={root}/model\nCACHE_ROOT={root}/cache\nGPU_A={uuids}\nGPU_B=\n')
            env=dict(os.environ,UCM_ENV_FILE=str(settings),EXPERIMENT_DIR=str(root/'actual'),CUDA_VISIBLE_DEVICES='')
            script=Path(__file__).resolve().parents[1]/'scripts/ruler_corpus.sh'
            subprocess.run([str(script),'configure','--limit-per-task','50'],env=env,check=True,capture_output=True)
            saved=json.loads((root/'actual/settings.json').read_text())
            self.assertEqual(saved['prepared'],str(root/'prepared'));self.assertFalse((root/'ignored').exists())
            self.assertEqual(json.loads((root/'actual/tranche.json').read_text())['limit_per_task'],50)
            subprocess.run([str(script),'configure','--limit-per-task','200'],env=env,check=True,capture_output=True)
            self.assertEqual(json.loads((root/'actual/settings.json').read_text()),saved)
            self.assertEqual(json.loads((root/'actual/tranche.json').read_text())['limit_per_task'],200)
            env['EXPERIMENT_DIR']=str(root/'infer')
            result=subprocess.run([str(script.with_name('router_infer.sh')),'--dataset','ruler'],env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('supplied --tree',result.stderr)

    def test_final_reports_require_owned_exit(self):
        from scripts.corpus_control import snapshot_report
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);protocol=dict(kind='collection',dataset='ruler',actions=DEFAULT_ACTIONS,hardware={})
            row=dict(id='p',ordinal=0,subtask=TASKS[0])
            result=dict(accuracy=1.,timings=dict(answer_engine_ttft_seconds=1.,routing_overhead_seconds=.1))
            atomic_json(root/'cleanup.json',dict(owned_engines_exited=False))
            with patch('scripts.corpus_control.verify',return_value=(protocol,[row])),patch('scripts.corpus_control.accepted',return_value=result):
                with self.assertRaisesRegex(ValueError,'exit'):snapshot_report(root,True)
            self.assertFalse((root/'report.json').exists())


class AdditionalTrainerTests(unittest.TestCase):
    def test_nonconstant_shared_tree_and_arbitrary_dense_position(self):
        from runner.corpus_train import fit,choose,leaf,leaf_count
        x=np.zeros((80,5));x[:,0]=np.arange(80)/80
        # Two sparse losses; the second action is better on the opposite half.
        y=np.column_stack((np.r_[np.zeros(40),np.ones(40)],np.r_[np.ones(40),np.zeros(40)]))
        times=np.tile([1.,2.,10.],(80,1));overhead=np.zeros(80)
        for h in [dict(family='loss',depth=3,min_leaf=10,shrinkage=0,threshold=.02),
                  dict(family='cost',depth=3,min_leaf=10,lam=100,penalty=0.)]:
            tree,costs,norm=fit(x,y,times,overhead,h,2,[0,1])
            self.assertEqual(leaf_count(tree),2)
            self.assertEqual(choose(leaf(tree,x[0]),costs,h,2,[0,1]),0)
            self.assertEqual(choose(leaf(tree,x[-1]),costs,h,2,[0,1]),1)
            self.assertEqual(norm,10.)
        samples=[dict(id=t,task=t) for t in TASKS]
        inventory=inventory_for(2);inventory.reverse()
        policies,_,_=search(samples,[dict.fromkeys(FEATURES,.5) for _ in samples],np.ones((13,2)),
            np.tile([1.,10.],(13,1)),np.ones(13)*.1,inventory,count=5)
        self.assertEqual(len(policies),5)
        self.assertEqual(policies[0]['tree']['action'],inventory[0]['id'])

    def test_exact_reduction_rejects_impossible_scores(self):
        from runner.corpus import reduction_matches
        means=[np.array([.1,.2,.3,.4],np.float32)*np.float32(i+1) for i in range(4)]
        scores=((means[0]+means[1])+(means[2]+means[3]))/np.float32(4)
        self.assertTrue(reduction_matches(means,scores,0))
        self.assertFalse(reduction_matches(means,scores+np.float32(.01),0))

    def test_duplicate_worker_lock_blocks_configure(self):
        import fcntl
        from scripts.corpus_control import idle
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            with (root/'group0.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                with self.assertRaisesRegex(ValueError,'lock'):idle(root)
            idle(root)

if __name__=='__main__':unittest.main()
