"""TP2 resource, replay, scheduling and late-extra contracts (CPU only)."""
import copy
import json
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import numpy as np

from scripts import longbench_a800_control as control
from scripts import longbench_a800_data as data
from runner.corpus import load_attention, reduction_matches, selection
from runner.longbench_features import features, HEAD_LAYERS
from runner.setups import atomic_json, file_hash
from test_a800_longbench import fixture


class TP2Tests(unittest.TestCase):
    def test_sharding_all_methods_and_tasks(self):
        lb = [[i for i in range(503) if i % 2 == group] for group in range(2)]
        self.assertEqual(list(map(len, lb)), [252, 251])
        self.assertEqual(sum(len(s)*len(data.ACTIONS) for s in lb), 6036)
        from scripts.ruler import TASKS
        ruler = [[(t,i) for t in TASKS for i in range(100) if i % 5 == group] for group in range(5)]
        self.assertEqual(list(map(len, ruler)), [260]*5)
        self.assertEqual(len(set(sum(ruler, []))), 1300)
        self.assertEqual(sum(len(s)*len(data.ACTIONS) for s in ruler), 15600)
        self.assertFalse(set(data.SCHEDULE['primary']) & set(data.SCHEDULE['extra']))
        self.assertEqual(set(sum((data.SCHEDULE[r] for r in ('primary','extra')), [])), {a['id'] for a in data.ACTIONS})

    def test_primary_only_configure_then_extra_preserves_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            args = NS(root=base, role='primary', model=Path('/m'), data=Path('/d'), prepared=base/'p',
                      cache_root=base/'c', tp=2, samples_per_task=100, seed=42, devices='0,1,2,3')
            uuids = [f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(8)]
            # Extra devices are absent during primary configure.
            raw = '\n'.join(f'{i}, {u}' for i,u in enumerate(uuids[:4]))
            with patch.object(control.subprocess, 'check_output', return_value=raw), patch.object(control, 'code_hashes', return_value={}):
                control.configure(args)
            atomic_json(base/'plan.json', {'frozen': True})
            data.publish_protocols(base)
            paths = ['settings.json','plan.json','primary/devices.json','primary/protocol.json','features/protocol.json']
            pins = {n: (file_hash(base/n), (base/n).stat().st_mtime_ns) for n in paths}
            args.role='extra'; args.devices='4,5,6,7'
            with patch.object(control.subprocess, 'check_output', return_value='\n'.join(f'{i}, {u}' for i,u in enumerate(uuids))), patch.object(control, 'code_hashes', return_value={}):
                control.configure(args)
            self.assertEqual(pins, {n: (file_hash(base/n), (base/n).stat().st_mtime_ns) for n in paths})
            self.assertEqual(data.read(base/'extra/devices.json')['groups'], [uuids[4:6],uuids[6:]])

    def test_unconfigured_extra_is_visible_and_primary_has_own_match(self):
        from scripts.longbench_a800_report import report
        from test_a800_longbench import record
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); fixture(base)
            (base/'extra/protocol.json').unlink(); (base/'extra/devices.json').unlink()
            _, _, rows=data.load(base)
            records={a['id']: ([record(rows[0],a['id'])] if a['id'] in data.SCHEDULE['primary'] else []) for a in data.ACTIONS}
            with patch('scripts.longbench_a800_report.records', return_value=records):
                result=report(base, True, emit=False)
            self.assertEqual(result['matching']['matched_samples'],0)
            self.assertEqual(result['primary_matched']['matching']['matched_samples'],1)
            self.assertEqual(len(result['methods']),12)
            self.assertEqual(result['waiting']['extra'], dict(configured=False,missing_answers=503*5))

    def test_tp2_archive_exact_reduction_heads_and_tp4_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); end=104; prefix=4
            sample=dict(boundaries=[0,prefix,end,360],question_positions=[105],token_ids=list(range(360)))
            layers=np.ones((64,end),np.float32); heads=np.ones((8,32,end),np.float32)
            heads[0,0,prefix]=0
            scores=np.ones(end-prefix,np.float32); artifacts=[]
            for rank in range(2):
                path=root/f'{rank}.npz'
                arrays=dict(layers=layers, heads=heads, head_layers=HEAD_LAYERS, local_mean=layers[0],scores=scores,
                    context_positions=np.arange(end),question_positions=sample['question_positions'],boundaries=sample['boundaries'],original_to_formatted=sample['token_ids'])
                arrays.update({a['id']:selection(scores,prefix,a['ratio']) for a in data.ACTIONS if a['ratio'] is not None})
                np.savez(path,**arrays);artifacts.append(dict(rank=rank,path=path.name,sha256=file_hash(path)))
            att=load_attention(root,dict(artifacts=artifacts),sample,data.ACTIONS,HEAD_LAYERS,tp=2)
            expected=features(att,prefix,end)
            # Same 64 physical heads represented as four ranks yields equal coverage.
            tp4=dict(att,layers=np.repeat(att['layers'],2,axis=0),heads=att['heads'].transpose(1,0,2,3).reshape(8,4,16,end).transpose(1,0,2,3))
            self.assertEqual(features(tp4,prefix,end),expected)
            with self.assertRaisesRegex(ValueError,'rank'):load_attention(root,dict(artifacts=artifacts[:1]),sample,data.ACTIONS,HEAD_LAYERS,tp=2)
            bad=dict(att,heads=att['heads'][:,:,:16])
            with self.assertRaises(ValueError):features(bad,prefix,end)
            self.assertTrue(reduction_matches([layers[0],layers[0]],scores,prefix))
            self.assertFalse(reduction_matches([layers[0],layers[0]],scores*2,prefix))

    def test_tp2_config_and_init_reject_missing_or_duplicate_rank(self):
        from runner.tree_profiles import config,validate_initialization,allocation
        for dataset in ('ruler','longbench-v2'):
            cfg=config('/m',dataset,True,'/c',tp=2)
            self.assertEqual(cfg['tensor_parallel_size'],2)
            self.assertEqual(cfg['dtype'],'bfloat16')
            p=allocation(dataset)
            receipts=[dict(rank=r,kv_tokens=p['kv_tokens'],kv_blocks=p['kv_blocks'],block_size=64,
                           rope=[dict(formula_bitwise_equal=True,table_positions=131072)]*64) for r in range(2)]
            validate_initialization(receipts,dataset,tp=2)
            with self.assertRaises(ValueError):validate_initialization(receipts[:1],dataset,tp=2)
            with self.assertRaises(ValueError):validate_initialization([receipts[0]]*2,dataset,tp=2)

    def test_server_tp2_auto_cache_preserves_execution_limits(self):
        from runner.tree_profiles import config,allocation
        for dataset,hardware in [('longbench-v2','server'),('ruler','l20-tp2')]:
            for cached in (False,True):
                with self.subTest(dataset=dataset,cached=cached):
                    cfg=config('/m',dataset,cached,'/c',hardware=hardware,tp=2)
                    self.assertNotIn('num_gpu_blocks_override',cfg)
                    self.assertEqual(cfg['gpu_memory_utilization'],.95)
                    self.assertEqual(cfg['max_model_len'],allocation(dataset)['kv_tokens'])
                    self.assertEqual(cfg['dtype'],'bfloat16')
                    self.assertEqual(cfg['max_num_seqs'],1)
                    self.assertEqual(cfg['block_size'],64)
                    self.assertEqual(cfg['hf_overrides']['ucm_activation_tile'],4096)
                    self.assertEqual('kv_transfer_config' in cfg,cached)

    def test_auto_cache_capacity_and_rope_checks_on_every_rank(self):
        from runner.tree_profiles import validate_initialization,allocation
        for dataset,hardware in [('longbench-v2','server'),('ruler','l20-tp2')]:
            p=allocation(dataset)
            receipts=[dict(rank=r,kv_tokens=p['kv_tokens'],kv_blocks=p['kv_blocks']+500,block_size=64,
                           rope=[dict(formula_bitwise_equal=True,table_positions=131072) for _ in range(64)]) for r in range(2)]
            validate_initialization(receipts,dataset,hardware,tp=2)
            for rank in range(2):
                for value in (p['kv_blocks']-1,p['kv_blocks']+499,None,True,1.5):
                    bad=copy.deepcopy(receipts);bad[rank]['kv_blocks']=value
                    with self.subTest(dataset=dataset,rank=rank,blocks=value):
                        with self.assertRaisesRegex(ValueError,'capacity'):validate_initialization(bad,dataset,hardware,tp=2)
                for key,value in [('kv_tokens',p['kv_tokens']+64),('block_size',128)]:
                    bad=copy.deepcopy(receipts);bad[rank][key]=value
                    with self.assertRaises(ValueError):validate_initialization(bad,dataset,hardware,tp=2)
                bad=copy.deepcopy(receipts);bad[rank]['rope'][63]['formula_bitwise_equal']=False
                with self.assertRaises(ValueError):validate_initialization(bad,dataset,hardware,tp=2)

    def test_tp4_and_local_cache_allocations_remain_fixed(self):
        from runner.tree_profiles import config,allocation,validate_initialization
        for dataset,hardware in [('longbench-v2','server'),('ruler','server'),('ruler','rtx4500ada')]:
            p=allocation(dataset,hardware)
            cfg=config('/m',dataset,True,'/c',hardware=hardware)
            self.assertEqual(cfg['num_gpu_blocks_override'],p['kv_blocks'])
            self.assertEqual(cfg['gpu_memory_utilization'],.9 if hardware=='server' else .95)
            receipts=[dict(rank=r,kv_tokens=p['kv_tokens'],kv_blocks=p['kv_blocks'],block_size=64,
                           rope=[dict(formula_bitwise_equal=True,table_positions=131072)]*64) for r in range(4)]
            validate_initialization(receipts,dataset,hardware)
            for receipt in receipts:receipt['kv_blocks']+=1
            with self.assertRaisesRegex(ValueError,'capacity'):validate_initialization(receipts,dataset,hardware)

    def test_tp2_memory_gate_counts_half_weights_and_kv_per_rank(self):
        from scripts.corpus_control import check_hardware
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); atomic_json(root/'model.safetensors.index.json',dict(metadata=dict(total_size=64*2**30)))
            settings=dict(dataset='ruler',model=str(root),tp=2,groups=[['a','b']])
            inventory='a,NVIDIA L20,100000,100000\nb,NVIDIA L20,100000,100000'
            with patch('scripts.corpus_control.subprocess.check_output',side_effect=[inventory,'']):
                receipt=check_hardware(settings)
            expected=(32*2**30+65920*262144/2+8*2**30)/.95/2**20
            self.assertAlmostEqual(receipt['required_mib_per_rank'],expected)
            self.assertEqual(receipt['kv_cache_mode'],'auto')
            self.assertTrue(receipt['kv_bytes_are_minimum'])
            with patch('scripts.corpus_control.subprocess.check_output',return_value=inventory.replace('100000','40000')):
                with self.assertRaisesRegex(ValueError,'cannot fit'):check_hardware(settings)

    def test_auto_cache_requires_budget_free_on_both_servers(self):
        from scripts.corpus_control import check_hardware
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);atomic_json(root/'model.safetensors.index.json',dict(metadata=dict(total_size=61.2*2**30)))
            for dataset,hardware,name,total in [('longbench-v2','server','A800',81920),('ruler','l20-tp2','L20',49152)]:
                settings=dict(dataset=dataset,hardware_profile=hardware,model=str(root),tp=2,groups=[['a','b']])
                inventory=f'a,NVIDIA {name},{total},{total*.95}\nb,NVIDIA {name},{total},{total*.95}'
                with patch('scripts.corpus_control.subprocess.check_output',side_effect=[inventory,'']):
                    self.assertEqual(check_hardware(settings)['gpu_memory_utilization'],.95)
                # Both devices can fit the minimum, but rank1 cannot provide the
                # requested automatic budget. Refuse without launching an engine.
                low=f'a,NVIDIA {name},{total},{total}\nb,NVIDIA {name},{total},{total*.94}'
                with patch('scripts.corpus_control.subprocess.check_output',return_value=low):
                    with self.assertRaisesRegex(ValueError,'automatic KV budget'):check_hardware(settings)
                with patch('scripts.corpus_control.subprocess.check_output',side_effect=[inventory,'b']):
                    with self.assertRaisesRegex(ValueError,'occupied'):check_hardware(settings)

    def test_ruler_prepares_100_seed42_and_schedules_five_complete_shards(self):
        self.check_ruler_preparation(100)

    def test_ruler_prepares_200_seed42_and_schedules_five_complete_shards(self):
        self.check_ruler_preparation(200)

    def check_ruler_preparation(self, count):
        from scripts.ruler import TASKS
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); prepared=base/'inputs'; prepared.mkdir()
            settings=dict(dataset='ruler',tp=2,model='/m',data='/ruler',prepared=str(prepared),
                          cache_root=str(base/'cache'),samples_per_task=count,seed=42,code={})
            atomic_json(base/'settings.json',settings)
            atomic_json(base/'ruler/devices.json',dict(groups=[[str(i),str(i+1)] for i in range(0,10,2)]))
            rows=[]; files={}
            for task in TASKS:
                for ordinal in range(count):
                    path=f'samples/{task}-{ordinal:03d}.json'
                    atomic_json(prepared/path,dict(token_ids=[ordinal],evaluation_protocol='test'))
                    files[path]=file_hash(prepared/path)
                    rows.append(dict(id=f'{task}-{ordinal:03d}',subtask=task,prepared=path))
            manifest=prepared/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            files['manifest.jsonl']=file_hash(manifest);atomic_json(prepared/'preparation.json',dict(files=files))
            with patch('scripts.ruler.prepare') as prepare:
                plan=data.prepare(base)
            self.assertEqual((prepare.call_args.args[0].samples,prepare.call_args.args[0].seed),(count,42))
            self.assertEqual((plan['answers'],plan['probes']),(13*count*12,13*count))
            protocol,rows=data.stage(base,'ruler')
            self.assertEqual(len(protocol['groups']),5)
            self.assertEqual(protocol['scheduled_actions'],[a['id'] for a in data.ACTIONS])
            for group in range(5):
                subset=[r for r in rows if r['ordinal']%5==group]
                self.assertEqual(len(subset),13*count//5)
                self.assertTrue(all(sum(r['subtask']==t for r in subset)==count//5 for t in TASKS))

    def test_ruler_configure_pins_sample_count_and_rejects_scope_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=NS(root=root,role='ruler',model=Path('/m'),data=Path('/ruler'),prepared=root/'inputs',
                    cache_root=root/'cache',tp=2,samples_per_task=200,seed=42,devices='0,1,2,3,4,5,6,7,8,9')
            groups=[[str(i),str(i+1)] for i in range(0,10,2)]
            with patch.object(control,'resolve_groups',return_value=groups),patch.object(control,'code_hashes',return_value={}):
                control.configure(args)
                self.assertEqual(data.read(root/'settings.json')['samples_per_task'],200)
                pin=file_hash(root/'settings.json')
                args.samples_per_task=100
                with self.assertRaisesRegex(ValueError,'Frozen metadata changed'):control.configure(args)
                self.assertEqual(file_hash(root/'settings.json'),pin)
                args.samples_per_task=150
                with self.assertRaisesRegex(ValueError,'100 or 200'):control.configure(args)

    def test_engine_tp2_warmup_requires_four_shards_and_retirement_two_ranks(self):
        from runner.corpus_runtime import Engine
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);prepared=base/'inputs'
            sample=dict(token_ids=[1]*512,max_output_tokens=128,boundaries=[0,64,256,512])
            atomic_json(prepared/'sample.json',sample)
            protocol=dict(tp=2,groups=[['a','b']],dataset='ruler',model='/m',prepared=str(prepared),cache_root=str(base/'cache'))
            llm=Mock();llm.collective_rpc.return_value=[]
            with ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, {'runner.worker': NS(setup=Mock())}))
                for name in ('validate','validate_initialization'):
                    stack.enter_context(patch('runner.corpus_runtime.'+name))
                stack.enter_context(patch('runner.corpus_runtime.start_engine',return_value=llm))
                stack.enter_context(patch('runner.reporting.OutputAnalyzer'))
                stack.enter_context(patch('runner.generation.generate'))
                stack.enter_context(patch('ucm.sparse.prophetkv.lifecycle.delete_retired_files'))
                ready=stack.enter_context(patch('runner.cache.wait_for_cache',return_value=dict(verified_shards=4)))
                retire=stack.enter_context(patch('runner.corpus_runtime.retirement'))
                engine=Engine(base/'results',protocol,0,'cached','ok',[dict(id='p',prepared='sample.json')])
                self.assertEqual(ready.call_args.args[0].tensor_parallel_size,2)
                retire.assert_called_once_with(llm,2)
                engine.close()
                ready.return_value=dict(verified_shards=2)
                with self.assertRaisesRegex(ValueError,'warmup shards'):
                    Engine(base/'results',protocol,0,'cached','missing',[dict(id='p',prepared='sample.json')])

    def test_failed_worker_terminates_all_owned_peers_and_preserves_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); (base/'primary').mkdir()
            saved=base/'accepted.json';saved.write_text('untouched')
            protocol=dict(groups=[['a','b'],['c','d']],tp=2)
            processes=[Mock(pid=101,returncode=1),Mock(pid=102,returncode=-15)]
            processes[0].poll.return_value=1;processes[1].poll.return_value=None
            calls=iter(processes)
            def launch(*args,**kwargs):
                kwargs['stdout'].write('Traceback (most recent call last):\nRuntimeError: synthetic worker failure\n')
                kwargs['stdout'].flush()
                return next(calls)
            state={}
            with patch.object(control,'load',return_value=(dict(dataset='longbench-v2',watchdog_seconds=7200),{},[])), \
                 patch.object(control,'stage',return_value=(protocol,[dict(id='p0',ordinal=0),dict(id='p1',ordinal=1)])), \
                 patch.object(control,'pending',side_effect=lambda p,r,*a:r),patch.object(control,'engines_idle'), \
                 patch.object(control,'hardware',return_value={}),patch.object(control,'cleanup_caches'), \
                 patch.object(control.subprocess,'Popen',side_effect=launch),patch.object(control,'identity',return_value={}), \
                 patch.object(control,'group_alive',return_value=False),patch.object(control,'terminate_owned') as kill:
                with self.assertRaisesRegex(RuntimeError,'synthetic worker failure') as caught:
                    control.execute_phase(base,'primary','cached',state)
            self.assertEqual(kill.call_args.args[0]['pid'],102)
            self.assertEqual(saved.read_text(),'untouched')
            self.assertEqual(len(list((base/'primary/exits').glob('*.json'))),2)
            failed=state['failed_worker']
            self.assertEqual((failed['pid'],failed['group'],failed['exit_code']), (101,0,1))
            self.assertEqual(failed['gpu_uuids'],['a','b'])
            self.assertTrue(Path(failed['log']).is_file())
            self.assertIn(failed['log'],str(caught.exception))

    def test_worker_saves_failure_before_shutdown_and_keeps_original_error(self):
        from runner.corpus_runtime import Engine
        for fail_begin in (True, False):
            for secondary_failures in (False, True):
                with self.subTest(begin=fail_begin,secondary=secondary_failures), tempfile.TemporaryDirectory() as tmp:
                    base=Path(tmp);(base/'ruler').mkdir()
                    protocol=dict(tp=2,groups=[['a','b']],scheduled_actions=['prophetkv-1'])
                    row=dict(id='p0',ordinal=0)
                    engine=Mock(spec=Engine)
                    original=RuntimeError('cache inventory failed' if fail_begin else 'Incomplete external cache load')
                    (engine.begin if fail_begin else engine.answer).side_effect=original
                    order=[]
                    def save(*args):
                        order.append('receipt')
                        if secondary_failures:raise OSError('disk full')
                        return base/'failure.json'
                    def close():
                        order.append('shutdown')
                        if secondary_failures:raise RuntimeError('engine already dead')
                    engine.record_failure.side_effect=save;engine.close.side_effect=close
                    with patch.object(control,'stage',return_value=(protocol,[row])), \
                         patch.object(control,'pending',return_value=[row]), \
                         patch.object(control,'accepted',return_value=None), \
                         patch('runner.setups.check_environment',return_value=['a','b']), \
                         patch('runner.corpus_runtime.Engine',return_value=engine), \
                         patch.object(control,'publish') as publish:
                        with self.assertRaises(RuntimeError) as caught:
                            control.worker(base,'ruler','cached','attempt',0)
                    self.assertIs(caught.exception,original)
                    self.assertEqual(order,['receipt','shutdown'])
                    engine.record_failure.assert_called_once_with(original,'cached',None if fail_begin else 'prophetkv-1')
                    publish.assert_not_called()

    def test_failure_inventory_survives_cache_cleanup_and_records_missing_extra(self):
        from runner.corpus_runtime import Engine
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);cache=base/'cache';session=base/'sessions'/'attempt'
            (cache/'kv').mkdir(parents=True)
            (cache/'kv'/'expected').write_bytes(b'kv')
            (cache/'kv'/'unexpected').write_bytes(b'other')
            engine=Engine.__new__(Engine)
            engine.group=3;engine.row=dict(id='p0');engine.namespace='a'*32
            engine.session=session;engine.cache=cache;engine.cached=True
            engine.pc=NS(files={'kv/expected','kv/missing'})
            try:raise RuntimeError('Incomplete external cache load')
            except RuntimeError as error:path=engine.record_failure(error,'cached','prophetkv-50')
            receipt=json.loads(path.read_text());inventory=receipt['inventory']
            self.assertEqual(receipt['case'],'prophetkv-50')
            self.assertEqual(receipt['prompt_id'],'p0')
            self.assertIn('RuntimeError: Incomplete external cache load',receipt['traceback'])
            self.assertEqual(inventory['extra'],['kv/unexpected'])
            self.assertEqual(inventory['missing'],['kv/missing'])
            self.assertEqual(inventory['actual_files']['kv/expected']['bytes'],2)
            self.assertFalse(inventory['quiescence_verified'])
            self.assertEqual((cache/'kv'/'unexpected').read_bytes(),b'other')
            shutil.rmtree(cache)
            self.assertEqual(json.loads(path.read_text()),receipt)

    def test_owned_cleanup_after_root_relocation_preserves_archived_cache_and_results(self):
        from runner.setups import fingerprint
        from scripts.router_control import cleanup_caches
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'ruler';cache=base/'cache'
            root.mkdir();cache.mkdir()
            accepted_path=root/'accepted.json';accepted_path.write_text('accepted')
            saved=file_hash(accepted_path)
            anchor=cache/fingerprint(str(root))[:16]/'group3'
            old=anchor/('a'*32);old.mkdir(parents=True)
            (old/'evidence').write_text('keep')
            atomic_json(root/'sessions/old/ownership.json',dict(cache=str(old),group=3,pid=100,identity=None))
            backup=base/'hdd-backup';cache.rename(backup)
            target=base/'ssd-cache';target.mkdir();cache.symlink_to(target,target_is_directory=True)
            new=anchor/('b'*32);new.mkdir(parents=True)
            (new/'temporary').write_text('retired')
            atomic_json(root/'sessions/new/ownership.json',dict(cache=str(new),group=3,pid=101,identity=None))
            with patch('scripts.router_control.alive',return_value=False), \
                 patch('scripts.router_control.group_alive',return_value=False):
                self.assertEqual(cleanup_caches(root,dict(cache_root=str(cache))),[str(new)])
            self.assertEqual((backup/old.relative_to(cache)/'evidence').read_text(),'keep')
            self.assertFalse(new.exists())
            self.assertEqual(file_hash(accepted_path),saved)

    def test_worker_failure_reads_bounded_tail_and_handles_unavailable_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'worker with spaces.log'
            receipt=dict(pid=42,phase='cached',group=4,log=str(path),gpu_uuids=['GPU-one','GPU-two'])
            path.write_bytes(b'not-in-tail\n'+b'x'*40000+b'\n'+b'line\n'*90+b'RuntimeError: actual worker error\xff\n')
            message=str(control.worker_failure(receipt,1))
            self.assertIn('cached group4 failed (1)',message)
            self.assertIn('actual worker error',message)
            self.assertIn('GPU-one, GPU-two',message)
            self.assertNotIn('not-in-tail',message)
            self.assertLess(len(message),4000)
            path.write_text('')
            self.assertIn('Worker log is empty',str(control.worker_failure(receipt,1)))
            path.unlink()
            self.assertIn('Could not read worker log',str(control.worker_failure(receipt,1)))

    def test_watchdog_message_keeps_log_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'worker.log';path.write_text('Waiting for committed shards\n')
            receipt=dict(pid=42,phase='cached',group=4,log=str(path),gpu_uuids=['a','b'])
            message=str(control.worker_failure(receipt,None,reason='Progress watchdog expired for group4'))
            self.assertIn('Progress watchdog expired for group4',message)
            self.assertIn('Waiting for committed shards',message)

if __name__=='__main__':unittest.main()
