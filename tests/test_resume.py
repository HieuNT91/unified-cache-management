from runner.layout import stamp_sample
"""Continuation preserves committed answers, rejects drift, and avoids reruns."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from runner import resume, sweep
from runner.config import VERSIONS, engine_config
from runner.setups import atomic_json, file_hash, fingerprint
from scripts.sweep_report import collect
from scripts import sweep_control


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root/'out'; self.model = self.root/'model'; self.cache = self.root/'cache'
        self.model.mkdir(); self.cache.mkdir(); self.output.mkdir()
        atomic_json(self.model/'config.json', dict(model_type='qwen3',num_hidden_layers=64,
            hidden_size=5120,num_attention_heads=40,num_key_value_heads=8,head_dim=1))
        self.sample = stamp_sample(dict(token_ids=[1]*63+[99]+[2]*63+[99]+[3]*256,
            boundaries=[0,64,128,384],question_positions=[380],thinking=True,max_output_tokens=2,
            model_config_sha256=file_hash(self.model/'config.json')))
        self.manifest = self.root/'manifest.jsonl'
        rows = []
        for i in range(4):
            path = self.root/f'{i}.json'; atomic_json(path, self.sample)
            rows.append(dict(id=str(i),prepared=str(path),subtask='task',references=['B'],scoring='choice'))
        self.manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.args = NS(model=self.model,manifest=self.manifest,output=self.output,cache_root=self.cache,
                       tp=2,shards=2,shard=0,memory=.9,percentages=[1,5],layers=[11,12,13,14,15],
                       context_length=114688,exact_input_tokens=None,dry_run=False,resume=True,skip_selective=True,
                       validation='full')
        self.configs = sweep.configurations([1,5]); self.entries = {}
        for shard in (0,1):
            self.args.shard = shard
            _, entries, identity = sweep.load_inputs(self.args)
            self.entries.update({i:(entry,sample) for i,entry,sample in entries})
            legacy = json.loads((resume.ROOT/'runner/legacy_sweep_runtimes.json').read_text())[-1]
            legacy_identity = fingerprint(dict(self.args.identity_spec, runtime=legacy['runtime']))
            cache = self.cache/('sweep-'+str(shard)*32); cache.mkdir(); (cache/'old').write_text('cache')
            group = self.output/f'group-{shard}'
            atomic_json(group/'plan.json',dict(identity=legacy_identity,gpu_uuids=[f'GPU-{shard}a',f'GPU-{shard}b'],
                cache=str(cache),prompt_ids=[e['id'] for _,e,_ in entries],configurations=self.configs,shard=shard,shards=2))
            atomic_json(group/'baseline-session.json',dict(engine_shutdown=True))
            atomic_json(group/'baseline-config.json',engine_config(self.model,'baseline',tp=2))
        atomic_json(self.output/'stop-receipt.json',dict(output=str(self.output),processes={}))
        for i in self.entries:
            self.write_record(i, self.configs[0])
        self.write_record(0, self.configs[1])
        self.write_record(2, self.configs[1])
        self.write_record(2, self.configs[2])  # prompt 2 needs no work at all
        selective = self.output/'selective-1/000000/result.json'
        atomic_json(selective,dict(preserved='selective partial'))
        self.original = {str(p.relative_to(self.output)):file_hash(p)
                         for p in self.output.rglob('*.json')}

    def diagnostics(self, config):
        baseline = config['method'] == 'baseline'
        positions = list(range(64,64+int(64*(config['ratio'] or 0))))+list(range(128,384))
        start = 0 if baseline else 64
        events = [dict(kind='prefill_step',start=start,end=384,scheduled_tokens=384-start,
                       prefill_complete=True,no_forward=False,recomputed_tokens=384 if baseline else len(positions))]
        if not baseline:
            events += [dict(kind='prophetkv_selection',scores=[1.]*64,selected_positions=positions[:-256],
                            scoring_layers=list(range(64)),fusion='mean_layers_fp32',alignment_count=64)]
            events += [dict(kind='layer_counts',start=64,end=384,layer=f'model.layers.{i}.self_attn.attn',
                            selected_set_verified=True,selected_positions=positions,
                            projection_tokens=len(positions),attention_tokens=len(positions),ffn_tokens=len(positions))
                       for i in range(64)]
        return [dict(rank=i,diagnostics=events) for i in range(2)]

    def write_record(self, index, config, cohort=None):
        entry, sample = self.entries[index]
        retired = [dict(rank=i,quiescent=True,transfers={'pending':0},request_bookkeeping=0) for i in range(2)]
        record = dict(prompt_id=entry['id'],input_sha256=entry['sha256'],model=str(self.model),method=config['method'],
            ratio=config['ratio'],scoring_layers=[] if config['method']=='baseline' else list(range(64)),
            prompt_tokens=384,thinking=True,max_output_tokens=2,runtime_versions=VERSIONS,
            gpu_uuids=[f'GPU-{index%2}a',f'GPU-{index%2}b'],retirement=retired,
            output_token_ids=[97,10],num_cached_tokens=0,thinking_tokens=0,answer_tokens=1,control_tokens=1,
            output_tokens=2,output_cap_reached=True,unfinished_thinking=False,accuracy=1.,
            timings=dict(ttft_seconds=.2,generation_seconds=.4),**entry['evaluation'])
        if cohort:
            record['continuation_fingerprint'] = cohort
        folder = self.output/config['name']/f'{index:06d}'
        atomic_json(folder/'diagnostics.json',self.diagnostics(config)); atomic_json(folder/'result.json',record)

    def test_prepare_recovers_committed_records_preserves_originals_and_reports_scope(self):
        incomplete = self.output/'prophetkv-5/000000/diagnostics.json'
        atomic_json(incomplete,dict(interrupted=True))
        scope = resume.prepare(self.args)
        self.assertEqual(scope['pending'],{'baseline':[],'prophetkv-1':[1,3],'prophetkv-5':[0,1,3]})
        for name,digest in self.original.items():
            self.assertEqual(file_hash(self.output/name),digest)
        self.assertFalse(incomplete.exists())
        self.assertTrue((self.output/scope['attempt']/'incomplete/prophetkv-5/000000/diagnostics.json').exists())
        self.assertFalse(list(self.cache.iterdir()))
        live = collect(self.output,self.manifest,[1,5])
        self.assertEqual(live['expected'],12); self.assertEqual(live['completed'],7)
        self.assertEqual(live['discontinued'],['selective-1','selective-5'])
        with self.assertRaises(FileNotFoundError): collect(self.output,self.manifest,[1,5],True)
        # A second interruption retains new committed results even without ledger acceptance.
        self.write_record(0,self.configs[2],scope['fingerprint'])
        again = resume.prepare(self.args)
        self.assertNotEqual(scope['attempt'],again['attempt'])
        self.assertEqual(again['pending']['prophetkv-5'],[1,3])
        self.assertEqual(again['fingerprint'],scope['fingerprint'])

    def test_drop_ratio_migrate_both_shards_and_resume_mixed_device_results(self):
        from scripts.sweep_counts import saved_counts
        old=resume.prepare(self.args)
        devices=['GPU-00000000-0000-0000-0000-000000000001',
                 'GPU-00000000-0000-0000-0000-000000000002']
        self.args.exclude_percentages=[5]
        self.args.single_group_gpus=','.join(devices)
        scope=resume.prepare(self.args)
        self.assertEqual(scope['gpu_uuids'],[devices,devices])
        self.assertEqual(scope['execution_mode'],'sequential')
        self.assertEqual(scope['pending'],{'baseline':[],'prophetkv-1':[1,3]})
        self.assertEqual(scope['expected'],8)
        self.assertIn('prophetkv-5',scope['discontinued'])
        self.assertIn(old['fingerprint'],scope['cohort_gpu_uuids'])
        for name,digest in self.original.items():self.assertEqual(file_hash(self.output/name),digest)
        live=collect(self.output,self.manifest,[1,5])
        matched=collect(self.output,self.manifest,[1,5],same_count=True)
        self.assertEqual((live['completed'],live['expected']),(6,8))
        self.assertEqual(matched['matching']['prompt_ids'],['0','2'])
        self.assertEqual(set(live['methods']),{'baseline','prophetkv-1'})
        counts=saved_counts(self.output,self.manifest,[1,5])
        self.assertEqual((counts['saved'],counts['expected']),(6,8))
        self.assertEqual(counts['methods']['prophetkv-5'],1)
        # Finish a missing odd-shard result on the surviving (formerly other) group.
        self.write_record(1,self.configs[1],scope['fingerprint'])
        path=self.output/'prophetkv-1/000001/result.json'
        value=json.loads(path.read_text());value['gpu_uuids']=devices;atomic_json(path,value)
        self.args.exclude_percentages=None;self.args.single_group_gpus=None
        again=resume.prepare(self.args)
        self.assertEqual(again['pending']['prophetkv-1'],[3])
        self.assertEqual(again['excluded_percentages'],[5])
        self.assertEqual(again['execution_mode'],'sequential')
        self.write_record(3,self.configs[1],again['fingerprint'])
        path=self.output/'prophetkv-1/000003/result.json'
        value=json.loads(path.read_text());value['gpu_uuids']=devices;atomic_json(path,value)
        finished=resume.prepare(self.args)
        for shard in (0,1):
            self.args.shard=shard
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':','.join(devices)}), \
                    patch.object(sweep,'check_environment',return_value=devices), \
                    patch.object(sweep,'start_engine',side_effect=AssertionError('Completed results rerun')):
                sweep.run_sweep(self.args)
            if shard==0:
                with self.assertRaises(FileNotFoundError):collect(self.output,self.manifest,[1,5],True)
        final=collect(self.output,self.manifest,[1,5],True)
        self.assertEqual((final['completed'],final['expected']),(8,8))
        self.assertEqual(collect(self.output,self.manifest,[1,5],same_count=True)['completed'],8)
        for name,digest in self.original.items():self.assertEqual(file_hash(self.output/name),digest)

    def test_single_group_lock_and_explicit_device_validation(self):
        import fcntl
        for bad in ('0,1','GPU-short,GPU-short'):
            self.args.single_group_gpus=bad
            with self.assertRaisesRegex(ValueError,'full GPU UUIDs'):resume.prepare(self.args)
            self.assertFalse((self.output/'continuation.json').exists())
        self.args.single_group_gpus='GPU-00000000-0000-0000-0000-000000000001,GPU-00000000-0000-0000-0000-000000000002'
        scope=resume.prepare(self.args)
        with (self.output/scope['attempt']/'single-group.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):sweep.run_sweep(self.args)
        self.args.shard=1
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-1a,GPU-1b'}):
            with self.assertRaisesRegex(RuntimeError,'GPU UUIDs'):
                resume.attach(self.args,scope['current_identity'])

    def test_reject_unknown_or_all_excluded_ratios_before_mutation(self):
        for excluded in ([20],[1,5]):
            self.args.exclude_percentages=excluded
            with self.assertRaisesRegex(ValueError,'Exclude only original ratios'):resume.prepare(self.args)
            self.assertFalse((self.output/'continuation.json').exists())
        for name,digest in self.original.items():self.assertEqual(file_hash(self.output/name),digest)

    def test_changes_or_unretired_results_rejected_before_mutation(self):
        cases = [('input_sha256','wrong'),('ratio',.4),('gpu_uuids',['GPU-other']),
                 ('retirement',[dict(rank=0,quiescent=False)])]
        path=self.output/'prophetkv-1/000000/result.json'; original=path.read_bytes()
        for key,value in cases:
            with self.subTest(key=key):
                obj=json.loads(original); obj[key]=value; atomic_json(path,obj)
                with self.assertRaises((RuntimeError,KeyError)):resume.prepare(self.args)
                self.assertFalse((self.output/'continuation.json').exists())
                self.assertTrue(list(self.cache.iterdir()))
                path.write_bytes(original)
        original_manifest=self.manifest.read_text()
        self.manifest.write_text(original_manifest+'\n')
        with self.assertRaisesRegex(RuntimeError,'fingerprint'):resume.prepare(self.args)

    def test_previous_retained_hash_drift_is_rejected(self):
        resume.prepare(self.args)
        path=self.output/'baseline/000000/result.json'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(RuntimeError,'artifact changed'):resume.prepare(self.args)

    def test_resume_preserves_original_vendor_hashes_and_repeats(self):
        released=next(row for row in json.loads((resume.ROOT/'runner/legacy_sweep_runtimes.json').read_text())
                      if row['commit'].startswith('501fc70'))
        vendor={'ucm/vendor/RULER/scripts/data/synthetic/niah.py':'vendor-digest',
                'ucm/.cache/vendor/RULER/scripts/data/synthetic/niah.py':'vendor-digest'}
        original=fingerprint(dict(self.args.identity_spec,runtime={**released['runtime'],**vendor}))
        for shard in (0,1):
            path=self.output/f'group-{shard}/plan.json'
            plan=json.loads(path.read_text());plan['identity']=original;atomic_json(path,plan)
        preserved={str(p.relative_to(self.output)):file_hash(p) for p in self.output.rglob('*.json')}
        runtime={**sweep.runtime_identity(),**vendor}
        self.args.validation='fast'
        with patch.object(sweep,'runtime_identity',return_value=runtime):
            scope=resume.prepare(self.args)
            self.assertEqual(scope['source_revision'],released['commit']+' with preserved ucm/vendor files')
            again=resume.prepare(self.args)
            self.assertEqual(scope['pending'],again['pending'])
        for name,digest in preserved.items():
            self.assertEqual(file_hash(self.output/name),digest)

    def test_vendor_compatibility_rejects_changed_missing_or_extra_files_and_inputs(self):
        released=next(row for row in json.loads((resume.ROOT/'runner/legacy_sweep_runtimes.json').read_text())
                      if row['commit'].startswith('501fc70'))
        name='ucm/vendor/RULER/generator.py'
        spec=copy.deepcopy(self.args.identity_spec)
        spec['runtime'][name]='original-vendor'
        original=fingerprint(dict(spec,runtime={**released['runtime'],name:'original-vendor'}))
        self.assertIn('preserved ucm/vendor',resume.compatible_identity(spec,original))
        cases=[]
        changed=copy.deepcopy(spec);changed['runtime'][name]='changed';cases.append(changed)
        missing=copy.deepcopy(spec);del missing['runtime'][name];cases.append(missing)
        extra=copy.deepcopy(spec);extra['runtime']['ucm/vendor/RULER/extra.py']='extra';cases.append(extra)
        inputs=copy.deepcopy(spec);inputs['inputs'][0]='changed';cases.append(inputs)
        manifest=copy.deepcopy(spec);manifest['manifest']='changed';cases.append(manifest)
        for index,candidate in enumerate(cases):
            with self.subTest(case=index),self.assertRaisesRegex(RuntimeError,'fingerprint'):
                resume.compatible_identity(candidate,original)

    def test_vendor_compatibility_does_not_admit_arbitrary_local_runtime_files(self):
        released=json.loads((resume.ROOT/'runner/legacy_sweep_runtimes.json').read_text())[-1]
        spec=copy.deepcopy(self.args.identity_spec)
        spec['runtime']['ucm/local.py']='unreviewed'
        original=fingerprint(dict(spec,runtime={**released['runtime'],'ucm/local.py':'unreviewed'}))
        with self.assertRaisesRegex(RuntimeError,'fingerprint'):
            resume.compatible_identity(spec,original)

    def test_hidden_vendor_layout_requires_exact_hashes_for_each_copy(self):
        released=json.loads((resume.ROOT/'runner/legacy_sweep_runtimes.json').read_text())[-1]
        hidden='ucm/.cache/vendor/RULER/generator.py'
        visible='ucm/vendor/RULER/generator.py'
        for paths in ((hidden,), (visible,hidden)):
            with self.subTest(paths=paths):
                spec=copy.deepcopy(self.args.identity_spec)
                vendor=dict.fromkeys(paths,'same-original-content')
                spec['runtime'].update(vendor)
                original=fingerprint(dict(spec,runtime={**released['runtime'],**vendor}))
                self.assertIn('preserved ucm/vendor',resume.compatible_identity(spec,original))
                for name in paths:
                    changed=copy.deepcopy(spec);changed['runtime'][name]='modified'
                    with self.assertRaisesRegex(RuntimeError,'fingerprint'):
                        resume.compatible_identity(changed,original)
                    missing=copy.deepcopy(spec);del missing['runtime'][name]
                    with self.assertRaisesRegex(RuntimeError,'fingerprint'):
                        resume.compatible_identity(missing,original)

    def test_identity_failure_reports_candidates_without_touching_saved_results(self):
        self.manifest.write_text(self.manifest.read_text()+'\n')
        with self.assertRaisesRegex(RuntimeError, 'resume-compatibility.json'):
            resume.prepare(self.args)
        report=json.loads((self.output/'resume-compatibility.json').read_text())
        self.assertEqual(report['current_fingerprint'],fingerprint(report['current_spec']))
        self.assertEqual(report['original_fingerprint'],
                         json.loads((self.output/'group-0/plan.json').read_text())['identity'])
        self.assertTrue(report['candidates'])
        self.assertTrue(any(row['commit'].startswith('45a3f3a') for row in report['candidates']))
        self.assertTrue(all(row['fingerprint'] != report['original_fingerprint']
                            for row in report['candidates']))
        for name,digest in self.original.items():
            self.assertEqual(file_hash(self.output/name),digest)
        self.assertFalse((self.output/'continuation').exists())
        self.assertTrue(list(self.cache.iterdir()))

    def test_counts_precede_input_loading_and_exclude_partial_directories(self):
        import io
        from contextlib import redirect_stdout
        from scripts.sweep_counts import saved_counts
        partial=self.output/'prophetkv-5/000001/diagnostics.json'
        atomic_json(partial, {'incomplete':True})
        with redirect_stdout(io.StringIO()) as stream:
            counts=saved_counts(self.output,self.manifest,[1,5])
        self.assertEqual(counts['saved'],7)
        self.assertEqual(counts['expected'],12)
        self.assertEqual(counts['complete_prompts'],1)
        self.assertEqual(counts['methods']['selective-1'],1)
        self.assertIn('Group 0: baseline 2/2, ProphetKV 3/4 saved.',stream.getvalue())
        with redirect_stdout(io.StringIO()) as stream, \
                patch.object(sweep,'load_inputs',side_effect=RuntimeError('input check reached')):
            with self.assertRaisesRegex(RuntimeError,'input check reached'):
                resume.prepare(self.args)
        self.assertIn('Baseline + ProphetKV: 7/12 saved; 5 remaining.',stream.getvalue())
        self.assertLess(stream.getvalue().index('7/12 saved'),stream.getvalue().index('Checking prepared inputs'))

    def test_fast_mode_never_reads_or_hashes_saved_diagnostics(self):
        # Fast resume relies on original per-request validation, even when a
        # diagnostic file is now unreadable as JSON. It still requires presence.
        resume.prepare(self.args)
        path=self.output/'prophetkv-1/000000/diagnostics.json'
        path.write_text('deliberately invalid diagnostics for the fast-mode regression')
        self.args.validation='fast'
        real_read=Path.read_text
        def read(path,*args,**kwargs):
            if path.name=='diagnostics.json':
                raise AssertionError('Fast mode read a diagnostic payload')
            return real_read(path,*args,**kwargs)
        def digest(path):
            if Path(path).name=='diagnostics.json':
                raise AssertionError('Fast mode hashed a diagnostic payload')
            return file_hash(path)
        with patch.object(Path,'read_text',read),patch.object(resume,'file_hash',side_effect=digest), \
                patch('run.verify_diagnostics',side_effect=AssertionError('Unexpected saved replay')):
            scope=resume.prepare(self.args)
        self.assertEqual(scope['validation']['mode'],'fast')
        self.assertFalse(scope['validation']['diagnostic_replay'])
        report=collect(self.output,self.manifest,[1,5])
        self.assertEqual(report['resume_validation']['mode'],'fast')
        self.assertIn('not replayed or rehashed',(self.output/'live_summary.md').read_text())
        # Fast mode preserves prior hashes, so a later full audit catches drift.
        self.args.validation='full'
        with self.assertRaisesRegex(RuntimeError,'artifact changed'):resume.prepare(self.args)

    def test_fast_mode_keeps_result_metadata_hash_and_diagnostic_presence_checks(self):
        self.args.validation='fast'
        diag=self.output/'prophetkv-1/000000/diagnostics.json'
        content=diag.read_bytes();diag.unlink()
        with self.assertRaisesRegex(RuntimeError,'Missing retained diagnostics'):resume.prepare(self.args)
        diag.write_bytes(content)
        record=self.output/'prophetkv-1/000000/result.json'
        content=record.read_bytes();data=json.loads(content);data['input_sha256']='wrong'
        atomic_json(record,data)
        with self.assertRaisesRegex(RuntimeError,'incompatible metadata'):resume.prepare(self.args)
        record.write_bytes(content)
        scope=resume.prepare(self.args)
        self.assertNotIn(str(diag.relative_to(self.output)),scope['retained_files'])
        record.write_bytes(content+b'\n')
        with self.assertRaisesRegex(RuntimeError,'artifact changed'):resume.prepare(self.args)

    def test_fast_upgrade_accepts_old_continuation_cohort_and_repeated_resume(self):
        scope=resume.prepare(self.args)
        # Emulate the released resume runtime (also used by the .env release).
        runtime=next(row for row in json.loads((resume.ROOT/'runner/legacy_sweep_runtimes.json').read_text())
                     if row['commit'].startswith('16c1f75'))
        old_current=fingerprint(dict(self.args.identity_spec,runtime=runtime['runtime']))
        old_fingerprint=fingerprint(dict(original=scope['original_identity'],current=old_current,
                                         configs=scope['configurations']))
        scope.update(current_identity=old_current,fingerprint=old_fingerprint,
                     accepted_fingerprints=[old_fingerprint])
        atomic_json(self.output/'continuation.json',scope)
        self.write_record(1,self.configs[1],old_fingerprint)
        self.args.validation='fast'
        upgraded=resume.prepare(self.args)
        self.assertEqual(upgraded['previous_source_revision'],runtime['commit'])
        self.assertIn(old_fingerprint,upgraded['accepted_fingerprints'])
        self.assertEqual(upgraded['pending']['prophetkv-1'],[3])
        self.write_record(3,self.configs[1],upgraded['fingerprint'])
        repeated=resume.prepare(self.args)
        self.assertEqual(repeated['pending']['prophetkv-1'],[])
        self.assertIn(old_fingerprint,repeated['accepted_fingerprints'])
        self.assertIn(upgraded['fingerprint'],repeated['accepted_fingerprints'])

    def test_diagnostic_replay_and_engine_configuration_are_required(self):
        path=self.output/'prophetkv-1/000000/diagnostics.json'
        original=path.read_bytes(); value=json.loads(original); value.pop()
        atomic_json(path,value)
        with self.assertRaisesRegex(RuntimeError,'Missing tensor-parallel rank'):resume.prepare(self.args)
        path.write_bytes(original)
        self.args.memory=.8
        with self.assertRaisesRegex(RuntimeError,'engine configuration'):resume.prepare(self.args)

    def test_resume_lock_blocks_duplicate_preparation(self):
        import fcntl
        with (self.output/'sweep.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):resume.prepare(self.args)
        self.assertFalse((self.output/'continuation.json').exists())

    def test_cpu_cli_can_borrow_its_coordinators_lock(self):
        command=['bash','-c',
                 'exec 9>"$1/sweep.lock"; flock -n 9 || exit 1; '
                 'export UCM_SWEEP_COORDINATOR_PID=$$; shift; "$@"; rc=$?; exit "$rc"',
                 'coordinator',str(self.output),sys.executable,'-m','runner.resume',
                 '--model',str(self.model),'--manifest',str(self.manifest),'--output',str(self.output),
                 '--cache-root',str(self.cache),'--tp','2','--percentages','1','5']
        result=subprocess.run(command,cwd=resume.ROOT,capture_output=True,text=True,
                              env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertTrue((self.output/'continuation.json').exists())

    def test_attach_rejects_gpu_or_runtime_drift_and_dry_run_reports_missing(self):
        scope=resume.prepare(self.args)
        self.args.shard=0
        _,_,current=sweep.load_inputs(self.args)
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-other,GPU-wrong'}):
            with self.assertRaisesRegex(RuntimeError,'GPU UUIDs'):resume.attach(self.args,current)
        with self.assertRaisesRegex(RuntimeError,'Prepare the continuation'):resume.attach(self.args,'changed')
        self.args.dry_run=True
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()) as stream:sweep.run_sweep(self.args)
        dry=json.loads(stream.getvalue())
        self.assertEqual(dry['pending_measurements'],1)
        self.assertEqual(len(dry['configs']),3)
        self.assertNotIn('kv_transfer_config',dry['baseline'])
        self.assertFalse((self.output/scope['attempt']/'group-0').exists())

    def test_completed_baseline_and_stopped_processes_required(self):
        (self.output/'baseline/000003/result.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'Baseline must be complete'):resume.prepare(self.args)
        (self.output/'stop-receipt.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'sweep_control'):resume.prepare(self.args)

    def test_attached_resume_skips_engines_for_complete_shards_and_final_requires_both(self):
        scope=resume.prepare(self.args)
        for index in self.entries:
            for config in self.configs[1:3]:
                if not (self.output/config['name']/f'{index:06d}/result.json').exists():
                    self.write_record(index,config,scope['fingerprint'])
        scope=resume.prepare(self.args)
        for shard in (0,1):
            self.args.shard=shard
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':','.join(scope['gpu_uuids'][shard])}), \
                    patch.object(sweep,'check_environment',return_value=scope['gpu_uuids'][shard]), \
                    patch.object(sweep,'start_engine',side_effect=AssertionError('Unexpected engine')):
                sweep.run_sweep(self.args)
            if shard==0:
                with self.assertRaises(FileNotFoundError):collect(self.output,self.manifest,[1,5],True)
        final=collect(self.output,self.manifest,[1,5],True)
        self.assertEqual(final['completed'],12)
        for name,digest in self.original.items():self.assertEqual(file_hash(self.output/name),digest)

    def test_stop_target_discovery_excludes_other_experiments(self):
        table={100:dict(command=['python','-m','runner.sweep','--output',str(self.output)],parent=1,start='1'),
               101:dict(command=['engine'],parent=100,start='2'),
               102:dict(command=['python','-m','runner.sweep','--output',str(self.root/'unrelated')],parent=1,start='3')}
        found=sweep_control.owned_processes(self.output,table)
        self.assertEqual(set(found),{100,101})
        with patch.object(sweep_control,'process_table',return_value=table):
            atomic_json(self.output/'stop-receipt.json',dict(output=str(self.output),processes={'100':table[100]}))
            with self.assertRaisesRegex(RuntimeError,'still alive'):sweep_control.assert_stopped(self.output)

    def test_stop_real_cpu_child_preserves_unrelated_child(self):
        # Ordinary sleeping CPU subprocesses, no model or CUDA workload.
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)',
                                'runner.sweep','--output',str(self.output)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'])
        self.addCleanup(lambda: (unrelated.kill(),unrelated.wait()) if unrelated.poll() is None else None)
        self.addCleanup(lambda: (child.kill(),child.wait()) if child.poll() is None else None)
        sweep_control.stop(self.output)
        child.wait(timeout=5)
        self.assertIsNone(unrelated.poll())
        sweep_control.assert_stopped(self.output)


if __name__=='__main__':unittest.main()
