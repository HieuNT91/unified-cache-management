"""Read-only L40 input adoption and L20 feature/trajectory scheduling on CPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from runner.layout import stamp_sample
from runner.setups import atomic_json, file_hash, fingerprint
from runner import output_budget as budget
from runner.thinking_budget import KEY, BudgetState
from runner.corpus_records import accepted, publish
from runner.last_traj_4 import FIXED_METHOD
from scripts import l20_ruler_traj4f_data as collection
from scripts import longbench_a800_control as control
from scripts.ruler import TASKS
from test_ruler_thinking import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
GPUS = [f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(10)]


def fixture(root, n=5):
    model = root/'model'; prepared = root/'prepared'
    atomic_json(model/'config.json', {'model_type': 'qwen3'})
    atomic_json(model/'tokenizer.json', {'test': True})
    spec = dict(samples=n, seed=42, protocol=budget.RULER_PROTOCOL, execution_profile='ruler-thinking',
        tasks=list(TASKS), tokenizer_hashes={name: file_hash(model/name) for name in ('config.json','tokenizer.json')})
    files = {}; manifest = []
    for task in TASKS:
        for i in range(n):
            name = f'samples/{task}-{i:03d}.json'
            sample = dict(token_ids=list(range(576)), boundaries=[0,64,320,576], question_positions=[320],
                thinking=True, max_output_tokens=16384, task=task, source_row=i,
                model_config_sha256=spec['tokenizer_hashes']['config.json'], generator_config_sha256=fingerprint(spec),
                execution_profile='ruler-thinking', evaluation_protocol=budget.RULER_PROTOCOL,
                **{KEY: budget.make_policy(Tokenizer(), ' Answer:')})
            stamp_sample(sample, 'native-template'); atomic_json(prepared/name, sample)
            files[name] = file_hash(prepared/name)
            manifest.append(dict(id=f'{task}-{i:03d}', prepared=name, subtask=task, references=['A'], scoring='ruler_all'))
    (prepared/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    files['manifest.jsonl'] = file_hash(prepared/'manifest.jsonl')
    atomic_json(prepared/'spec.json', spec)
    atomic_json(prepared/'preparation.json', dict(spec=spec, files=files))
    return NS(root=root/'result', prepared=prepared, model=model, cache_root=root/'cache',
              devices=','.join(GPUS), tp=2, seed=42, samples_per_task=n, percentages=list(collection.RATIOS))


class L20CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.args = fixture(Path(self.temp.name))
        with patch('scripts.router_control.code_hashes', return_value={'old.py':'provenance'}):
            collection.adopt(self.args)

    def test_inventory_phases_shards_and_fast_source_updates(self):
        settings, plan, rows = collection.load(self.args.root)
        self.assertEqual((plan['samples'], plan['probes'], plan['answers'], plan['baseline_answers']), (65,65,520,0))
        self.assertEqual(plan['phase_order'], ['features','cached'])
        self.assertEqual(settings['groups'], [GPUS[i:i+2] for i in range(0,10,2)])
        self.assertEqual([r['subtask'] for r in rows[:13]], list(TASKS))
        self.assertEqual([sum(r['ordinal']%5==g for r in rows) for g in range(5)], [13]*5)
        full_rows = [dict(id=f'{t}-{i}', subtask=t, ordinal=i) for t in TASKS for i in range(500)]
        collection.validate_rows(dict(settings, samples_per_task=500), full_rows)
        self.assertEqual(len(full_rows)*len(plan['actions']), 52000)
        methods, _ = collection.stage(self.args.root,'ruler')
        features, _ = collection.stage(self.args.root,'features')
        from runner.naive_reuse import inventory
        self.assertEqual(list(inventory(methods)), [f'last_traj_4f_{p}pct' for p in collection.RATIOS])
        self.assertEqual(features['scheduled_actions'], ['probe'])
        self.assertEqual(methods['position_attention_kernel'], collection.KERNEL)
        self.assertNotIn('priming_action', methods)
        pins = {p:(file_hash(p),p.stat().st_mtime_ns) for parent in (self.args.root,self.args.prepared)
                for p in parent.rglob('*') if p.is_file()}
        # Metadata-only resume: no sample scan, generator or source hash guard.
        original = collection.read
        def fast_read(path):
            if Path(path).parent.name == 'samples': raise AssertionError('Resume rescanned input')
            return original(path)
        with patch.object(collection,'read',side_effect=fast_read), \
                patch('scripts.ruler.prepare',side_effect=AssertionError('No prepare')), \
                patch('scripts.router_control.code_hashes',side_effect=AssertionError('No source pin')):
            collection.adopt(self.args)
        self.assertEqual(pins,{p:(file_hash(p),p.stat().st_mtime_ns) for p in pins})

    def test_adoption_rejects_input_model_or_scope_changes(self):
        for kind in ('input','tokenizer','count','overlap'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                args=fixture(Path(temp),n=1)
                if kind=='input': (args.prepared/f'samples/{TASKS[0]}-000.json').write_text('{}')
                elif kind=='tokenizer': (args.model/'tokenizer.json').write_text('{}')
                elif kind=='count': args.samples_per_task=2
                else: args.root=args.prepared/'results'
                with self.assertRaises(ValueError): collection.adopt(args)
                self.assertFalse((args.root/'plan.json').exists())

    def test_supervisor_uses_only_features_then_cached_and_no_completion_on_failure(self):
        for fail in (True,False):
            with self.subTest(fail=fail), patch.object(control,'execute_phase') as execute, \
                    patch.object(collection,'report') as report:
                if fail:
                    execute.side_effect=RuntimeError('read failed')
                    with self.assertRaisesRegex(RuntimeError,'read failed'): collection.supervise(self.args.root)
                    self.assertFalse((self.args.root/'ruler/complete.json').exists())
                    report.assert_not_called()
                else:
                    collection.supervise(self.args.root)
                    self.assertEqual([c.args[2] for c in execute.call_args_list], ['features','cached'])
                    for call in execute.call_args_list:
                        self.assertIs(call.kwargs['stage_loader'], collection.stage)
                        self.assertEqual(Path(call.kwargs['entrypoint']).resolve(), Path(collection.__file__).resolve())
                    self.assertEqual(collection.read(self.args.root/'ruler/complete.json')['baseline_answers'],0)

    def test_workers_request_only_probe_or_eight_methods_and_resume(self):
        for phase in ('features','cached'):
            protocol, rows=collection.stage(self.args.root,'features' if phase=='features' else 'ruler')
            rows=rows[:1]; engine=Mock()
            engine.answer.return_value=({},[]);engine.probe.return_value=({},[],{});engine.end.return_value={}
            cases=protocol['scheduled_actions']
            for finished in (False,True):
                engine.reset_mock()
                with patch('runner.corpus_runtime.Engine',return_value=engine) as ctor, \
                        patch('runner.setups.check_environment',return_value=GPUS[:2]), \
                        patch.object(control,'accepted',return_value={} if finished else None), \
                        patch.object(control,'publish') as publish_result:
                    control.worker(self.args.root,'ruler',phase,'attempt',0,stage_loader=lambda *_:(protocol,rows))
                if finished: ctor.assert_not_called(); continue
                engine.begin.assert_called_once_with(rows[0]);engine.end.assert_called_once();engine.close.assert_called_once()
                if phase=='features':
                    engine.answer.assert_not_called();engine.probe.assert_called_once()
                else:
                    engine.probe.assert_not_called()
                    self.assertEqual([c.args[0]['method'] for c in engine.answer.call_args_list], [FIXED_METHOD]*8)
                    self.assertEqual([c.args[1] for c in engine.answer.call_args_list], cases)
                self.assertEqual(publish_result.call_count,len(cases))

    def test_custom_controller_is_forwarded_to_detached_supervisor_and_workers(self):
        def inspect_helper(*args, **kwargs):
            payload=json.loads(kwargs['input'])
            self.assertEqual(Path(payload['command'][3]).resolve(),Path(collection.__file__).resolve())
            self.assertIn('supervise',payload['command'])
            self.assertEqual(payload['env']['CUDA_VISIBLE_DEVICES'],'')
            raise RuntimeError('inspected without spawning')
        with patch.object(control,'idle'), patch.object(control.subprocess,'run',side_effect=inspect_helper):
            with self.assertRaisesRegex(RuntimeError,'without spawning'):
                control.detach(self.args.root,'ruler',stage_loader=collection.stage,entrypoint=collection.__file__)
        children=[]
        def spawn(argv,**kwargs):
            self.assertEqual(Path(argv[2]).resolve(),Path(collection.__file__).resolve())
            self.assertEqual(argv[3],'worker')
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],','.join(GPUS[len(children)*2:len(children)*2+2]))
            child=Mock(pid=10000+len(children),returncode=0);child.poll.return_value=0
            children.append(child);return child
        # Initial pending check, one per TP group, then the completion check.
        with patch.object(control,'pending',side_effect=[['pending']]*6+[[]]), \
                patch.object(control,'engines_idle'), patch.object(control,'hardware',return_value={}), \
                patch.object(control,'cleanup_caches'), patch.object(control,'identity',return_value='synthetic'), \
                patch.object(control,'group_alive',return_value=False), patch.object(control.subprocess,'Popen',side_effect=spawn):
            control.execute_phase(self.args.root,'ruler','cached',{},run_loader=collection.load,
                stage_loader=collection.stage,entrypoint=collection.__file__)
        self.assertEqual(len(children),5)

    def test_thinking_trajectory_record_accepted_and_reported_without_baseline(self):
        from runner.reporting import OutputAnalyzer
        protocol,rows=collection.stage(self.args.root,'ruler'); row=rows[0];case=protocol['scheduled_actions'][0]
        root=self.args.root/'ruler';atomic_json(root/'init.json',dict(validated=True,engine_config={}))
        p=row[KEY];tokens=[65,p['close_id']]+p['transition_ids']+p['prefix_ids']+[65,3]
        record=dict(prompt_id=row['id'],method=case,executed_action=case,input_sha256=row['sha256'],group=0,
            gpu_uuids=GPUS[:2],cache_immutable=True,initialization='init.json',initialization_sha256=file_hash(root/'init.json'),
            answer_validation=protocol['answer_validation'],evaluation_protocol=budget.RULER_PROTOCOL,
            retirement=[dict(rank=r,quiescent=True,transfers={'pending':0},request_bookkeeping=0) for r in range(2)],
            accuracy=1.,subtask=row['subtask'],scoring='ruler_all',prediction='A',output_cap_reached=False,
            output_token_ids=tokens,max_output_tokens=16384,timings=dict(ttft_seconds=1.,first_answer_content_seconds=1.1),
            **{KEY:p})
        record.update(OutputAnalyzer(Tokenizer()).analyze(tokens,[],True),**BudgetState(p).consume(tokens).metrics(Tokenizer()))
        publish(root,case,row,protocol,record,[])
        self.assertEqual(accepted(root,case,row,protocol),record)
        result=collection.report(self.args.root)
        self.assertEqual((result['available_answers'],result['baseline_answers']),(1,0))
        with self.assertRaisesRegex(ValueError,'incomplete'): collection.report(self.args.root,final=True)
        self.assertFalse((self.args.root/'final.json').exists())

    def test_launcher_loads_env_and_forwards_paths_ratios_and_ten_uuids(self):
        fake=Path(self.temp.name)/'python'
        fake.write_text(f'#!{sys.executable}\nimport json,sys,os\nprint(json.dumps([sys.argv[1:],os.environ["CUDA_VISIBLE_DEVICES"]]))\n')
        fake.chmod(0o755)
        env=Path(self.temp.name)/'env'
        env.write_text(f'PYTHON_BIN="{fake}"\nPREPARED_DIR="/existing inputs"\nGPU_DEVICES="{self.args.devices}"\n')
        for command in ('start','resume','status','stop'):
            out=subprocess.run(['bash',str(ROOT/'scripts/launcher/l20_ruler_thinking_traj4f.sh'),command],cwd='/tmp',
                env=dict(PATH=os.environ['PATH'],UCM_ENV_FILE=str(env)),capture_output=True,text=True)
            self.assertEqual(out.returncode,0,out.stderr)
            argv,devices=json.loads(out.stdout);self.assertEqual(devices,'')
            self.assertEqual(argv[1],command);self.assertEqual(argv[argv.index('--prepared')+1],'/existing inputs')
            self.assertEqual(argv[argv.index('--devices')+1],self.args.devices)
            self.assertEqual(argv[argv.index('--percentages')+1:],[str(p) for p in collection.RATIOS])


if __name__ == '__main__': unittest.main()
