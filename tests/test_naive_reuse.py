"""CPU-only full-cohort adoption and zero-repair execution contracts."""
import copy
import ast
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
from runner.naive_reuse import ACTION, inventory, select_without_scoring
from runner.setups import atomic_json, file_hash
from scripts import longbench_a800_control as control, longbench_a800_data as data
from scripts.longbench_a800_report import report
from scripts.ruler import TASKS, CAPS, EVALUATION_PROTOCOL

ROOT = Path(__file__).resolve().parents[1]
GPUS = [f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(10)]


def prepared(root, model, dataset):
    model.mkdir(); atomic_json(model/'config.json', {'test': True})
    digest = file_hash(model/'config.json')
    spec = (dict(samples=500, seed=42, tokenizer_hashes={'config.json': digest})
            if dataset == 'ruler' else dict(model_config_sha256=digest, tokenizer_files={'config.json': digest}))
    files = {}; manifest = []
    tasks = TASKS if dataset == 'ruler' else ['domain']
    for task in tasks:
        for i in range(500 if dataset == 'ruler' else 503):
            name = f'samples/{task}-{i:03d}.json'
            sample = dict(token_ids=list(range(383))+[1000+i], boundaries=[0,64,128,384], question_positions=[128],
                          thinking=dataset != 'ruler', max_output_tokens=CAPS[task] if dataset == 'ruler' else 16384,
                          model_config_sha256=digest, source_id=f'source-{i}',
                          source_metadata={'length': data.LENGTHS[i%3]}, task=task)
            if dataset == 'ruler': sample['evaluation_protocol'] = EVALUATION_PROTOCOL
            stamp_sample(sample, 'native-template'); atomic_json(root/name, sample)
            files[name] = file_hash(root/name)
            manifest.append(dict(id=f'{task}-{i:03d}', prepared=name, subtask=task,
                                 references=['A'], scoring='ruler_all' if dataset == 'ruler' else 'longbench_v2'))
    (root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    files['manifest.jsonl'] = file_hash(root/'manifest.jsonl')
    atomic_json(root/'preparation.json', dict(files=files, spec=spec))


class NaiveReuseTests(unittest.TestCase):
    def test_existing_full_cohorts_readonly_scopes_and_supervisor(self):
        for dataset in ('ruler', 'longbench-v2'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); source = root/'prepared'; model = root/'model'; base = root/'result'
                prepared(source, model, dataset)
                pins = {p: (file_hash(p), p.stat().st_mtime_ns) for p in source.rglob('*') if p.is_file()}
                role = 'ruler' if dataset == 'ruler' else 'primary'
                n = 10 if dataset == 'ruler' else 4
                args = NS(root=base, prepared=source, model=model, data=Path('/unused'), cache_root=root/'cache',
                          role=role, devices=','.join(map(str,range(n))), tp=2, seed=42, samples_per_task=500,
                          naive_reuse=True)
                with patch.object(control, 'code_hashes', return_value={}), patch.object(control.subprocess, 'check_output',
                        return_value='\n'.join(f'{i}, {g}' for i,g in enumerate(GPUS))):
                    control.configure(args)
                with patch('scripts.ruler.prepare', side_effect=AssertionError('No generation')), \
                     patch('scripts.longbench_v2.prepare', side_effect=AssertionError('No generation')):
                    plan = data.prepare(base)
                count = 6500 if dataset == 'ruler' else 503
                self.assertEqual((plan['samples'],plan['answers'],plan['probes']), (count,count,0))
                self.assertEqual(plan['actions'], [ACTION])
                self.assertEqual(plan['schedule'], {role:['naive-reuse']})
                protocol, rows = data.stage(base, role)
                self.assertEqual(inventory(protocol), {'naive-reuse': ACTION})
                self.assertEqual(len(protocol['groups']), n//2)
                if dataset == 'longbench-v2':
                    self.assertEqual(protocol['groups'], [GPUS[:2], GPUS[2:4]])
                counts = [sum(r['ordinal']%(n//2)==g for r in rows) for g in range(n//2)]
                self.assertEqual(counts, [1300]*5 if dataset == 'ruler' else [252,251])
                self.assertEqual(pins, {p: (file_hash(p),p.stat().st_mtime_ns) for p in pins})
                status = report(base, emit=False)
                self.assertEqual(status['expected_answers'], count)
                self.assertEqual(list(status['methods']), ['naive-reuse'])
                with patch('scripts.router_control.code_hashes', return_value={}), \
                     patch.object(control, 'execute_phase') as execute, \
                     patch.object(control, 'wait_extra', side_effect=AssertionError('No extra stage')), \
                     patch('scripts.longbench_a800_report.report') as final:
                    control.supervise(base, role)
                self.assertEqual([c.args[2] for c in execute.call_args_list], ['cached'])
                final.assert_called_once_with(base, final=True, emit=False)
                self.assertFalse(data.read(base/role/'complete.json')['feature_collection'])
                self.assertFalse((base/'features').exists())
                # Neither forged file hashes nor wrong model/profile may slip into a new plan.
                first = source/rows[0]['prepared']; original = first.read_bytes()
                first.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'Prepared artifact changed'):
                    data.prepare(base)
                first.write_bytes(original)
                args.root = source/'bad-result'
                with self.assertRaisesRegex(ValueError, 'separate'):
                    control.configure(args)

    def test_naive_inventory_does_not_expand_router_actions(self):
        from runner.tree_policy import actions
        with self.assertRaises(ValueError): actions([dict(id='nocache',method='baseline',ratio=None),ACTION])
        with self.assertRaises(ValueError): inventory(dict(naive_reuse=True,actions=[ACTION],scheduled_actions=['probe']))

    def test_zero_selection_loads_all_layers_without_scores(self):
        aligned=set()
        def load(name): aligned.add(name)
        sparse = NS(ratio=0., router_capture=False, model=NS(layers=[None]*64),
                    request=NS(request_id='test'), selection_diagnostics=[],
                    connector=NS(wait_for_layer_load=Mock(side_effect=load), prophet_aligned=aligned))
        fake_torch = NS(long='int64', empty=Mock(return_value=[]))
        with patch.dict(sys.modules, torch=fake_torch):
            self.assertEqual(select_without_scoring(sparse,'cpu'), [])
            self.assertEqual(sparse.connector.wait_for_layer_load.call_count,64)
            event=sparse.selection_diagnostics[0]
            self.assertEqual(event['alignment_count'],64)
            self.assertEqual(event['scoring_layers'],[])
            self.assertNotIn('scores',event)
            sparse.ratio=.01
            with self.assertRaises(ValueError): select_without_scoring(sparse,'cpu')

    def test_zero_ratio_worker_rpc_is_not_replaced_by_one_percent(self):
        from runner.corpus_worker import tree_mode
        sparse = NS(router_arrays=None,router_capture=False,model=NS(layers=[]),
                    connector=NS(store=NS(operations={})))
        configure=Mock(); barrier=Mock()
        modules={'runner.worker':NS(configure=configure),
                 'ucm.sparse.state':NS(get_ucm_sparse=lambda:sparse),
                 'vllm.distributed':NS(get_tensor_model_parallel_rank=lambda:0,get_tp_group=lambda:NS(barrier=barrier)),
                 'ucm.sparse.prophetkv.attention':NS(install=Mock())}
        with patch.dict(sys.modules,modules):
            tree_mode(None,'request:read',ACTION)
            configure.assert_called_once_with(None,'prophetkv',0.)
            self.assertTrue(sparse.naive_reuse)
            tree_mode(None,'request:read',dict(id='prophetkv-1',method='prophetkv',ratio=.01))
            self.assertFalse(sparse.naive_reuse)

    def test_real_prepare_step_dispatch_skips_probe_and_embeddings(self):
        # Execute the real method without importing CUDA/vLLM; stop immediately
        # after committing its selected set, before the shared tensor compaction.
        tree=ast.parse((ROOT/'ucm/sparse/prophetkv/prophetkv.py').read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='prepare_step')
        namespace={'probe':Mock(side_effect=AssertionError('Scoring called')),'torch':Mock()}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<prepare_step>','exec'),namespace)
        class Selected(Exception): pass
        sparse=NS(active=True,coverage_selected=None,naive_reuse=True,
                  prefill_state=NS(select_once=Mock(side_effect=Selected)))
        selection=NS(cpu=lambda:NS(tolist=lambda:[]))
        with patch('runner.naive_reuse.select_without_scoring',return_value=selection) as choose:
            with self.assertRaises(Selected): namespace['prepare_step'](sparse,NS(device='cpu'))
            choose.assert_called_once_with(sparse,'cpu')
        sparse.prefill_state.select_once.assert_called_once_with([])
        namespace['probe'].assert_not_called()
        self.assertEqual(namespace['torch'].mock_calls,[])

    def test_diagnostics_reject_context_repair_scoring_and_incomplete_alignment(self):
        from runner.generation import verify_diagnostics
        event=dict(kind='naive_reuse_selection',selected_positions=[],ratio=0.,scoring_layers=[],
                   probe_layers=0,suffix_forward_layers=0,alignment_count=64)
        events=[event]
        for start,end,positions in [(64,128,[]),(128,384,list(range(128,384)))]:
            events.append(dict(kind='prefill_step',start=start,end=end,scheduled_tokens=end-start,
                               recomputed_tokens=len(positions),no_forward=not positions,prefill_complete=end==384))
            events.extend(dict(kind='layer_counts',layer=f'model.layers.{i}.self_attn.attn',start=start,end=end,
                               selected_positions=positions,selected_set_verified=True,projection_tokens=len(positions),
                               attention_tokens=len(positions),ffn_tokens=len(positions)) for i in range(64))
        workers=[dict(rank=i,diagnostics=copy.deepcopy(events)) for i in range(2)]
        verify_diagnostics(workers,dict(boundaries=[0,64,128,384]),'naive_reuse',0.,2,range(64))
        from runner.corpus_records import accepted, publish, NATIVE_ANSWER_VALIDATION
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);atomic_json(root/'inputs/p.json',dict(boundaries=[0,64,128,384]))
            row=dict(id='p',ordinal=0,prepared='p.json',sha256=file_hash(root/'inputs/p.json'))
            protocol=dict(naive_reuse=True,tp=2,actions=[ACTION],scheduled_actions=['naive-reuse'],
                          prepared=str(root/'inputs'),groups=[GPUS[:2]],answer_validation=NATIVE_ANSWER_VALIDATION)
            atomic_json(root/'init.json',dict(validated=True,engine_config={}))
            record=dict(prompt_id='p',method='naive-reuse',executed_action='naive-reuse',input_sha256=row['sha256'],group=0,
                        gpu_uuids=GPUS[:2],cache_immutable=True,answer_validation=NATIVE_ANSWER_VALIDATION,
                        retirement=[dict(rank=r,quiescent=True,transfers=dict(pending=0),request_bookkeeping=0) for r in range(2)],
                        initialization='init.json',initialization_sha256=file_hash(root/'init.json'),accuracy=1.,timings=dict(ttft_seconds=.2))
            publish(root,'naive-reuse',row,protocol,record,workers)
            self.assertEqual(accepted(root,'naive-reuse',row,protocol,full=True),record)
        for key,value in [('selected_positions',[64]),('scoring_layers',[0]),('alignment_count',63),('probe_layers',64)]:
            broken=copy.deepcopy(workers);broken[0]['diagnostics'][0][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                verify_diagnostics(broken,dict(boundaries=[0,64,128,384]),'naive_reuse',0.,2,range(64))

    def test_launchers_forward_single_zero_control_and_existing_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            fake=Path(temp)/'python'; fake.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n');fake.chmod(0o755)
            env_file=Path(temp)/'env';env_file.write_text(f'PYTHON_BIN={fake}\nMODEL_PATH=/model\nEXPERIMENT_DIR=/new\nPREPARED_DIR=/old\nCACHE_ROOT=/cache\n')
            for launcher in ('l20_ruler_naive_reuse','a800_longbench_naive_reuse'):
                for command in ('configure','prepare','detach','resume','status','stop'):
                    out=subprocess.run(['bash',str(ROOT/f'scripts/launcher/{launcher}.sh'),command],cwd='/tmp',
                                       env=dict(PATH=os.environ['PATH'],UCM_ENV_FILE=str(env_file),GPU_EXTRA='4,5,6,7'),capture_output=True,text=True)
                    self.assertEqual(out.returncode,0,out.stderr)
                    args=json.loads(out.stdout);self.assertIn('--naive-reuse',args)
                    self.assertNotIn('--extend-from',args);self.assertNotIn('--thinking',args)
                    self.assertEqual(args[args.index('--prepared')+1],'/old')
                    if launcher == 'a800_longbench_naive_reuse':
                        self.assertEqual(args[args.index('--devices')+1],'0,1,2,3')


if __name__ == '__main__': unittest.main()
