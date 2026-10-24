"""Copied no-cache comparison uses real committed receipts, without prepared inputs."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from runner.corpus_records import publish, record_dir
from runner.reporting import OutputAnalyzer
from runner.setups import atomic_json, file_hash, fingerprint
from runner.thinking_budget import KEY, BudgetState
from runner.output_budget import make_policy, RULER_PROTOCOL, LONGBENCH_PROTOCOL
from runner.tree_collection import SCHEMA, actions
from scripts.collection_v3 import declaration
from scripts import experiment_task_breakdown as report
from test_ruler_thinking import Tokenizer


def fixture(base, dataset='ruler', change=None):
    policy = make_policy(Tokenizer(), ' Answer:' if dataset == 'ruler' else '')
    evaluation = RULER_PROTOCOL if dataset == 'ruler' else LONGBENCH_PROTOCOL
    rows = [dict(id=f'prompt-{i}', ordinal=i+3, subtask=f'task-{i%2}', sha256=fingerprint(i),
                 references=['A'], scoring='ruler_all' if dataset == 'ruler' else 'longbench_v2',
                 evaluation_protocol=evaluation, **{KEY: policy}) for i in range(3)]
    outputs = []
    for baseline in (False, True):
        root = base/'baseline-source' if baseline else base
        role = ('ruler' if dataset == 'ruler' else 'primary') if baseline else 'tree'
        definitions = [dict(id='nocache', method='baseline', ratio=None)] if baseline else actions([5,20,40,60])
        n = 1 if dataset == 'ruler' else 4 if baseline else 2
        groups = [[f'GPU-{g}-{r}' for r in range(2)] for g in range(4 if baseline else 5)]
        settings = dict(schema='tp2-data-config-v3' if baseline else SCHEMA, dataset=dataset, tp=2,
                        groups=groups, model='/unavailable/model', prepared='/unavailable/prepared',
                        **declaration(dataset, n, thinking=True))
        if dataset == 'ruler': settings['execution_profile'] = 'ruler-thinking'
        items = copy.deepcopy(rows)
        if baseline and change: change(settings, items)
        atomic_json(root/'settings.json', settings); atomic_json(root/'rows.json', items)
        plan = dict(settings_sha256=file_hash(root/'settings.json'), rows_sha256=file_hash(root/'rows.json'))
        atomic_json(root/'plan.json', plan)
        protocol = dict(settings, actions=definitions, scheduled_actions=[a['id'] for a in definitions],
                        answer_validation='native-answer-diagnostics-v1', plan_sha256=fingerprint(plan))
        atomic_json(root/role/'protocol.json', protocol)
        atomic_json(root/role/'sessions/init/initialization.json', dict(validated=True, engine_config={}))
        outputs.append((root/role, protocol, items))
    return outputs


def commit(stage, case, i=0, seed=0, score=1., ttft=2., token_hash='a'*64):
    root, protocol, rows = stage; row = rows[i]; p = row[KEY]
    tokens = [65,p['close_id']]+p['transition_ids']+p['prefix_ids']+[65,3]
    group = row['ordinal'] % len(protocol['groups'])
    record = dict(prompt_id=row['id'], sequence_id=row['id'], seed=seed, method=case, executed_action=case,
        input_sha256=row['sha256'], token_sha256=token_hash, prompt_protocol='rpkv-original-tokens-v1',
        evaluation_protocol=row['evaluation_protocol'], group=group, gpu_uuids=protocol['groups'][group],
        cache_immutable=True, initialization='sessions/init/initialization.json',
        initialization_sha256=file_hash(root/'sessions/init/initialization.json'),
        answer_validation=protocol['answer_validation'], num_cached_tokens=0 if case=='nocache' else 320,
        retirement=[dict(rank=r,quiescent=True,transfers={'pending':0},request_bookkeeping=0) for r in range(2)],
        accuracy=score, subtask=row['subtask'], scoring=row['scoring'], prediction='A', output_cap_reached=False,
        output_token_ids=tokens, max_output_tokens=16384,
        timings=dict(ttft_seconds=ttft,first_answer_content_seconds=ttft+.1), **{KEY:p})
    record.update(OutputAnalyzer(Tokenizer()).analyze(tokens,[],True),
                  **BudgetState(p).consume(tokens).metrics(Tokenizer()))
    publish(root,case,row,protocol,record,[],seed=seed)


class BaselineBreakdownTests(unittest.TestCase):
    def test_both_datasets_original_receipts_exact_intersection_and_seed_subset(self):
        for dataset in ('ruler','longbench-v2'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temp:
                base=Path(temp); target, baseline=fixture(base,dataset)
                for seed in baseline[1]['answer_seeds']:
                    commit(baseline,'nocache',seed=seed,ttft=2. if seed==0 else 100.)
                commit(baseline,'nocache',i=1,score=0.,ttft=1000.)
                for case in target[1]['scheduled_actions']:
                    commit(target,case)
                commit(target,'tree-prophetkv-5',i=1,score=0.,ttft=1000.)
                pins={p:(file_hash(p),p.stat().st_mtime_ns) for p in base.rglob('*') if p.is_file()}
                text,tables,matching=report.report(base)
                self.assertEqual(matching['matched_samples'],1)
                self.assertEqual(matching['available_counts']['nocache'],2 if dataset=='ruler' else 3)
                for value in tables['overall'].values():
                    self.assertEqual(value['completed'],1)
                    self.assertEqual(value['accuracy_percent'],100.)
                    self.assertEqual(value['mean_ttft_seconds'],2.)
                self.assertIn('exact shared prompt/seed pairs only',text)
                self.assertIn('original hardware provenance',text)
                for name in ('experiment_task_breakdown.py','experiment_same_count.py'):
                    result=subprocess.run([sys.executable,str(Path(report.__file__).with_name(name)),str(base)],
                                          cwd='/tmp',capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertIn('| nocache | 1/',result.stdout)
                self.assertEqual(pins,{p:(file_hash(p),p.stat().st_mtime_ns) for p in pins})

    def test_mismatched_source_identity_rejected_even_with_equal_ids_and_counts(self):
        for field in ('sha256','subtask','references','evaluation_protocol',KEY):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temp:
                def change(settings,rows): rows[0][field]='changed'
                fixture(Path(temp),change=change)
                with self.assertRaisesRegex(ValueError,'Baseline input/policy differs'):
                    report.report(Path(temp))
        with tempfile.TemporaryDirectory() as temp:
            fixture(Path(temp),'longbench-v2',lambda s,r:s.update(answer_seeds=[0],answers_per_question=1))
            with self.assertRaisesRegex(ValueError,'seed schedule'): report.report(Path(temp))

    def test_corrupt_or_different_tokens_never_report_as_baseline(self):
        for failure in ('tokens','result','initialization','protocol'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as temp:
                base=Path(temp); target,baseline=fixture(base)
                commit(baseline,'nocache',token_hash='b'*64 if failure=='tokens' else 'a'*64)
                for case in target[1]['scheduled_actions']: commit(target,case)
                folder=record_dir(baseline[0],'nocache',baseline[2][0]['id'],0)
                if failure=='result': (folder/'result.json').write_text('{}')
                if failure=='initialization': (baseline[0]/'sessions/init/initialization.json').unlink()
                if failure=='protocol':
                    protocol=report.read(baseline[0]/'protocol.json'); protocol['code']={'modified':True}
                    atomic_json(baseline[0]/'protocol.json',protocol)
                with self.assertRaises(ValueError): report.report(base)

    def test_no_shared_results_is_na_and_partial_copy_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp); target,baseline=fixture(base)
            commit(baseline,'nocache'); commit(target,'tree-prophetkv-5')
            _,tables,matching=report.report(base)
            self.assertEqual(matching['matched_samples'],0)
            for value in tables['overall'].values():
                self.assertEqual(value['completed'],0); self.assertIsNone(value['accuracy_percent'])
            (base/'baseline-source/settings.json').unlink()
            with self.assertRaises(OSError): report.report(base)


if __name__=='__main__': unittest.main()
