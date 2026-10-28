"""Tree NPZ export with original copied-baseline receipts and seed selection."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from runner.setups import file_hash
from scripts import export_features as exporter
from test_baseline_breakdown import fixture, commit


def completed(root, dataset='ruler'):
    target, baseline = fixture(root, dataset)
    for stage in (target, baseline):
        for case in stage[1]['scheduled_actions']:
            for i in range(len(stage[2])):
                for seed in stage[1]['answer_seeds']:
                    commit(stage, case, i=i, seed=seed, score=i/2, ttft=seed+2.)
    return target, baseline


class TreeOutcomeExportTests(unittest.TestCase):
    def test_single_argument_cli_both_datasets_includes_matching_baseline_seeds(self):
        for dataset, seeds in (('ruler', 1), ('longbench-v2', 2)):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                completed(root, dataset)
                before = {p: (file_hash(p), p.stat().st_mtime_ns)
                          for p in root.rglob('*') if p.is_file()}
                result = subprocess.run([sys.executable, str(Path(exporter.__file__).resolve()), str(root)],
                                        cwd='/tmp', capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('nocache included', result.stdout)
                with np.load(root/'tree-prophetkv-outcomes.npz', allow_pickle=False) as data:
                    self.assertEqual(set(data.files), {'prompt_id', 'method', 'accuracy', 'ttft',
                                                       'thinking_tokens', 'answer_tokens'})
                    for field in ('thinking_tokens', 'answer_tokens'):
                        self.assertEqual(data[field].dtype, np.dtype('int64'))
                        for index, (pid, method) in enumerate(zip(data['prompt_id'], data['method'])):
                            prompt, seed = pid.rsplit('--seed-', 1)
                            stage = (root/'baseline-source'/('ruler' if dataset == 'ruler' else 'primary')
                                     if method == 'nocache' else root/'tree')
                            result = json.loads((stage/'records'/method/prompt/f'seed-{seed}'/'result.json').read_text())
                            self.assertEqual(data[field][index], result[field])
                    self.assertEqual(len(data['prompt_id']), 3*seeds*5)
                    self.assertEqual(set(data['method']),
                                     {'nocache', 'tree-prophetkv-5', 'tree-prophetkv-20',
                                      'tree-prophetkv-40', 'tree-prophetkv-60'})
                    for method in set(data['method']):
                        mask = data['method'] == method
                        self.assertEqual(set(data['prompt_id'][mask]),
                                         {f'prompt-{i}--seed-{s}' for i in range(3) for s in range(seeds)})
                        np.testing.assert_array_equal(data['accuracy'][mask],
                                                      [i/2 for i in range(3) for s in range(seeds)])
                        np.testing.assert_array_equal(data['ttft'][mask],
                                                      [s+2. for i in range(3) for s in range(seeds)])
                self.assertEqual(before, {p:(file_hash(p),p.stat().st_mtime_ns) for p in before})

    def test_corrupt_baseline_does_not_publish(self):
        for failure in ('token-mismatch', 'corrupt-result', 'bad-copy'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                target, baseline = completed(root)
                folder = baseline[0]/'records/nocache/prompt-0/seed-0'
                if failure == 'token-mismatch':
                    shutil.rmtree(folder)
                    commit(baseline, 'nocache', token_hash='b'*64)
                elif failure == 'corrupt-result':
                    (folder/'result.json').write_text('{}')
                else:
                    (root/'baseline-source/settings.json').unlink()
                with self.assertRaises((ValueError, OSError)):
                    exporter.export(root, quiet=True)
                self.assertFalse((root/'tree-prophetkv-outcomes.npz').exists())

    def test_partial_methods_and_baseline_export_every_committed_answer(self):
        for dataset in ('ruler', 'longbench-v2'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                target, baseline = fixture(root, dataset)
                # No prompt has a complete set of methods. No completion
                # receipts exist; baseline and tree have different membership.
                commit(baseline, 'nocache', i=0)
                commit(baseline, 'nocache', i=1)
                commit(target, 'tree-prophetkv-5', i=0, score=.5)
                commit(target, 'tree-prophetkv-20', i=2, score=0.)
                if dataset == 'longbench-v2':
                    commit(target, 'tree-prophetkv-40', i=1, seed=1)
                    commit(baseline, 'nocache', i=0, seed=3)  # Outside target seeds.
                # An answer without its commit receipt must not be exported.
                pending = target[0]/'records/tree-prophetkv-60/prompt-2/seed-0'
                pending.mkdir(parents=True)
                (pending/'result.json').write_text('{}')
                output = exporter.export(root, quiet=True)
                with np.load(output, allow_pickle=False) as data:
                    got = dict(zip(zip(data['method'], data['prompt_id']), data['accuracy']))
                    wanted = {('nocache', 'prompt-0--seed-0'): 1.,
                              ('nocache', 'prompt-1--seed-0'): 1.,
                              ('tree-prophetkv-5', 'prompt-0--seed-0'): .5,
                              ('tree-prophetkv-20', 'prompt-2--seed-0'): 0.}
                    if dataset == 'longbench-v2':
                        wanted[('tree-prophetkv-40', 'prompt-1--seed-1')] = 1.
                    self.assertEqual(got, wanted)
                    self.assertEqual(len(data['method']), len(wanted))

    def test_empty_collection_exports_empty_arrays_without_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            output = exporter.export(root, quiet=True)
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(set(data.files), {'prompt_id', 'method', 'accuracy', 'ttft',
                                                   'thinking_tokens', 'answer_tokens'})
                self.assertTrue(all(data[key].shape == (0,) for key in data.files))
                self.assertEqual(data['thinking_tokens'].dtype, np.dtype('int64'))
                self.assertEqual(data['answer_tokens'].dtype, np.dtype('int64'))

    def test_token_counts_are_preserved_and_invalid_counts_never_publish(self):
        from scripts import export_tree_outcomes as tree_exporter
        expected = [dict(prompt_id=json.dumps(['p',0]), subtask='task')]
        base = dict(expected[0], accuracy=.5, timings=dict(ttft_seconds=2.))
        records = {'nocache': [dict(base, thinking_tokens=13, answer_tokens=7)],
                   'tree-prophetkv-5': [dict(base, thinking_tokens=0, answer_tokens=19)]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(tree_exporter, 'snapshot', return_value=(expected,records)):
                path = tree_exporter.export(root, quiet=True)
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['thinking_tokens'], [13,0])
                np.testing.assert_array_equal(data['answer_tokens'], [7,19])
            for field in ('thinking_tokens', 'answer_tokens'):
                for value in (None, -1, 1.5, True):
                    with self.subTest(field=field, value=value):
                        invalid = dict(records['nocache'][0], **{field: value})
                        output = root/'invalid.npz'
                        with patch.object(tree_exporter, 'snapshot', return_value=(expected, {'nocache':[invalid]})):
                            with self.assertRaisesRegex(ValueError, 'token counts'):
                                tree_exporter.export(root, output, quiet=True)
                        self.assertFalse(output.exists())

    def test_output_override_no_overwrite_and_no_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            completed(root)
            shutil.rmtree(root/'baseline-source')
            output = root/'custom.npz'
            self.assertEqual(exporter.export(root, output, quiet=True), output)
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(len(data['method']), 12)
                self.assertNotIn('nocache', data['method'])
            before = output.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                exporter.export(root, output, quiet=True)
            self.assertEqual(output.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, 'end in .npz'):
                exporter.export(root, root/'bad.txt', quiet=True)


if __name__ == '__main__':
    unittest.main()
