"""Tree NPZ export with original copied-baseline receipts and seed selection."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

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
                    self.assertEqual(set(data.files), {'prompt_id', 'method', 'accuracy', 'ttft'})
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

    def test_missing_or_corrupt_baseline_and_partial_tree_do_not_publish(self):
        for failure in ('missing-baseline-answer', 'missing-tree-answer', 'token-mismatch', 'corrupt-result', 'bad-copy'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                target, baseline = completed(root)
                folder = baseline[0]/'records/nocache/prompt-0/seed-0'
                if failure == 'missing-baseline-answer':
                    shutil.rmtree(folder)
                elif failure == 'missing-tree-answer':
                    shutil.rmtree(target[0]/'records/tree-prophetkv-20/prompt-0/seed-0')
                elif failure == 'token-mismatch':
                    shutil.rmtree(folder)
                    commit(baseline, 'nocache', token_hash='b'*64)
                elif failure == 'corrupt-result':
                    (folder/'result.json').write_text('{}')
                else:
                    (root/'baseline-source/settings.json').unlink()
                with self.assertRaises((ValueError, OSError)):
                    exporter.export(root, quiet=True)
                self.assertFalse((root/'tree-prophetkv-outcomes.npz').exists())

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
