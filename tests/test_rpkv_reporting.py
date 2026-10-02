import json
from pathlib import Path
import tempfile
import unittest
from scripts.rpkv_gpu_results import expected_scope,validate_rows
from scripts.ruler import TASKS


class CohortTests(unittest.TestCase):
    def test_expanded_counts_and_missing_duplicate_wrong_task(self):
        scope=dict(ruler_samples_per_task=40,longbench_samples=20,seed=42)
        rows=[dict(id=f'{t}-{i:03d}',subtask=t) for t in TASKS for i in range(40)]
        self.assertEqual(validate_rows('ruler',rows,scope),520)
        self.assertEqual(validate_rows('longbench-v2',[dict(id=str(i)) for i in range(20)],scope),20)
        for bad in (rows[:-1],rows[:-1]+[rows[0]],rows[:-1]+[dict(id='wrong-task',subtask='invalid')]):
            with self.assertRaises(ValueError):validate_rows('ruler',bad,scope)
        with self.assertRaises(ValueError):validate_rows('longbench-v2',[dict(id=str(i)) for i in range(19)],scope)

    def test_legacy_scope_and_explicit_invalid_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(expected_scope(root)['ruler_samples_per_task'],5)
            for value in (0,-1,True,4.5):
                (root/'experiment.json').write_text(json.dumps(dict(ruler_samples_per_task=value,longbench_samples=20)))
                with self.assertRaises(ValueError):expected_scope(root)
            (root/'experiment.json').write_text(json.dumps(dict(ruler_samples_per_task=40,longbench_samples=20,seed=42)))
            self.assertEqual(expected_scope(root)['longbench_samples'],20)


if __name__=='__main__':unittest.main()
