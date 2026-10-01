"""Final success criterion on a complete synthetic paired 260-row cohort."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from runner.gpu_study import report,sealed

class StudyReportTests(unittest.TestCase):
    def run_report(self,accuracy=.90,latency=1.):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ids=[f'p-{i}' for i in range(260)]
            matrix=dict(ids=ids,tasks=[f'task-{i%13}' for i in range(260)],scores=[[.91] for _ in ids])
            protocol={'actions':[{'id':'nocache'}]}
            sealed(root/'selection.json',dict(winner={'extras':['top5_mass']},strongest_simulated={'ttft':.5}))
            def accepted(root,label,pid):
                return dict(executed_action='nocache',accuracy=.90 if label=='current' else accuracy,
                    timings={'ttft_seconds':2. if label=='current' else latency})
            with patch('runner.gpu_study.verify',return_value=(protocol,None,matrix)),patch('runner.gpu_study.accepted',side_effect=accepted):
                result=report(root)
            self.assertEqual(sealed(root/'complete.json')['answers'],520)
            self.assertEqual(len((root/'live-paired.csv').read_text().splitlines()),261)
            self.assertEqual(len((root/'live-per-task.csv').read_text().splitlines()),27)
            return result
    def test_confirm_only_when_both_requirements_hold(self):
        self.assertTrue(self.run_report()['confirmed_improvement'])
        self.assertFalse(self.run_report(latency=2.)['confirmed_improvement'])
        self.assertFalse(self.run_report(accuracy=.80)['confirmed_improvement'])
if __name__=='__main__':unittest.main()
