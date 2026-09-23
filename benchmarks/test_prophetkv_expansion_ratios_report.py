"""CPU regression for five-way reporting and matched-baseline speedups."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent/'prophetkv_with_expansion_ratios'))
import report
from settings import PARENT, CASES, PARAMETERS, METHOD


class CombinedReport(unittest.TestCase):
    def test_retained_labels_pairs_and_speedups(self):
        parent=json.loads((PARENT/'final/records.json').read_text())
        baseline=[r for r in parent if r['case']=='baseline']
        new=[]
        speedups=(2.,1.5,1.25)
        for case,speedup in zip(CASES,speedups):
            for original in baseline:
                row=copy.deepcopy(original)
                row.update(case=case,method=METHOD,parameters=PARAMETERS[case],
                           ttft_seconds=original['ttft_seconds']/speedup)
                new.append(row)
        protocol=dict(case_parameters=PARAMETERS,timing='synthetic reporting test',
                      timing_cohort_note='different timing cohorts')
        # Synthetic records only test aggregation inside an automatically removed
        # directory; they never enter the measured result tree or its report.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,value in [('protocol.json',protocol),('supervisor.json',dict(pid=-1,identity={},state='complete')),
                               ('cleanup.json',dict(complete=True,owned_workers_exited=True)),
                               ('preserved-parent.json',{})]:
                (root/name).write_text(json.dumps(value))
            with patch.object(report,'ROOT',root),patch.object(report,'REPO',root), \
                 patch.object(report,'verify',return_value=protocol),patch.object(report,'ensure_no_workers'), \
                 patch.object(report,'identity',return_value=None),patch.object(report,'lock'), \
                 patch.object(report.val,'all_records',return_value=new),patch.object(report,'retained_rows',return_value=parent):
                summary=report.finalize()
            self.assertEqual(len(summary),25)
            aggregate={r['method']:r for r in summary if r['task']=='aggregate'}
            self.assertEqual(set(aggregate),set(report.REPORT_CASES))
            self.assertEqual(aggregate[METHOD+'-20']['total_ratio'],.2)
            for case,speedup in zip(CASES,speedups):
                self.assertEqual(aggregate[case]['samples'],200)
                self.assertAlmostEqual(aggregate[case]['paired_mean_ttft_speedup'],speedup)
                self.assertEqual(aggregate[case]['anchor_ratio'],PARAMETERS[case]['anchor_ratio'])
            certificate=json.loads((root/'final/validation.json').read_text())
            self.assertEqual(certificate['new_validated'],600)
            self.assertEqual(certificate['retained_validated'],400)
            self.assertEqual(certificate['validated'],1000)
            with (root/'final/measurements.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),1000)


if __name__=='__main__':unittest.main()
