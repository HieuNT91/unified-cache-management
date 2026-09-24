"""Read-only snapshot, integrity and pairing regressions; standard library only."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import a800_live_results as live


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class LiveResultsTests(unittest.TestCase):
    def fixture(self, root, job='0'):
        directory = root / ('job-' + job)
        sample = dict(id='sample-' + job, dataset='ruler', label='qa_2', input_sha256='input')
        protocol = dict(samples=[sample], cases=['baseline', 'expansion-10'], measured_requests=2)
        dump(directory / 'protocol.json', protocol)
        dump(directory / 'progress.json', dict(state='running'))
        record = dict(sample_id=sample['id'], dataset='ruler', label='qa_2', input_sha256='input',
            protocol_sha256=hashlib.sha256((directory / 'protocol.json').read_bytes()).hexdigest(),
            case='baseline', score=1, ttft_seconds=8, finish_reason='stop')
        path = directory / 'records' / sample['id'] / 'baseline.json'
        dump(path, record)
        dump(path.with_suffix('.validated.json'), dict(complete=True, sample_id=sample['id'],
            case='baseline', hashes={'.json': hashlib.sha256(path.read_bytes()).hexdigest()}))
        # An output exists but has not passed acceptance yet. Never count it.
        dump(path.with_name('expansion-10.json'), dict(record, case='expansion-10'))
        return directory, path

    def test_only_accepted_records_and_no_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory, _ = self.fixture(root)
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            rows, cases, info = live.read_job(directory)
            self.assertEqual(len(rows), 1)
            self.assertEqual(info['accepted'], 1)
            self.assertEqual(info['target'], 2)
            self.assertEqual(cases, ['baseline', 'expansion-10'])
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_corruption_and_receipt_identity_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            directory, path = self.fixture(Path(folder))
            raw = path.read_bytes()
            record = json.loads(raw)
            record['score'] = 0
            dump(path, record)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                live.read_job(directory)
            path.write_bytes(raw)
            marker = path.with_suffix('.validated.json')
            receipt = json.loads(marker.read_bytes())
            receipt['sample_id'] = 'wrong'
            dump(marker, receipt)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                live.read_job(directory)

    def test_paired_speedup_excludes_unpaired_outputs(self):
        base = dict(dataset='ruler', label='qa_2', finish_reason='stop')
        rows = [dict(base, sample_id='a', case='baseline', score=1, ttft_seconds=8),
                dict(base, sample_id='a', case='expansion-10', score=0, ttft_seconds=2),
                dict(base, sample_id='b', case='expansion-10', score=1, ttft_seconds=20)]
        table = live.summarize(rows, ['baseline', 'expansion-10', 'expansion-20'])
        selected = next(r for r in table if r['task']=='qa_2' and r['method']=='expansion-10')
        self.assertEqual(selected['samples'], 2)
        self.assertEqual(selected['accuracy_percent'], 50)
        self.assertEqual(selected['mean_ttft_seconds'], 11)
        self.assertEqual(selected['paired_samples'], 1)
        self.assertEqual(selected['paired_mean_ttft_speedup'], 4)
        empty = next(r for r in table if r['task']=='qa_2' and r['method']=='expansion-20')
        self.assertIsNone(empty['mean_ttft_seconds'])
        self.assertIsNone(empty['paired_mean_ttft_speedup'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            live.summarize(rows + [copy.deepcopy(rows[0])], ['baseline'])

    def test_cli_both_jobs_json_and_missing_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root, '0')
            self.fixture(root, '1')
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = live.main(['--root', str(root), '--format', 'json'])
            self.assertEqual(code, 0)
            result = json.loads(stdout.getvalue())
            self.assertTrue(result['provisional'])
            self.assertEqual(sum(j['accepted'] for j in result['jobs']), 2)
            self.assertIn('PROVISIONAL', stderr.getvalue())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(live.main(['--root', str(root / 'absent')]), 1)

    def test_empty_snapshot(self):
        self.assertEqual(live.summarize([], ['baseline']), [])
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            live.print_table([])
        self.assertIn('Task', stream.getvalue())


if __name__ == '__main__':
    unittest.main()
