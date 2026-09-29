"""Live corpus reporting does not require model, token or attention artifacts."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from runner.corpus_records import protocol_identity
from runner.setups import atomic_json, file_hash
from runner.tree_policy import DEFAULT_ACTIONS
from scripts.corpus_status import collect, write_summary


def answer(pid, score=1., total=7., engine=2.):
    return dict(prompt_id=pid, subtask='task', accuracy=score,
                thinking_tokens=0, answer_tokens=3, control_tokens=1, output_tokens=4,
                output_cap_reached=False, unfinished_thinking=False,
                timings=dict(ttft_seconds=total, answer_engine_ttft_seconds=engine))


class CorpusStatusTests(unittest.TestCase):
    def setup_run(self, root, inference=False, inventory=None):
        prepared = root/'prepared'
        prepared.mkdir()
        (prepared/'manifest.jsonl').write_text(''.join(json.dumps(dict(id=pid, subtask='task', ordinal=i))+'\n'
                                                        for i, pid in enumerate(('p', 'q', 'r'))))
        protocol = dict(prepared=str(prepared), model='/missing/model', dataset='ruler',
                        kind='inference' if inference else 'collection',
                        actions=inventory or DEFAULT_ACTIONS, selected_ids=['p', 'q'])
        atomic_json(root/'protocol.json', protocol)
        atomic_json(root/'tranche.json', dict(limit_per_task=2))
        return protocol

    def commit(self, root, protocol, case, record):
        folder = root/'records'/case/record['prompt_id']
        atomic_json(folder/'result.json', dict(record, method=case))
        atomic_json(folder/'validated.json', dict(complete=True, protocol_sha256=protocol_identity(protocol),
                    files={'result.json':file_hash(folder/'result.json'), 'attention.npz':'intentionally unavailable'}))

    def test_empty_partial_and_matched_reports(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            protocol = self.setup_run(root)
            result = write_summary(root)
            self.assertEqual((result['accepted_answers'], result['expected_answers']), (0, 8))
            self.assertIn('N/A', (root/'live_summary.md').read_text())
            self.commit(root, protocol, 'nocache', answer('p', score=0))
            for action in DEFAULT_ACTIONS:
                self.commit(root, protocol, action['id'], answer('q'))
            self.commit(root, protocol, 'probe', dict(prompt_id='q', internal_tokens=1))
            # Interrupted publication must not count as accepted.
            atomic_json(root/'records'/'prophetkv-1'/'p'/'result.json', answer('p'))
            result = write_summary(root)
            self.assertEqual(result['accepted_answers'], 5)
            self.assertEqual(result['accepted_probes'], 1)
            self.assertEqual(result['reports']['nocache']['overall']['accuracy_percent'], 50)
            self.assertEqual(result['reports']['nocache']['overall']['mean_ttft_seconds'], 2)
            before = file_hash(root/'live_summary.json')
            result = write_summary(root, same_count=True)
            self.assertEqual(result['matching']['prompt_ids'], ['q'])
            self.assertEqual(result['reports']['nocache']['overall']['accuracy_percent'], 100)
            self.assertEqual(file_hash(root/'live_summary.json'), before)
            self.assertTrue((root/'same_count_summary.csv').is_file())

    def test_corrupt_results_fail_even_outside_matched_set(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            protocol = self.setup_run(root)
            self.commit(root, protocol, 'nocache', answer('p'))
            path = root/'records'/'nocache'/'p'/'result.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'pin or protocol'):
                collect(root, same_count=True)

    def test_generic_inventories_inference_and_relocation(self):
        for count in (2, 4, 6):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                inventory = DEFAULT_ACTIONS[:1]+[dict(id=f'prophetkv-{i}', method='prophetkv', ratio=i/100) for i in range(1,count)]
                self.setup_run(root, inventory=inventory)
                self.assertEqual(collect(root)['expected_answers'], 2*count)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            protocol = self.setup_run(root, inference=True)
            self.commit(root, protocol, 'router', answer('p'))
            relocated = root/'relocated'
            (root/'prepared').rename(relocated)
            result = collect(root, prepared=relocated)
            self.assertEqual(result['reports']['router']['overall']['mean_ttft_seconds'], 7)
            self.assertEqual(result['accepted_probes'], 1)

    def test_concurrent_reporters_preserve_accepted_data_and_cli(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            protocol = self.setup_run(root)
            self.commit(root, protocol, 'nocache', answer('p'))
            pins = {p:file_hash(p) for p in root.rglob('*.json')}
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda _:write_summary(root), range(4)))
            self.assertEqual(json.loads((root/'live_summary.json').read_text())['accepted_answers'], 1)
            self.assertEqual(pins, {p:file_hash(p) for p in pins})
            script = Path(__file__).resolve().parents[1]/'scripts'/'corpus_status.py'
            run = subprocess.run([sys.executable, str(script), '--root', str(root)], cwd='/tmp',
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('live_summary.md', run.stdout)
            run = subprocess.run([sys.executable, str(script.with_name('corpus_control.py')), 'status', '--root', str(root)],
                                 cwd='/tmp', env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('live_summary.md', run.stdout)


if __name__ == '__main__':
    unittest.main()
