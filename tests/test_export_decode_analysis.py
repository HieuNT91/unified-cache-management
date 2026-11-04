"""Portable decode export: lossless text, bounded size and immutable receipts."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import export_decode_analysis as exporter


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=True, indent=2).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True).encode()).hexdigest()


def fixture(root, tree=False, seeds=(0, 1), legacy=False):
    settings = dict(schema='tree-prophetkv-thinking-collection-v1' if tree else 'tp2-data-config-v3',
                    dataset='longbench-v2', model='/missing/model', prepared='/missing/prepared',
                    cache_root='/missing/cache', generation_protocol='thinking-test')
    if not legacy:
        settings.update(collection_protocol='tp2-data-v3', answers_per_question=len(seeds), answer_seeds=list(seeds))
    if tree:
        settings['kv_chunk_size'] = 1024
    rows = [dict(id=f'longbench-v2-{i:03}', source_id=f'public-question-{i}', sha256=str(i)*64,
                 ordinal=i, subtask='Agent history QA', scoring='longbench_v2', references=['B']) for i in range(2)]
    methods = {'tree': ['tree-prophetkv-5', 'tree-prophetkv-60']} if tree else (
        {'primary': ['nocache'], 'extra': ['prophetkv-5']} if legacy else {'primary': ['nocache', 'prophetkv-5']})
    settings_sha = write(root/'settings.json', settings)
    rows_sha = write(root/'rows.json', rows)
    plan = dict(settings_sha256=settings_sha, rows_sha256=rows_sha, schedule=methods)
    plan_sha = write(root/'plan.json', plan)
    for role, actions in methods.items():
        protocol = dict(settings, scheduled_actions=actions, tp=2, groups=[['GPU-a', 'GPU-b']],
                        plan_sha256=fingerprint(plan) if tree else plan_sha, answer_validation='native-answer-diagnostics-v1')
        write(root/role/'protocol.json', protocol)
        write(root/role/'session/init.json', dict(validated=True, engine_config={}))
    return rows


def commit(root, row, method='nocache', seed=0, role=None, text=None, validated=True):
    role = role or ('tree' if method.startswith('tree-') else 'primary')
    protocol = json.loads((root/role/'protocol.json').read_text())
    folder = root/role/'records'/method/row['id']
    if seed is not None:
        folder = folder/f'seed-{seed}'
    result = dict(prompt_id=row['id'], method=method, executed_action=method,
                  sequence_id=row['id'], seed=seed, input_sha256=row['sha256'], token_sha256='a'*64,
                  group=0, gpu_uuids=['GPU-a', 'GPU-b'], cache_immutable=True,
                  answer_validation='native-answer-diagnostics-v1',
                  retirement=[dict(rank=i, quiescent=True, transfers=dict(pending=0), request_bookkeeping=0) for i in range(2)],
                  initialization='session/init.json',
                  initialization_sha256=hashlib.sha256((root/role/'session/init.json').read_bytes()).hexdigest(),
                  num_cached_tokens=0, thinking_tokens=2, answer_tokens=2, control_tokens=1, output_tokens=5,
                  output_token_ids=[10, 11, 12, 13, 14], accuracy=0, output_cap_reached=False,
                  thinking_closure='natural', unfinished_thinking=False,
                  prediction=text if text is not None else '  Đếm player_7: 2 + 20 = 22.\n</think>\n\nThe correct answer is (B)  ',
                  answer_text='\n\nThe correct answer is (B)  ', scored_text='The correct answer is (B)  ',
                  context='DO NOT EXPORT CONTEXT', references=['B'], giant_diagnostic=[0]*100)
    sha = write(folder/'result.json', result)
    if validated:
        write(folder/'validated.json', dict(complete=True,
              protocol_sha256=fingerprint({k: v for k, v in protocol.items() if k not in ('prepared', 'model', 'cache_root')}),
              files={'result.json': sha, 'diagnostics.json': 'f'*64, 'huge-attention.npz': 'e'*64}))
    # Deliberately absent diagnostics and attention archives. The portable reader
    # must not require these large payloads or independently replay validation.
    return folder, result


def read_export(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


class DecodeExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'experiment'
        self.output = Path(self.tmp.name)/'decode.jsonl.gz'

    def export(self, **kwargs):
        return exporter.export(self.root, self.output, quiet=True, **kwargs)

    def test_partial_snapshot_preserves_exact_text_counts_and_public_ids(self):
        rows = fixture(self.root)
        _, original = commit(self.root, rows[0])
        commit(self.root, rows[1], 'prophetkv-5', 1)
        commit(self.root, rows[1], 'nocache', 0, validated=False)
        result = read_export(self.export())
        self.assertEqual(len(result), 4)
        first = result[1]
        for key in ('answer_text', 'scored_text', 'thinking_tokens', 'answer_tokens'):
            self.assertEqual(first[key], original[key])
        self.assertEqual(first['output'], original['prediction'])
        self.assertEqual(first['question_id'], rows[0]['source_id'])
        self.assertEqual(first['seed'], 0)
        self.assertEqual(result[-1]['missing_per_method'], {'nocache': 3, 'prophetkv-5': 3})
        for key in ('context', 'references', 'giant_diagnostic', 'output_token_ids'):
            self.assertNotIn(key, first)

    def test_standalone_copy_without_repo_packages_and_optional_ids(self):
        rows = fixture(self.root)
        commit(self.root, rows[0], text='No delimiters. 中文\n\t  exact output')
        script = Path(self.tmp.name)/'standalone.py'
        shutil.copyfile(exporter.__file__, script)
        run = subprocess.run([sys.executable, '-I', str(script), str(self.root), '--output', str(self.output),
                              '--include-token-ids'], cwd=self.tmp.name, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        answer = read_export(self.output)[1]
        self.assertEqual(answer['output'], 'No delimiters. 中文\n\t  exact output')
        self.assertEqual(answer['output_token_ids'], [10, 11, 12, 13, 14])

    def test_legacy_primary_and_extra(self):
        rows = fixture(self.root, legacy=True)
        commit(self.root, rows[0], seed=None)
        commit(self.root, rows[0], 'prophetkv-5', seed=None, role='extra')
        values = read_export(self.export())[1:-1]
        self.assertEqual({v['method'] for v in values}, {'nocache', 'prophetkv-5'})
        self.assertTrue(all(v['seed'] is None for v in values))

    def test_tree_includes_copied_baseline_only_requested_seeds_and_questions(self):
        rows = fixture(self.root, tree=True, seeds=(1,))
        commit(self.root, rows[0], 'tree-prophetkv-5', 1)
        baseline = self.root/'baseline-source'
        base_rows = fixture(baseline, seeds=(0, 1, 2))
        commit(baseline, base_rows[0], 'nocache', 0)
        commit(baseline, base_rows[0], 'nocache', 1)
        commit(baseline, base_rows[1], 'nocache', 1)
        commit(baseline, base_rows[0], 'prophetkv-5', 1)
        values = read_export(self.export(question_ids=[rows[0]['source_id']]))[1:-1]
        self.assertEqual({v['method'] for v in values}, {'nocache', 'tree-prophetkv-5'})
        self.assertEqual(len(values), 2)
        self.assertTrue(all(v['seed'] == 1 and v['prompt_id'] == rows[0]['id'] for v in values))

    def test_filters_never_require_or_read_excluded_records(self):
        rows = fixture(self.root)
        commit(self.root, rows[0], 'nocache', 0)
        folder, _ = commit(self.root, rows[1], 'prophetkv-5', 1)
        (folder/'result.json').write_text('corrupt excluded result')
        values = read_export(self.export(methods=['nocache'], seeds=[0], question_ids=[rows[0]['id']]))
        self.assertEqual(values[-1]['answers'], 1)
        self.assertEqual(values[-1]['missing_per_method'], {'nocache': 0})

    def test_size_limit_publishes_nothing_and_does_not_truncate(self):
        rows = fixture(self.root)
        import random
        noise = ''.join(random.Random(42).choices('abcdefghijklmnopqrstuvwxyz0123456789', k=30000))
        commit(self.root, rows[0], text=noise)
        with self.assertRaisesRegex(ValueError, 'exceeds size limit'):
            self.export(max_mb=.002)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(Path(self.tmp.name).glob('.decode-export-*')))

    def test_existing_output_empty_selection_and_bad_filter(self):
        rows = fixture(self.root)
        with self.assertRaisesRegex(ValueError, 'No committed answers'):
            self.export()
        with self.assertRaisesRegex(ValueError, 'Unknown/empty --methods'):
            self.export(methods=['typo'])
        commit(self.root, rows[0])
        self.output.write_bytes(b'preserve me')
        with self.assertRaisesRegex(ValueError, 'Output exists'):
            self.export()
        self.assertEqual(self.output.read_bytes(), b'preserve me')

    def test_corrupt_result_and_receipt_fail_without_publication(self):
        rows = fixture(self.root)
        folder, result = commit(self.root, rows[0])
        result['prediction'] = 'tampered'
        write(folder/'result.json', result)
        with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
            self.export()
        self.assertFalse(self.output.exists())
        commit(self.root, rows[0])
        receipt = json.loads((folder/'validated.json').read_text())
        receipt['protocol_sha256'] = '0'*64
        write(folder/'validated.json', receipt)
        with self.assertRaisesRegex(ValueError, 'Invalid commit receipt'):
            self.export()

    def test_committed_seed_identity_and_count_errors(self):
        for field, value, message in [('seed', 1, 'seed mismatch'), ('thinking_tokens', 200, 'token accounting'),
                                      ('answer_text', None, 'Missing decoded')]:
            with self.subTest(field=field):
                rows = fixture(self.root)
                folder, result = commit(self.root, rows[0])
                result[field] = value
                sha = write(folder/'result.json', result)
                receipt = json.loads((folder/'validated.json').read_text())
                receipt['files']['result.json'] = sha
                write(folder/'validated.json', receipt)
                with self.assertRaisesRegex(ValueError, message):
                    self.export()
                self.assertFalse(self.output.exists())

    def test_small_receipt_changed_mid_export_is_detected(self):
        rows = fixture(self.root)
        folder, _ = commit(self.root, rows[0])
        original = exporter.compact

        def changing(*args):
            value = original(*args)
            receipt = json.loads((folder/'validated.json').read_text())
            receipt['changed'] = True
            write(folder/'validated.json', receipt)
            return value

        with patch.object(exporter, 'compact', changing):
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                self.export()
        self.assertFalse(self.output.exists())

    def test_mismatched_baseline_tokens_are_rejected(self):
        rows = fixture(self.root, tree=True)
        folder, result = commit(self.root, rows[0], 'tree-prophetkv-5', 0)
        result['token_sha256'] = 'b'*64
        sha = write(folder/'result.json', result)
        receipt = json.loads((folder/'validated.json').read_text())
        receipt['files']['result.json'] = sha
        write(folder/'validated.json', receipt)
        baseline = self.root/'baseline-source'
        br = fixture(baseline)
        commit(baseline, br[0])
        with self.assertRaisesRegex(ValueError, 'submitted tokens differ'):
            self.export()

    def test_output_under_records_is_rejected(self):
        rows = fixture(self.root)
        commit(self.root, rows[0])
        self.output = self.root/'primary/records/decode.jsonl.gz'
        with self.assertRaisesRegex(ValueError, 'outside stage'):
            self.export()


if __name__ == '__main__':
    unittest.main()
