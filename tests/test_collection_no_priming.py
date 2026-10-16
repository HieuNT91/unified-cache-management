"""Collection begins with construction only; the first read is requested work."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from runner.corpus_runtime import Engine
from runner.setups import atomic_json, file_hash


class NoPrimingTests(unittest.TestCase):
    def fixture(self, root, cached=True):
        atomic_json(root/'config.json', {})
        sample = dict(token_ids=list(range(576)), boundaries=[0, 64, 320, 576],
            question_positions=[320], thinking=True, model_config_sha256=file_hash(root/'config.json'))
        atomic_json(root/'sample.json', sample)
        engine = Engine.__new__(Engine)
        engine.protocol = dict(prepared=str(root), model=str(root), dataset='ruler', execution_profile='ruler-thinking')
        engine.cached = cached; engine.tp = 2; engine.pc = None; engine.cache = root/'cache'; engine.llm = Mock()
        row = dict(id='sample', prepared='sample.json', sha256=file_hash(root/'sample.json'))
        return engine, row, sample

    def test_cached_begin_only_populates_retires_and_checks_readiness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); engine, row, sample = self.fixture(root)
            before = (root/'sample.json').read_bytes(), (root/'sample.json').stat().st_mtime_ns
            cache = Mock(chunks=[sample['token_ids'][:64], sample['token_ids'][64:320]])
            events = []
            cache.ready.side_effect = lambda: events.append('ready') or {'verified_shards': 10}
            cache.unchanged.side_effect = lambda: events.append('unchanged')
            def generate(_engine, tokens, cap, rid, **kwargs):
                self.assertEqual(cap, 1)
                self.assertEqual(kwargs, {'phase': 'populate'})
                self.assertIn(':populate:', rid)
                events.append(tokens)
            with patch('runner.corpus_runtime.validate'), patch('runner.sweep.PromptCache', return_value=cache), \
                    patch('runner.generation.generate', side_effect=generate) as gen, \
                    patch('runner.corpus_runtime.retirement', side_effect=lambda *a: events.append('retire')), \
                    patch('runner.corpus_runtime.arm_tree') as arm, \
                    patch('runner.corpus_runtime.verify_retired') as retired, \
                    patch('runner.corpus_runtime.clean_capture', side_effect=lambda *a: events.append('clean')):
                engine.begin(row)
            self.assertEqual(events, [sample['token_ids'][:64], 'retire', sample['token_ids'][64:320],
                                      'retire', 'ready', 'unchanged', 'clean'])
            self.assertEqual(gen.call_count, 2)
            arm.assert_not_called(); retired.assert_not_called()
            engine.llm.collective_rpc.assert_not_called()
            self.assertEqual(engine.sample['token_ids'], sample['token_ids'])
            self.assertNotIn('priming_seconds', engine.construction)
            self.assertEqual(before, ((root/'sample.json').read_bytes(), (root/'sample.json').stat().st_mtime_ns))

    def test_baseline_begin_does_not_submit_a_discarded_answer(self):
        with tempfile.TemporaryDirectory() as temp:
            engine, row, _ = self.fixture(Path(temp), cached=False)
            with patch('runner.corpus_runtime.validate'), patch('runner.generation.generate') as generate, \
                    patch('runner.sweep.PromptCache') as cache:
                engine.begin(row)
            generate.assert_not_called(); cache.assert_not_called()
            self.assertEqual(engine.construction, {})

    def test_readiness_or_cache_mutation_failure_still_stops_begin(self):
        for operation in ('ready', 'unchanged'):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as temp:
                engine, row, _ = self.fixture(Path(temp)); cache = Mock(chunks=[])
                getattr(cache, operation).side_effect = RuntimeError(operation)
                with patch('runner.corpus_runtime.validate'), patch('runner.sweep.PromptCache', return_value=cache), \
                        patch('runner.generation.generate') as generate, patch('runner.corpus_runtime.arm_tree') as arm:
                    with self.assertRaisesRegex(RuntimeError, operation): engine.begin(row)
                generate.assert_not_called(); arm.assert_not_called()


if __name__ == '__main__': unittest.main()
