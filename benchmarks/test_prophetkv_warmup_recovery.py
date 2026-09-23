"""CPU-only regression for asynchronous warmup disk commit readiness."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).parent / 'prophetkv_with_expansion_ratios_recovery'
sys.path.insert(0, str(SOURCE))
import warmup_readiness as recovery
from cacheblend_ruler import wait_for_cache
from lifecycle import delete_retired_files
from prophetkv_common import cache_snapshot


class WarmupReadinessTests(unittest.TestCase):
    def test_waits_for_both_full_size_committed_blocks_before_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'; model.mkdir()
            (model / 'config.json').write_text(json.dumps(dict(
                num_attention_heads=1, hidden_size=2, num_hidden_layers=1)))
            cache = root / 'cache'; temp = cache / 'kv/.temp'; temp.mkdir(parents=True)
            namespace = 'a' * 32
            meta = f'{model}:1:torch.bfloat16:0'.encode()
            def hashed(value):
                return hashlib.md5(meta + pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).digest()
            parent = hashed(recovery.seed_value(namespace)); targets = []
            for offset in (0, 64):
                parent = hashed((parent, tuple(recovery.WARMUP_TOKENS[offset:offset+64])))
                name = parent.hex(); target = cache / 'kv' / name[:8] / name
                target.parent.mkdir(); (temp / name).write_bytes(b'pending')
                targets.append(target)
            # Reproduce the original failure without touching any temporary file.
            with self.assertRaisesRegex(ValueError, 'non-block file'):
                delete_retired_files(cache, cache_snapshot(cache))
            now = [0.0]
            def sleep(delay):
                now[0] += delay
                if now[0] == .25:
                    (temp / targets[0].name).rename(targets[0])  # Wrong size still fails.
                elif now[0] == .5:
                    targets[0].write_bytes(b'x' * 512)  # Second shard still missing.
                else:
                    (temp / targets[1].name).write_bytes(b'x' * 512)
                    (temp / targets[1].name).rename(targets[1])
            def checked_wait(args, groups):
                self.assertEqual(groups, [recovery.WARMUP_TOKENS])
                self.assertEqual(args.hash_seed, recovery.seed_value(namespace))
                return wait_for_cache(args, groups, clock=lambda: now[0], sleep=sleep)
            with patch.object(recovery, 'wait_for_cache', checked_wait):
                receipt = recovery.wait_for_warmup_cache(model, cache, namespace)
            self.assertEqual(receipt['readiness_wait_seconds'], .75)
            self.assertEqual(receipt['verified_shards'], 2)
            delete_retired_files(cache, cache_snapshot(cache))
            self.assertTrue(temp.is_dir())
            self.assertFalse(any(p.is_file() for p in cache.rglob('*')))

    def test_readiness_failure_propagates_without_cleanup(self):
        with patch.object(recovery, 'wait_for_cache', side_effect=RuntimeError('Cache readiness timed out')):
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                recovery.wait_for_warmup_cache('/model', '/cache', 'b' * 32)


if __name__ == '__main__':
    unittest.main()
