"""Independent CPU references for native AdaTreeV1 feature masks and routing."""
import unittest

import numpy as np
import torch

from ucm.sparse.prophetkv.adatree import compute_features, decide, route_features


def reference(raw, scores, boundaries):
    p = raw.astype(np.float64)
    p /= p.sum(1, keepdims=True)
    order = np.argsort(-scores, kind='stable')
    chunk = np.repeat(np.arange(len(boundaries) - 3), np.diff(boundaries[1:-1]))
    values = []
    for rate in (.05, .60):
        tail = np.ones(len(scores))
        tail[order[:int(np.floor(rate * len(scores)))]] = 0
        q = np.array([np.bincount(chunk, weights=row * tail,
                                 minlength=len(boundaries) - 3) for row in p])
        squared = np.square(q).sum(1)
        if rate == .05:
            values.append(np.sqrt(squared).mean())
        else:
            values.append(np.quantile(q.sum(1)**2 / (q.shape[1] * squared), .90, method='linear'))
    return values


class AdaTreeFeatureTests(unittest.TestCase):
    def test_matches_independent_reference_with_unequal_chunks_and_ties(self):
        rng = np.random.default_rng(42)
        for layers in (1, 8, 64):
            for tied in (False, True):
                b = [0, 64, 192, 256, 448, 459]
                raw = rng.random((layers, b[-2] - b[1]), dtype=np.float32)
                scores = (rng.integers(0, 4, raw.shape[1]).astype(np.float32)
                          if tied else rng.random(raw.shape[1], dtype=np.float32))
                actual = compute_features(torch.from_numpy(raw), torch.from_numpy(scores), b)
                np.testing.assert_allclose(actual.numpy(), reference(raw, scores, b), atol=1e-12, rtol=0)
                self.assertEqual(actual.dtype, torch.float64)

    def test_stable_native_mask_and_eligible_chunk_denominator(self):
        # Tied scores mask earliest eligible tokens; the prefix and suffix are
        # excluded from both normalized attention and the two-chunk denominator.
        features = compute_features(torch.ones(64, 100), torch.ones(100), [0, 10, 60, 110, 111])
        np.testing.assert_allclose(features.numpy(), [np.hypot(.45, .50), .50], atol=1e-12, rtol=0)

    def test_layer_scale_invariance_and_inputs_not_mutated(self):
        raw = torch.arange(1, 401, dtype=torch.float32).reshape(4, 100)
        scores = torch.ones(100)
        original = raw.clone(), scores.clone()
        b = [0, 10, 60, 110, 111]
        first = compute_features(raw, scores, b)
        second = compute_features(raw * torch.tensor([1., 2., 4., 8.])[:, None], scores, b)
        torch.testing.assert_close(first, second, atol=1e-14, rtol=0)
        self.assertTrue(torch.equal(raw, original[0]))
        self.assertTrue(torch.equal(scores, original[1]))

    def test_uniform_tail_spans_all_chunks(self):
        scores = torch.tensor([4., 3., 2., 1., 0.] * 4)
        b = [0, 1, 6, 11, 16, 21, 22]
        f = compute_features(torch.ones(3, 20), scores, b)
        self.assertAlmostEqual(float(f[1]), 1.)
        concentrated = torch.zeros(3, 20)
        concentrated[:, -5:] = 1.
        f = compute_features(concentrated, scores, b)
        self.assertAlmostEqual(float(f[1]), .25)

    def test_exact_inclusive_thresholds_and_monotonicity(self):
        g = [0., .22, np.nextafter(.22, 1.), 1.]
        s = [0., .73, np.nextafter(.73, 1.), 1.]
        matrix = []
        for gv in g:
            row = []
            for sv in s:
                percent = int(decide(torch.tensor([gv, sv], dtype=torch.float64)))
                self.assertEqual(route_features(gv, sv), f'tree-prophetkv-{percent}')
                row.append(percent)
            matrix.append(row)
        self.assertTrue((np.diff(matrix, axis=0) >= 0).all())
        self.assertTrue((np.diff(matrix, axis=1) >= 0).all())
        self.assertEqual(route_features(.9, .73), 'tree-prophetkv-5')
        self.assertEqual(route_features(.22, .74), 'tree-prophetkv-20')
        self.assertEqual(route_features(np.nextafter(.22, 1.), .74), 'tree-prophetkv-40')

    def test_bad_attention_scores_and_boundaries_fail_closed(self):
        raw, scores, b = torch.ones(2, 100), torch.ones(100), [0, 10, 60, 110, 111]
        bad_raw = [raw[:0], raw[:, :99], -raw, raw * float('nan'), raw * float('inf'), raw * 0]
        for value in bad_raw:
            with self.subTest(raw_shape=value.shape), self.assertRaises(ValueError):
                compute_features(value, scores, b)
        for value in [scores[:99], scores * float('nan'), scores * float('inf')]:
            with self.subTest(scores=value.shape), self.assertRaises(ValueError):
                compute_features(raw, value, b)
        for value in [[], [0, 10, 110], [0, 10, 10, 110, 111], [1, 10, 60, 110, 111],
                      [0, 10., 60, 110, 111]]:
            with self.subTest(boundaries=value), self.assertRaises(ValueError):
                compute_features(raw, scores, value)

    def test_zero_tail_is_undefined_and_invalid_features_rejected(self):
        raw = torch.zeros(2, 100)
        raw[:, :60] = 1.
        with self.assertRaisesRegex(ValueError, 'Zero residual'):
            compute_features(raw, torch.ones(100), [0, 10, 60, 110, 111])
        for values in [[float('nan'), .5], [.5, float('inf')], [-.1, .5], [1.01, .5], [.5, 1.01]]:
            with self.subTest(features=values), self.assertRaises(ValueError):
                decide(torch.tensor(values, dtype=torch.float64))
            with self.assertRaises(ValueError):
                route_features(*values)


if __name__ == '__main__':
    unittest.main()
