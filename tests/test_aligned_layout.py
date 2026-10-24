import unittest

from runner.cache import aligned_prompt, cacheblend_prompt
from runner.config import chunk_end_token, validate_sample

EOT = 9


class Tokenizer:
    """Marker text maps to a fixed 7-token sequence that never contains EOT."""
    def encode(self, text, add_special_tokens=False):
        return [1000 + len(text) % 50] * 7


def strip_markers(chunks):
    return [t for c in chunks for t in c[7:]]


class AlignedLayoutTests(unittest.TestCase):
    def sample(self, n=20000, suffix=300):
        ids = list(range(10, 10 + n))
        chunks, tokens, suffix_len = aligned_prompt(ids, Tokenizer(), 4096, suffix)
        bounds = [0]
        for c in chunks:
            bounds.append(bounds[-1] + len(c))
        bounds.append(len(tokens))
        q = list(range(len(tokens) - 40, len(tokens) - 10))
        return ids, chunks, tokens, suffix_len, dict(
            token_ids=tokens, boundaries=bounds, question_positions=q, thinking=False,
            max_output_tokens=128, layout='aligned', chunk_end_token_id=-1, forbidden_token_id=EOT)

    def test_chunks_are_whole_blocks_and_source_is_preserved(self):
        ids, chunks, tokens, suffix_len, _ = self.sample()
        self.assertTrue(all(len(c) % 64 == 0 and len(c) <= 4096 for c in chunks))
        self.assertGreaterEqual(suffix_len, 300)
        self.assertEqual(strip_markers(chunks) + tokens[len(tokens) - suffix_len:], ids)
        self.assertNotIn(EOT, tokens)

    def test_validation_and_connector_delimiter(self):
        *_, sample = self.sample()
        validate_sample(sample, 4096, 131072)
        self.assertEqual(chunk_end_token(sample), -1)
        bad = dict(sample, token_ids=[EOT] + sample['token_ids'][1:])
        with self.assertRaises(ValueError):
            validate_sample(bad, 4096, 131072)
        with self.assertRaises(ValueError):
            validate_sample(dict(sample, chunk_end_token_id=EOT), 4096, 131072)

    def test_padded_default_unchanged(self):
        ids = list(range(10, 20010))
        chunks, tokens = cacheblend_prompt(ids, Tokenizer(), EOT, 4095, 320)
        bounds = [0]
        for c in chunks:
            bounds.append(bounds[-1] + len(c))
        bounds.append(len(tokens))
        sample = dict(token_ids=tokens, boundaries=bounds, question_positions=[len(tokens) - 5],
                      thinking=False, max_output_tokens=8)
        validate_sample(sample, 4096, 131072)
        self.assertEqual(chunk_end_token(sample), EOT)
        self.assertTrue(all(c[-1] == EOT for c in chunks))


if __name__ == '__main__':
    unittest.main()
