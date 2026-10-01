"""LongBench adapter contracts; no model or GPU construction."""
from pathlib import Path
import unittest

from scripts.longbench_v2 import prepare_prompt, render
from runner.reporting import evaluation_metadata, score_answer


class Tokenizer:
    chat_template = 'test native template'
    pad_token_id=999
    def encode(self,text,**kwargs):
        return [ord(c) for c in text]
    def decode(self,ids,**kwargs):
        return ''.join(chr(i) for i in ids)
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['enable_thinking']
        return '<user>'+messages[0]['content']+'<assistant><think>'
    def __call__(self,text,**kwargs):
        return dict(input_ids=self.encode(text),offset_mapping=[(i,i+1) for i in range(len(text))])


class LongBenchTests(unittest.TestCase):
    def test_middle_truncation_recounts_formatting_and_preserves_complete_tail(self):
        template=(Path(__file__).resolve().parents[1]/'scripts/longbench_0shot.txt').read_text()
        row=dict(context='first '+('abcdef '*3000)+' last',question='Which choice?',
                 choice_A='one',choice_B='two',choice_C='three',choice_D='four')
        full,query,tail=render(row,template)
        sample=prepare_prompt(Tokenizer(),row,template,8192)
        self.assertTrue(sample['truncation']['truncated'])
        self.assertLessEqual(sample['tokens'],8192)
        self.assertTrue(sample['rendered_user_prompt'].endswith(tail))
        a,b=sample['truncation']['retained_source_token_spans']
        self.assertEqual(a[1]-a[0],b[1]-b[0])
        self.assertEqual(sample['rendered_user_prompt'],full[:a[1]]+full[b[0]:])
        self.assertTrue(all(p>=sample['boundaries'][-2] for p in sample['query']['positions']))
        self.assertEqual(sample['query']['text'],query)
        self.assertTrue(all(z-a<=4096 for a,z in zip(sample['boundaries'][:-2],sample['boundaries'][1:-1])))
        intact=prepare_prompt(Tokenizer(),row,template,114688)
        self.assertFalse(intact['truncation']['truncated'])
        self.assertEqual(intact['rendered_user_prompt'],full)

    def test_official_choice_extractor_is_not_generic_choice_scoring(self):
        evaluation=evaluation_metadata(dict(scoring='longbench_v2',references=['B']))
        for text in ('The correct answer is (B)', '**The correct answer is B**'):
            self.assertEqual(score_answer(text,evaluation),1)
        for text in ('B','Answer: B','the correct answer is B','The correct answer is (A)'):
            self.assertEqual(score_answer(text,evaluation),0)
        # Official extractor prioritizes parenthesized matches over bare ones.
        self.assertEqual(score_answer('The correct answer is A. The correct answer is (B)',evaluation),1)


if __name__=='__main__':
    unittest.main()
