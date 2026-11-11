"""LongBench adapter contracts; no model or GPU construction."""
from pathlib import Path
import copy
import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.data.longbench_v2 import prepare, prepare_prompt, render
from runner.reporting.aggregate import evaluation_metadata, score_answer


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
    def test_preparation_has_no_seed_schedule_and_reuses_legacy_data(self):
        from run_fixtures import Tokenizer as PolicyTokenizer, fixture
        from scripts.collection.collection_v3 import declaration
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run_args = fixture(base, 'longbench-v2', 4096)
            sample = json.loads((run_args.prepared / 'samples/0.json').read_text())
            sample.update(query=dict(positions=sample['question_positions']),
                          formatted_text='<think>\n', truncation=dict(truncated=False))
            data = base / 'data.json'
            data.write_text(json.dumps([dict(_id='source', domain='d', sub_domain='s',
                                            difficulty='easy', length='short', answer='A')]))
            args = SimpleNamespace(model=run_args.model, data=data, output=base/'fresh', collection_v3=True)
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), \
                    patch('runner.inputs.preparation.check_model'), \
                    patch('transformers.AutoTokenizer.from_pretrained', return_value=PolicyTokenizer()), \
                    patch('scripts.data.longbench_v2.prepare_prompt', side_effect=lambda *a: copy.deepcopy(sample)), \
                    patch('scripts.data.longbench_v2.bytes_per_shard', return_value=1):
                prepare(args)
                receipt_path = args.output / 'preparation.json'
                receipt = json.loads(receipt_path.read_text())
                self.assertNotIn('answer_seeds', receipt['spec'])
                self.assertNotIn('answers_per_question', receipt['spec'])
                self.assertNotIn('answers', receipt['summary'])
                receipt['spec'].update(declaration('longbench-v2', 4))
                receipt['spec']['adapter_sha256'] = 'old-adapter-receipt'
                receipt_path.write_text(json.dumps(receipt))
                before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                          for p in args.output.rglob('*') if p.is_file()}
                prepare(args)
                self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
                (args.output / 'samples/000.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'Prepared file changed'):
                    prepare(args)

    def test_middle_truncation_recounts_formatting_and_preserves_complete_tail(self):
        template=(Path(__file__).resolve().parents[1]/'scripts/data/longbench_0shot.txt').read_text()
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
