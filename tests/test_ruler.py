"""Exact RULER input lengths, native thinking, scope and final-report gates."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from scripts.data.ruler import TASKS, PERCENTAGES, format_sample, generator_command, query_span, CAPS
from runner.reporting.aggregate import aggregate, score_answer
from runner.cache.setups import atomic_json
from scripts.data import ruler
from scripts import run


class CharacterTokenizer:
    chat_template = 'native-test-template'
    def apply_chat_template(self,messages,**kwargs):
        assert not kwargs['enable_thinking']
        return '[USER]'+messages[0]['content']+'[END][ASSISTANT]<think>\n\n</think>\n\n'
    def __call__(self,text,**kwargs):
        return dict(input_ids=list(map(ord,text)),offset_mapping=[(i,i+1) for i in range(len(text))])
    def encode(self,text,**kwargs):return list(map(ord,text))


def original_row(content):
    return dict(index=0,input=CharacterTokenizer().apply_chat_template([dict(content=content)],enable_thinking=False),
                answer_prefix=' Answer:',outputs=['TARGET'])


class RulerTests(unittest.TestCase):
    def source_fixture(self, root):
        for name in ('scripts/synthetic.yaml','scripts/data/synthetic/constants.py'):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
        path=root/'scripts/data/synthetic/json/english_words.json'
        path.parent.mkdir(parents=True)
        path.write_text('version https://git-lfs.github.com/spec/v1\n'
                        f'oid sha256:{ruler.ENGLISH_WORDS_SHA256}\nsize 1\n')
        return path

    def test_source_validation_does_not_require_git_or_a_matching_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.source_fixture(root)
            with patch.object(ruler.subprocess,'check_output',side_effect=AssertionError('No Git revision gate')):
                ruler.source_check(root)
                (root/'.git').mkdir();(root/'.git/HEAD').write_text('unrelated local revision\n')
                ruler.source_check(root)

    def test_source_validation_still_rejects_missing_files_and_changed_wordlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);wordlist=self.source_fixture(root)
            wordlist.write_text('{"changed":"asset"}')
            with self.assertRaisesRegex(ValueError,'digest'):ruler.source_check(root)
            wordlist.unlink()
            with self.assertRaisesRegex(ValueError,'Missing RULER'):ruler.source_check(root)

    def test_run_checks_other_asset_content_without_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);wordlist=self.source_fixture(root);expected={}
            for name in run.RULER_ASSETS:
                path=wordlist.with_name(name)
                if name!=wordlist.name:path.write_text(name)
                expected[name]=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(run,'RULER_ASSETS',expected), \
                    patch.object(ruler.subprocess,'check_output',side_effect=AssertionError('No Git revision gate')):
                run.validate_ruler_assets(root)
                wordlist.with_name('squad.json').write_text('changed content')
                with self.assertRaisesRegex(ValueError,'asset differs'):run.validate_ruler_assets(root)

    def test_all_tasks_original_tokens_prefix_and_caps(self):
        tokenizer=CharacterTokenizer()
        for task in TASKS:
            query='What are all the special magic numbers for KEY?' if task.startswith('niah_') else 'Question: Return the TARGET answer.'
            row=original_row('Instructions.\n'+'context TARGET '*4300+'\n'+query)
            sample=format_sample(tokenizer,row,task)
            self.assertEqual(sample['token_ids'],tokenizer.encode(row['input']+row['answer_prefix']))
            self.assertEqual(sample['max_output_tokens'],CAPS[task])
            self.assertLessEqual(len(sample['token_ids'])+CAPS[task],65536)
            self.assertFalse(sample['thinking']);self.assertFalse(sample['truncated'])
            self.assertEqual(''.join(chr(sample['token_ids'][p]) for p in sample['question_positions']),query)
            self.assertNotIn('padding',sample)

    def test_long_question_is_fresh_and_overlength_is_rejected(self):
        row=original_row('Instructions\n'+'context '*800+'\nQuestion: '+'long question '*60)
        sample=format_sample(CharacterTokenizer(),row,'qa_1')
        self.assertGreater(sample['boundaries'][-1]-sample['boundaries'][-2],256)
        self.assertTrue(all(p>=sample['boundaries'][-2] for p in sample['question_positions']))
        row=original_row('x'*65536+'\nQuestion: answer?')
        before=copy.deepcopy(row)
        with self.assertRaisesRegex(ValueError,'retain raw and halt'):
            format_sample(CharacterTokenizer(),row,'qa_1')
        self.assertEqual(row,before)

    def test_ruler_official_scoring_semantics(self):
        evaluation=dict(references=['A B','OTHER'],scoring='ruler_all')
        self.assertEqual(score_answer('a b',evaluation),.5)
        self.assertEqual(score_answer('a  b',evaluation),0.)  # no whitespace normalization
        evaluation['scoring']='ruler_any'
        self.assertEqual(score_answer('other',evaluation),1.)
        self.assertEqual(score_answer('',evaluation),0.)

    def test_generator_preserves_upstream_parameters_and_uses_same_interpreter(self):
        import sys
        args=NS(ruler=Path('/official'),model=Path('/model'),samples=100)
        customized={'qa_1':dict(task='qa',args={'dataset':'squad'})}
        constants={'qa':dict(template='context {context} question {query}',answer_prefix=' Answer:',tokens_to_generate=32)}
        with patch('scripts.data.ruler.definitions',return_value=(customized,constants)):
            command=generator_command(args,'qa_1',Path('/raw'),CharacterTokenizer())
        self.assertEqual(command[0],sys.executable)
        for flag,value in [('--num_samples','100'),('--max_seq_length','65536'),('--random_seed','42'),
                           ('--tokens_to_generate','32'),('--dataset','squad')]:
            self.assertEqual(command[command.index(flag)+1],value)



if __name__=='__main__':unittest.main()
