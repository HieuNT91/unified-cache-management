"""Exact RULER input lengths, native thinking, scope and final-report gates."""
import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from scripts.ruler_64000 import TASKS, PERCENTAGES, format_sample, generator_command, query_span, CAPS
from scripts.sweep_report import collect
from runner.reporting import aggregate, score_answer
from runner.setups import atomic_json
from runner.sweep import configurations


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
        with patch('scripts.ruler_64000.definitions',return_value=(customized,constants)):
            command=generator_command(args,'qa_1',Path('/raw'),CharacterTokenizer())
        self.assertEqual(command[0],sys.executable)
        for flag,value in [('--num_samples','100'),('--max_seq_length','65536'),('--random_seed','42'),
                           ('--tokens_to_generate','32'),('--dataset','squad')]:
            self.assertEqual(command[command.index(flag)+1],value)

    def test_19_configurations_and_balanced_task_shards(self):
        configs=configurations(PERCENTAGES)
        self.assertEqual(len(configs),19)
        self.assertEqual([c['ratio'] for c in configs[1:10]],[p/100 for p in PERCENTAGES])
        self.assertTrue(all(c['layers']==[11,12,13,14,15] for c in configs[10:]))
        tasks=[task for task in TASKS for _ in range(100)]
        from collections import Counter
        for shard in (0,1):
            self.assertEqual(Counter(tasks[shard::2]),dict.fromkeys(TASKS,50))
        self.assertEqual(len(tasks)*len(configs),24700)
        for bad in ([],[1,1],[0],[101]):
            with self.assertRaises(ValueError):configurations(bad)

    def test_live_and_final_aggregate_all_24700_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'out';manifest=root/'manifest.jsonl'
            entries=[dict(id=f'{task}-{i}',prepared='unused.json',subtask=task) for task in TASKS for i in range(100)]
            manifest.write_text(''.join(json.dumps(row)+'\n' for row in entries))
            live=collect(output,manifest,PERCENTAGES)
            self.assertEqual(live['completed'],0);self.assertEqual(live['expected'],24700)
            self.assertFalse((output/'final_summary.json').exists())
            expected=[dict(prompt_id=r['id'],subtask=r['subtask']) for r in entries]
            records=[dict(prompt_id=r['prompt_id'],subtask=r['subtask'],accuracy=.75,
                thinking_tokens=100,answer_tokens=4,control_tokens=2,output_tokens=106,
                output_cap_reached=False,unfinished_thinking=False,timings={'ttft_seconds':2.}) for r in expected]
            for config in configurations(PERCENTAGES):
                report=aggregate(records,expected,dict(method=config['method'],ratio=config['ratio'],sweep_fingerprint='test'),'completed')
                atomic_json(output/config['name']/'final_aggregation.json',report)
            with self.assertRaises(FileNotFoundError):collect(output,manifest,PERCENTAGES,True)
            for shard in (0,1):
                atomic_json(output/f'group-{shard}'/'complete.json',dict(engine_shutdown=True,cache_deleted=True,measurements=12350))
            final=collect(output,manifest,PERCENTAGES,True)
            self.assertEqual(final['completed'],24700)
            self.assertEqual(final['methods']['selective-80']['subtasks']['cwe']['accuracy_percent'],75.)
            with (output/'final_summary.csv').open() as stream:
                self.assertEqual(len(list(csv.reader(stream))),1+19*14)
            atomic_json(output/'group-1/complete.json',dict(engine_shutdown=True,cache_deleted=False,measurements=12350))
            with self.assertRaisesRegex(RuntimeError,'cache deletion'):
                collect(output,manifest,PERCENTAGES,True)

if __name__=='__main__':unittest.main()
