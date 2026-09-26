"""Exact RULER input lengths, native thinking, scope and final-report gates."""
import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from scripts.ruler_64000 import TASKS, PERCENTAGES, format_sample, generator_command, query_span
from scripts.sweep_report import collect
from runner.reporting import aggregate, score_answer
from runner.setups import atomic_json
from runner.sweep import configurations


class CharacterTokenizer:
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['enable_thinking']
        return '[USER]'+messages[0]['content']+'[END]<think>\n'
    def __call__(self,text,**kwargs):
        return dict(input_ids=list(map(ord,text)),offset_mapping=[(i,i+1) for i in range(len(text))])
    def encode(self,text,**kwargs):return list(map(ord,text))
    def convert_tokens_to_ids(self,text):return 999


class RulerTests(unittest.TestCase):
    def test_all_tasks_exact_length_keep_source_question_and_thinking(self):
        tokenizer=CharacterTokenizer()
        for task in TASKS:
            with self.subTest(task=task):
                query='What are all the special magic numbers for KEY?' if task.startswith('niah_') else 'Question: Return the TARGET answer.'
                row=dict(index=0,input='Instructions.\n'+'context TARGET '*430+'\n'+query,
                         outputs=['TARGET'],answer_prefix='FORCED PREFIX MUST NOT ENTER THINKING')
                sample=format_sample(tokenizer,row,task)
                self.assertEqual(len(sample['token_ids']),64000)
                self.assertEqual(sample['max_output_tokens'],16384)
                self.assertTrue(sample['thinking']);self.assertFalse(sample['truncated'])
                self.assertEqual(''.join(chr(sample['token_ids'][p]) for p in sample['question_positions']),query)
                # Remove only the documented chunk markers/padding and length padding.
                recovered=[]
                for i,(a,b) in enumerate(zip(sample['boundaries'][:-2],sample['boundaries'][1:-1])):
                    chunk=sample['token_ids'][a:b]
                    marker=tokenizer.encode(f'\n[Context chunk {i+1}]\n')
                    self.assertEqual(chunk[:len(marker)],marker)
                    payload=chunk[len(marker):]
                    while payload[-1]==999:payload=payload[:-1]
                    recovered.extend(payload)
                recovered.extend(sample['token_ids'][sample['boundaries'][-2]:])
                insertion=sample['padding']['original_insertion_position']
                count=sample['padding']['count']
                del recovered[insertion:insertion+count]
                text=''.join(map(chr,recovered))
                self.assertEqual(text,'[USER]'+row['input']+'[END]<think>\n')
                self.assertNotIn('FORCED PREFIX',text)

    def test_long_question_is_fresh_and_overlength_is_rejected(self):
        row=dict(index=0,input='Instructions\n'+'context '*800+'\nQuestion: '+'long question '*60)
        sample=format_sample(CharacterTokenizer(),row,'qa_1')
        self.assertGreater(sample['boundaries'][-1]-sample['boundaries'][-2],256)
        self.assertTrue(all(p>=sample['boundaries'][-2] for p in sample['question_positions']))
        row['input']='x'*64000+'\nQuestion: answer?'
        with self.assertRaisesRegex(ValueError,'truncation is forbidden'):
            format_sample(CharacterTokenizer(),row,'qa_1')

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
        constants={'qa':dict(template='context {context} question {query}',answer_prefix=' Answer:')}
        with patch('scripts.ruler_64000.definitions',return_value=(customized,constants)):
            command=generator_command(args,'qa_1',Path('/raw'))
        self.assertEqual(command[0],sys.executable)
        for flag,value in [('--num_samples','100'),('--max_seq_length','63360'),('--random_seed','42'),
                           ('--tokens_to_generate','128'),('--dataset','squad')]:
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
        self.assertEqual(len(tasks)*len(configs),15200)
        for bad in ([],[1,1],[0],[101]):
            with self.assertRaises(ValueError):configurations(bad)

    def test_live_and_final_aggregate_all_15200_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'out';manifest=root/'manifest.jsonl'
            entries=[dict(id=f'{task}-{i}',prepared='unused.json',subtask=task) for task in TASKS for i in range(100)]
            manifest.write_text(''.join(json.dumps(row)+'\n' for row in entries))
            live=collect(output,manifest,PERCENTAGES)
            self.assertEqual(live['completed'],0);self.assertEqual(live['expected'],15200)
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
                atomic_json(output/f'group-{shard}'/'complete.json',dict(engine_shutdown=True,cache_deleted=True,measurements=7600))
            final=collect(output,manifest,PERCENTAGES,True)
            self.assertEqual(final['completed'],15200)
            self.assertEqual(final['methods']['selective-80']['subtasks']['cwe']['accuracy_percent'],75.)
            with (output/'final_summary.csv').open() as stream:
                self.assertEqual(len(list(csv.reader(stream))),1+19*9)
            atomic_json(output/'group-1/complete.json',dict(engine_shutdown=True,cache_deleted=False,measurements=7600))
            with self.assertRaisesRegex(RuntimeError,'cache deletion'):
                collect(output,manifest,PERCENTAGES,True)

if __name__=='__main__':unittest.main()
