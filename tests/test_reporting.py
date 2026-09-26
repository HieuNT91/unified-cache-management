"""Live/final metric calculations and token accounting without a GPU."""
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from runner.reporting import (AggregationReporter, OutputAnalyzer, aggregate,
                              evaluation_metadata, extract_choice, score_answer)


class Tokenizer:
    # Content pieces deliberately include whitespace to test exact token counts.
    pieces = {1:'reason',2:' ',3:'Answer: B',4:'A',90:'<think>',91:'</think>',92:'<|im_end|>'}
    all_special_ids = [92]
    def convert_tokens_to_ids(self, text):
        return {v:k for k,v in self.pieces.items()}[text]
    def convert_ids_to_tokens(self, token):
        return self.pieces.get(token)
    def decode(self, tokens, **kwargs):
        return ''.join(self.pieces[t] for t in tokens)


class ScoringTests(unittest.TestCase):
    def test_exact_and_reference_rules(self):
        def score(answer, refs, rule):
            return score_answer(answer, evaluation_metadata(dict(references=refs, scoring=rule)))
        self.assertEqual(score('  FoO\nbar ', ['foo bar'], 'exact_match'),1)
        self.assertEqual(score('foo bar!', ['foo bar'], 'exact_match'),0)
        self.assertEqual(score('foo and baz', ['foo','bar','baz'], 'reference_coverage'),2/3)
        self.assertEqual(score('foO here', ['else','foo'], 'contains_any'),1)
        self.assertEqual(score('', ['foo'], 'reference_coverage'),0)
        self.assertIsNone(score('anything', [], 'exact_match'))

    def test_choice_and_no_guessing_ambiguous_responses(self):
        evaluation=evaluation_metadata(dict(scoring='choice',references=['B']))
        for answer in ('B', '(B)', 'The answer is B.', 'Final answer: B', 'A is wrong. Final answer: B'):
            self.assertEqual(score_answer(answer,evaluation),1, answer)
        for answer in ('A or B', 'Maybe', ''):
            self.assertEqual(score_answer(answer,evaluation),0, answer)
        self.assertIsNone(extract_choice('Both A and B'))

    def test_metadata_validation_and_import_aliases(self):
        self.assertEqual(evaluation_metadata(dict(task='qa',answers=['yes']))['subtask'],'qa')
        self.assertEqual(evaluation_metadata(dict(task='qa',references=['x']),dict(subtask='custom'))['subtask'],'custom')
        self.assertEqual(evaluation_metadata({})['subtask'],'unlabeled')
        for source in (dict(references=[1]),dict(references=['']),dict(scoring='unknown'),
                       dict(subtask=''),dict(scoring='choice',references=['ABC'])):
            with self.assertRaises(ValueError):
                evaluation_metadata(source)

    def test_native_thinking_explicit_thinking_and_truncated_outputs(self):
        analyzer=OutputAnalyzer(Tokenizer())
        cases=[([1,2,91,3,92],[90],True,(2,1,2,False,'Answer: B')),
               ([90,1,91,3,92],[],False,(1,1,3,False,'Answer: B')),
               ([1,2,92],[90],True,(2,0,1,True,'')),
               ([3,92],[90,91],True,(0,1,1,False,'Answer: B')),
               ([3,92],[],False,(0,1,1,False,'Answer: B')),
               ([91,3],[],True,(0,1,1,False,'Answer: B')),
               ([],[90],True,(0,0,0,True,''))]
        for tokens,prompt,thinking,expected in cases:
            with self.subTest(tokens=tokens,prompt=prompt):
                result=analyzer.analyze(tokens,prompt,thinking)
                self.assertEqual(tuple(result[k] for k in ('thinking_tokens','answer_tokens','control_tokens',
                                                           'unfinished_thinking','answer_text')),expected)
                self.assertEqual(sum(result[k] for k in ('thinking_tokens','answer_tokens','control_tokens')),len(tokens))
        # A correct option in reasoning cannot rescue a wrong final answer.
        result=analyzer.analyze([90,3,91,4,92],[],True)
        self.assertEqual(score_answer(result['answer_text'],dict(references=['B'],scoring='choice')),0)


def record(pid, task, ttft, accuracy, thinking=10, answer=2):
    return dict(prompt_id=pid,subtask=task,timings={'ttft_seconds':ttft},accuracy=accuracy,
                thinking_tokens=thinking,answer_tokens=answer,control_tokens=2,
                output_tokens=thinking+answer+2,output_cap_reached=False,unfinished_thinking=False)


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.expected=[dict(prompt_id=pid,subtask=task) for pid,task in
                       [('a','qa'),('b','qa'),('c','choice'),('d','missing')]]
        self.context=dict(method='prophetkv',ratio=.1,scoring_layers=list(range(64)),setup_fingerprint='fixed')
        self.records=[record('a','qa',1,1),record('b','qa',3,.5,20,4),record('c','choice',8,0,0,1)]

    def test_weighted_overall_per_subtask_empty_groups_and_missing_accuracy(self):
        report=aggregate(self.records,self.expected,self.context,'running')
        self.assertEqual(report['overall']['mean_ttft_seconds'],4)
        self.assertEqual(report['overall']['median_ttft_seconds'],3)
        self.assertEqual(report['overall']['accuracy_percent'],50)
        self.assertEqual(report['subtasks']['qa']['accuracy_percent'],75)
        self.assertEqual(report['subtasks']['qa']['mean_ttft_seconds'],2)
        self.assertEqual(report['overall']['mean_thinking_tokens'],10)
        self.assertEqual(report['overall']['total_answer_tokens'],7)
        self.assertEqual(report['subtasks']['missing']['completed'],0)
        self.assertIsNone(report['subtasks']['missing']['accuracy_percent'])
        records=self.records+[record('d','missing',4,None)]
        report=aggregate(records,self.expected,self.context,'completed')
        self.assertEqual(report['overall']['accuracy_scored'],3)
        self.assertEqual(report['overall']['accuracy_unscored'],1)
        self.assertEqual(report['overall']['accuracy_percent'],50)
        self.assertEqual(report['overall']['completed'],4)

    def test_reject_duplicate_invalid_counts_and_premature_final(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):
            aggregate(self.records,self.expected,self.context,'completed')
        with self.assertRaises(ValueError):
            aggregate(self.records+self.records[:1],self.expected,self.context,'running')
        for change in (dict(accuracy=float('nan')),dict(thinking_tokens=-1),dict(output_tokens=100),
                       dict(subtask='other'),dict(timings={'ttft_seconds':float('inf')})):
            rows=copy.deepcopy(self.records)
            rows[0].update(change)
            with self.assertRaises(ValueError):
                aggregate(rows,self.expected,self.context,'running')

    def test_live_snapshots_failure_and_successful_final_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            reporter=AggregationReporter(root,self.expected,self.context)
            live=json.loads((root/'live_aggregation.json').read_text())
            self.assertEqual(live['status'],'initializing')
            self.assertIsNone(live['overall']['mean_ttft_seconds'])
            for row in self.records:
                reporter.accept(row)
                self.assertFalse((root/'final_aggregation.json').exists())
            reporter.publish('live','failed',RuntimeError('engine stopped'))
            live=json.loads((root/'live_aggregation.json').read_text())
            self.assertEqual(live['overall']['completed'],3)
            self.assertEqual(live['error'],'engine stopped')
            with self.assertRaises(ValueError):
                reporter.finish()
            self.assertFalse((root/'final_aggregation.json').exists())
            reporter.accept(record('d','missing',4,None))
            final=reporter.finish()
            self.assertEqual(final['status'],'completed')
            self.assertEqual(final['overall'],json.loads((root/'live_aggregation.json').read_text())['overall'])
            rows=list(csv.DictReader(io.StringIO((root/'final_aggregation.csv').read_text())))
            self.assertEqual(len(rows),4)
            self.assertEqual(rows[0]['accuracy_percent'],'50.0')
            self.assertEqual(rows[0]['completed'],'4')
            self.assertIn('Mean thinking tokens',(root/'final_aggregation.md').read_text())
            self.assertFalse(list(root.glob('*.tmp')))


if __name__=='__main__':
    unittest.main()
