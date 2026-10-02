import copy
import unittest
from runner.result_comparison import IDENTITY_FIELDS,OUTPUT_FIELDS,compare_result,summarize


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.result=dict.fromkeys(IDENTITY_FIELDS,'same')
        self.result.update(dict.fromkeys(OUTPUT_FIELDS,0))
        self.result.update(token_sha256='tokens',executed_action='prophetkv-1',output_token_ids=[1,2],prediction='answer',accuracy=1.)
        self.old=dict(self.result,prompt_id='old-row',result_sha256='pinned',ttft_seconds=2.)
        self.reference={'tokens':{'prophetkv-1':self.old}}

    def test_exact_output_match_ignores_new_row_id_runtime_and_timing(self):
        new=dict(self.result,prompt_id='new-row',generator_config_sha256='new-batch',timings=dict(ttft_seconds=3.))
        comparison=compare_result(new,self.reference)
        self.assertEqual(comparison['status'],'matched')
        self.assertEqual(comparison['old_prompt_id'],'old-row')

    def test_different_output_and_incompatible_protocol_are_distinct(self):
        new=copy.deepcopy(self.result);new['output_token_ids']=[1,3]
        value=compare_result(new,self.reference);self.assertEqual(value['status'],'different');self.assertFalse(value['fields']['output_token_ids'])
        new=dict(self.result,max_output_tokens=256)
        value=compare_result(new,self.reference);self.assertEqual(value['status'],'incompatible');self.assertIn('max_output_tokens',value['fields'])

    def test_new_inputs_are_not_counted_as_matches(self):
        new=dict(self.result,token_sha256='new')
        comparison=compare_result(new,self.reference);self.assertEqual(comparison['status'],'new_input')
        summary=summarize([dict(historical_comparison=comparison),dict(historical_comparison=compare_result(self.result,self.reference))])
        self.assertEqual(summary,dict(compared=1,matched=1,different=0,incompatible=0,new_inputs=1))
