import os
import unittest
from ulumbench.data import Dataset
from ulumbench.protocol import make_plan
from ulumbench.experiments._context_prompts import render_prompt
from ulumbench.runner import render


@unittest.skipUnless(os.environ.get('ULUMBENCH_DATA'), 'Set ULUMBENCH_DATA for full-data checks')
class ContextIntegrationTests(unittest.TestCase):
    def test_all_academic_schedules_and_context_dependency_closure(self):
        data = Dataset(os.environ['ULUMBENCH_DATA'])
        for name, count in [('academic',6167),('answer_format',8316),('context',5064)]:
            cfg = {'experiment':name,'repetitions':1}
            requests, plan = make_plan(data,cfg)
            self.assertEqual(len(requests),count)
            if name != 'context':
                continue
            scored_d = [r for r in requests if r['condition']=='D' and r['scored']]
            self.assertEqual(len(scored_d),1081)
            self.assertEqual(sum(r['condition']=='C' for r in requests),1102)
            generated = {}
            for request in requests:
                self.assertTrue(set(request['depends_on_calls']).issubset(generated))
                if request['condition']=='D':
                    prompt = render(request,generated)
                    self.assertNotIn('reference_answer',prompt)
                    for segment in request['prompt_segments']:
                        if segment['kind']=='answer':
                            self.assertIn('synthetic-history-'+segment['record_id'],prompt)
                generated[request['call_id']]={'text':'synthetic-history-'+request['record_id']}
