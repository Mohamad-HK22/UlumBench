"""Optional full-data validation: set ULUMBENCH_DATA to a downloaded snapshot."""
import json
import os
import unittest
from pathlib import Path
from collections import Counter
from ulumbench.data import Dataset
from ulumbench.protocol import make_plan
from ulumbench.scoring.typed import score_answer
from ulumbench.scoring.dispatch import score


@unittest.skipUnless(os.environ.get('ULUMBENCH_DATA'), 'Set ULUMBENCH_DATA for full-data checks')
class DatasetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Dataset(os.environ['ULUMBENCH_DATA'])

    def test_panel_and_asset_integrity(self):
        data = self.data
        eligible = [r for r in data.records.values() if r['evaluation_eligible']]
        self.assertEqual((len(data.records), len(eligible)), (6295, 6167))
        self.assertEqual(sum(r['question_type'] == 'multiple_choice' for r in eligible), 4158)
        self.assertEqual(sum(bool(r['assets']) for r in eligible), 743)
        images = {a['path'] for r in data.records.values() for a in r['assets']}
        self.assertEqual(len(images), 605)
        self.assertTrue(all(data.path(p).is_file() for p in images))
        self.assertEqual(len({r['topic'] for r in eligible}), 140)
        for family in data.families.values():
            self.assertTrue(set(family['source_ids']).issubset(data.records))
            seen = set()
            for step in family['steps']:
                self.assertTrue(set(step['depends_on']).issubset(seen))
                seen.add(step['step_id'])
                self.assertTrue(score_answer(step['gold_answer'], step)['correct'], (family['family_id'],step['step_id']))

    def test_full_schedules_and_gold_responses(self):
        counts = {'state_assistance': 1600, 'dependency_depth': 12000, 'error_recovery': 3600, 'reasoning_effort': 1974}
        for experiment, expected in counts.items():
            cfg = {'experiment': experiment, 'repetitions': 3 if experiment in ['dependency_depth','error_recovery'] else 1}
            requests, plan = make_plan(self.data, cfg)
            self.assertEqual(len(requests), expected)
            self.assertEqual(len({r['call_id'] for r in requests}), expected)
            if experiment == 'reasoning_effort':
                continue
            # One repetition covers every distinct response schema and every real reference.
            for r in (r for r in requests if r['repetition'] == 0):
                f = self.data.families[r['family_id']]
                if experiment == 'state_assistance':
                    if r['condition'] == 'atomic':
                        raw = {'family_id': f['family_id'], 'step_id': r['step_id'],
                               'answer': f['steps'][int(r['step_id'].split('_')[1])-1]['gold_answer']}
                    else:
                        raw = {'family_id': f['family_id'], 'horizon': r['horizon'],
                               'steps': [{'step_id':s['step_id'],'answer':s['gold_answer']} for s in f['steps'][:r['horizon']]],
                               'final_answer': f['steps'][r['horizon']-1]['gold_answer']}
                elif experiment == 'dependency_depth':
                    raw = {'request_id':r['request_id'],'answers':[
                        {'task_id':t['task_id'],'answer':self.data.families[t['family_id']]['steps'][t['position']-1]['gold_answer']} for t in r['tasks']]}
                else:
                    raw = {'request_id':r['request_id'],'answers':[
                        {'step_id':s['step_id'],'answer':s['gold_answer']} for s in f['steps'][r['position']-1:]]}
                result = score(r, {'text':json.dumps(raw),'finish_reason':'stop'}, self.data)
                self.assertTrue(result['terminal_correct'], r['call_id'])
                self.assertTrue(result['whole_correct'], r['call_id'])
