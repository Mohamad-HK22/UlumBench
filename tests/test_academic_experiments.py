import unittest
from ulumbench.experiments import academic, answer_format
from ulumbench.experiments._permissive import parse_permissive_mcq_answer
from ulumbench.prompts import LABEL_ONLY, ACADEMIC


class AcademicExperimentTests(unittest.TestCase):
    def test_prompt_and_eligibility(self):
        class Data:
            records = {'one': {'id': 'one', 'question': 'Synthetic question', 'topic': 'Synthetic', 'assets': [],
                               'question_type': 'multiple_choice', 'evaluation_eligible': True},
                       'two': {'id': 'two', 'question': 'Synthetic open question', 'topic': 'Synthetic', 'assets': [],
                               'question_type': 'short_answer', 'evaluation_eligible': True},
                       'bad': {'id': 'bad', 'evaluation_eligible': False}}
        requests = academic.build(Data(), {})
        self.assertEqual([r['system_prompt'] for r in requests], [LABEL_ONLY, ACADEMIC])
        requests = answer_format.build(Data(), {})
        self.assertEqual([r['condition'] for r in requests], ['baseline', 'label_only'])
        self.assertNotIn('solution', requests[0])

    def test_permissive_extraction_is_gold_blind_and_conflict_sensitive(self):
        choices = [{'label': 'A', 'text': 'Red'}, {'label': 'B', 'text': 'Blue'}]
        parsed = parse_permissive_mcq_answer('Reasoning here. Final answer: B', choices, ['a', 'b'])
        self.assertEqual(parsed.value, 'b')
        parsed = parse_permissive_mcq_answer('Answer: A\nFinal answer: B', choices, ['a', 'b'])
        self.assertEqual(parsed.status, 'invalid')
