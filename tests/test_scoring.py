import json
import unittest
from ulumbench.scoring.academic import parse_mcq_answer, _mcq_is_correct, parse_judge_label
from ulumbench.scoring.typed import score_answer
from ulumbench.scoring.dispatch import score
from ulumbench.experiments._depth_scoring import parse_response
from ulumbench.util import strict_json


def step(gold='2', answer_type='number', tolerance=0):
    return {'step_id': 'step_1', 'gold_answer': gold, 'answer_type': answer_type,
            'tolerance': tolerance, 'evaluation_method': {'number': 'exact_tolerance', 'number_with_unit': 'numeric_units'}.get(answer_type, 'exact_text')}


class ScoringTests(unittest.TestCase):
    def test_choice_resolution_and_conflict(self):
        choices = [{'label': 'A', 'text': 'Red'}, {'label': 'B', 'text': 'Blue'}]
        self.assertEqual(parse_mcq_answer('Answer: B', choices).value, 'b')
        self.assertEqual(parse_mcq_answer('blue', choices).value, 'b')
        self.assertEqual(parse_mcq_answer('A or B', choices).status, 'invalid')
        self.assertTrue(_mcq_is_correct('b', 'Blue', 'choice_text', choices))
        self.assertEqual(parse_mcq_answer('A', choices + [choices[0]]).status, 'invalid')

    def test_judge_contract(self):
        self.assertEqual(parse_judge_label('CORRECT').status, 'valid')
        self.assertEqual(parse_judge_label('Probably CORRECT').status, 'invalid')

    def test_scalar_tolerance_and_reference_format(self):
        self.assertTrue(score_answer('2.005', step(tolerance=.01))['correct'])
        self.assertFalse(score_answer('2.02', step(tolerance=.01))['correct'])
        self.assertFalse(score_answer('4/2', step())['correct'])
        self.assertFalse(score_answer('NaN', step())['correct'])

    def test_units_and_leading_zeros(self):
        self.assertTrue(score_answer('2 joules', step('2 J', 'number_with_unit'))['correct'])
        self.assertFalse(score_answer('0.002 kJ', step('2 J', 'number_with_unit'))['correct'])
        self.assertFalse(score_answer('10', step('0010', 'bitstring'))['correct'])

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            strict_json('{"x": 1, "x": 2}')
        request = {'request_id': 'x', 'tasks': [{'task_id': 'task_1'}]}
        raw = '{"request_id":"x","answers":[{"task_id":"task_1","answer":"bad","answer":"2"}]}'
        self.assertFalse(parse_response(raw, request)['valid'])

    def test_depth_order_is_enforced(self):
        request = {'request_id': 'x', 'tasks': [{'task_id': 'task_1'}, {'task_id': 'task_2'}]}
        raw = json.dumps({'request_id': 'x', 'answers': [{'task_id': 'task_2', 'answer': '2'}, {'task_id': 'task_1', 'answer': '2'}]})
        self.assertFalse(parse_response(raw, request)['valid'])

    def test_invalid_outer_json_and_truncation_cannot_pass(self):
        class Data:
            families = {'synthetic': {'family_id': 'synthetic', 'steps': [step()]}}
        request = {'request_id': 'synthetic:atomic:step_1', 'family_id': 'synthetic',
                   'scorer': 'state_assistance', 'condition': 'atomic', 'step_id': 'step_1'}
        raw = json.dumps({'family_id': 'wrong', 'step_id': 'step_1', 'answer': '2'})
        self.assertFalse(score(request, {'text': raw, 'finish_reason': 'stop'}, Data())['terminal_correct'])
        raw = raw.replace('wrong', 'synthetic')
        self.assertTrue(score(request, {'text': raw, 'finish_reason': 'stop'}, Data())['terminal_correct'])
        self.assertFalse(score(request, {'text': raw, 'finish_reason': 'length'}, Data())['terminal_correct'])
