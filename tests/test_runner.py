import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from ulumbench.runner import Journal, run, render, read_rows
from ulumbench.prompts import academic_request
from ulumbench.util import digest


class FakeData:
    records = {'synthetic': {'id': 'synthetic', 'topic': 'Synthetic', 'question_type': 'short_answer',
                             'question': 'What is one plus one?', 'solution': '2', 'assets': []}}
    def content(self, question, assets):
        return [{'type': 'text', 'text': question}]


class RunnerTests(unittest.TestCase):
    def test_journal_does_not_repeat_and_binds_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def transport(payload):
                calls.append(payload); return {'text': '2', 'finish_reason': 'stop'}
            path = Path(tmp) / 'calls.jsonl'
            Journal(path, transport).call('x', {'model': 'test'})
            Journal(path, transport).call('x', {'model': 'test'})
            self.assertEqual(len(calls), 1)
            with self.assertRaises(ValueError):
                Journal(path, transport).call('x', {'model': 'changed'})

    def test_unresolved_attempt_needs_explicit_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'calls.jsonl'
            def fail(payload):
                raise TimeoutError('synthetic timeout')
            with self.assertRaises(TimeoutError):
                Journal(path, fail).call('x', {})
            with self.assertRaises(RuntimeError):
                Journal(path, fail).call('x', {})
            got = Journal(path, lambda _: {'text': 'ok'}, retry_errors=True).call('x', {})
            self.assertEqual(got['text'], 'ok')

    def test_resume_reuses_candidate_after_judge_transport_failure(self):
        request = {**academic_request(FakeData.records['synthetic']), 'experiment': 'reasoning_effort',
                   'call_id': 'synthetic:rep0', 'repetition': 0}
        config = {'max_tokens': 32, 'experiment': 'reasoning_effort', 'temperature': None}
        plan = {'partial': False, 'plan_sha256': digest([request])}
        payloads = []; fail_judge = [True]
        def transport(payload):
            payloads.append(payload)
            if payload['model'] == 'judge' and fail_judge[0]:
                raise TimeoutError()
            return {'text': 'CORRECT' if payload['model'] == 'judge' else '2', 'finish_reason': 'stop'}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(TimeoutError):
                run(FakeData(), config, [request], plan, tmp, 'candidate', 'judge', transport)
            fail_judge[0] = False
            report = run(FakeData(), config, [request], plan, tmp, 'candidate', 'judge', transport, retry_errors=True)
            self.assertTrue(report['complete'])
            self.assertEqual(sum(p['model'] == 'candidate' for p in payloads), 1)
            run(FakeData(), config, [request], plan, tmp, 'candidate', 'judge', transport)
            self.assertEqual(len(payloads), 3)
            with self.assertRaises(ValueError):
                run(FakeData(), {**config, 'max_tokens': 64}, [request], plan, tmp, 'candidate', 'judge', transport)

    def test_history_uses_generated_answers_and_requires_dependencies(self):
        request = {'repetition': 0, 'prompt_segments': [
            {'kind': 'text', 'text': 'Use '}, {'kind': 'answer', 'prefix': 'prior:', 'record_id': 'prev'}]}
        self.assertEqual(render(request, {'prev:D:rep0': {'text': 'MODEL VALUE'}}), 'Use prior: MODEL VALUE')
        with self.assertRaises(KeyError):
            render(request, {})

    def test_corrupt_journal_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'calls.jsonl'; p.write_text('{"call_id":')
            with self.assertRaises(ValueError):
                read_rows(p)

    def test_invalid_judge_output_can_retry_without_new_candidate(self):
        request = {**academic_request(FakeData.records['synthetic']), 'experiment': 'reasoning_effort',
                   'call_id': 'synthetic:rep0', 'repetition': 0}
        config = {'max_tokens': 32, 'experiment': 'reasoning_effort'}
        plan = {'partial': False}
        calls = []; answer = ['not a valid verdict']
        def transport(payload):
            calls.append(payload['model'])
            return {'text': answer[0] if payload['model'] == 'judge' else '2', 'finish_reason': 'stop', 'usage': {'cost': 1}}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                run(FakeData(),config,[request],plan,tmp,'candidate','judge',transport)
            answer[0] = 'CORRECT'
            report = run(FakeData(),config,[request],plan,tmp,'candidate','judge',transport,retry_errors=True)
            self.assertTrue(report['complete'])
            self.assertEqual(calls, ['candidate','judge','judge'])
            self.assertEqual(report['observed_cost'], 3)
