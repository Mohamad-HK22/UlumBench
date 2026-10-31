"""Score completed responses. Infrastructure failures never become model mistakes."""
import json
from .academic import parse_mcq_answer, _mcq_is_correct, is_refusal, parse_judge_label, JUDGE_SYSTEM_PROMPT
from .typed import score_call
from ulumbench.util import strict_json


def score(request, response, data, judge=None):
    raw = response['text']
    failed = response.get('finish_reason') != 'stop' or response.get('refusal') or is_refusal(raw)
    if request['scorer'] == 'academic':
        record = data.records[request['record_id']]
        if failed:
            return {'terminal_correct': False, 'format_valid': False, 'answer_status': 'generation_failure'}
        if record['question_type'] == 'multiple_choice':
            target = data.mcq[record['id']]
            parsed = parse_mcq_answer(raw, target['choices'], target['allowed_answers'])
            result = {'terminal_correct': parsed.status == 'valid' and _mcq_is_correct(
                        parsed.value, target['target'], target['target_kind'], target['choices']),
                      'format_valid': parsed.status == 'valid', 'answer_status': parsed.status}
            if request['experiment'] == 'answer_format' and request['condition'] == 'baseline':
                from ulumbench.experiments._permissive import parse_permissive_mcq_answer
                permissive = parse_permissive_mcq_answer(raw, target['choices'], target['allowed_answers'])
                result['permissive_correct'] = permissive.status == 'valid' and _mcq_is_correct(
                    permissive.value, target['target'], target['target_kind'], target['choices'])
            return result
        if judge is None:
            raise ValueError('A judge model is required for open-ended answers')
        # Always grade against the canonical original question, including its images.
        messages = [{'role': 'system', 'content': JUDGE_SYSTEM_PROMPT},
                    {'role': 'user', 'content': data.content(record['question'], record['assets'])},
                    {'role': 'user', 'content': json.dumps({'reference_answer': record['solution'],
                                                          'candidate_answer': raw}, ensure_ascii=False, sort_keys=True)}]
        judged = judge(messages)
        label = parse_judge_label(judged['text'])
        if judged.get('finish_reason') != 'stop' or label.status != 'valid':
            raise ValueError('Judge failed to return exactly CORRECT or INCORRECT; run remains incomplete')
        return {'terminal_correct': label.value == 'CORRECT', 'format_valid': True,
                'answer_status': 'graded', 'judge_label': label.value}
    if request['scorer'] == 'state_assistance':
        # Validate duplicate keys before invoking the retained typed parser.
        try:
            strict_json(raw)
        except (ValueError, TypeError):
            raw = ''
        result = score_call({**request, 'raw_response': '' if failed else raw}, data.families[request['family_id']])
        valid = result['format_valid'] and not failed
        steps = result['step_scores']
        return {'format_valid': valid, 'format_error': result['format_error'], 'steps': steps,
                'terminal_correct': valid and bool((result['final_score'] or steps[-1])['correct']),
                'whole_correct': valid and result['all_outputs_correct']}
    if request['scorer'] == 'dependency_depth':
        from ulumbench.experiments._depth_scoring import score_response
        return score_response('' if failed else raw, request, data.families)
    if request['scorer'] == 'error_recovery':
        from ulumbench.experiments._recovery import score_response
        injection = next(x for x in data.rows('diagnostic/error_injections.jsonl') if x['id'] == request['injection_id'])
        result = score_response('' if failed else raw, data.families[request['family_id']], injection, request['condition'])
        result['whole_correct'] = result['format_valid'] and all(x['gold_correct'] for x in result['steps'])
        return result
    raise ValueError('Unknown scorer: ' + request['scorer'])
