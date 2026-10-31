"""Descriptive metrics for new evaluations; no historical table reconstruction."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from .runner import read_rows


def summarize(output):
    output = Path(output)
    run = json.loads((output / 'run.json').read_text())
    requests = read_rows(output / 'requests.jsonl')
    scores = {r['call_id']: r['score'] for r in read_rows(output / 'scores.jsonl')}
    calls = {r['call_id']: r for r in read_rows(output / 'calls.jsonl')}
    groups = defaultdict(list)
    for r in requests:
        if not r['scored']:
            continue
        scopes = ['all'] + (['primary'] if r.get('primary_panel') else [])
        for scope in scopes:
            key = (scope, r['condition'], r.get('horizon'), r.get('position') if r['experiment'] == 'error_recovery' else None, r.get('native_length'))
            groups[key].append(r)
    table = []
    for (scope, condition, horizon, position, length), selected in groups.items():
        completed = [(r, scores[r['call_id']]) for r in selected if scores.get(r['call_id']) is not None]
        complete = len(completed) == len(selected)
        row = {'scope': scope, 'condition': condition, 'horizon': horizon, 'position': position,
               'native_length': length, 'planned': len(selected), 'scored': len(completed), 'complete': complete}
        for field in ['terminal_correct', 'whole_correct', 'candidate_state_correct', 'permissive_correct', 'format_valid']:
            values = [s[field] for _, s in completed if field in s]
            if values:
                row[field + '_rate'] = mean(values) if complete else None
        topics = defaultdict(list)
        for request, s in completed:
            if 'topic' in request:
                topics[request['topic']].append(s['terminal_correct'])
        if topics:
            row['topic_balanced_accuracy'] = mean(mean(v) for v in topics.values()) if complete else None
        token_values = []
        for r in selected:
            call = calls.get(r['call_id'], {})
            usage = call.get('response', {}).get('usage') or {}
            tokens = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
            if tokens is not None:
                token_values.append(tokens)
        row['reasoning_telemetry_count'] = len(token_values)
        row['median_reasoning_tokens'] = median(token_values) if token_values else None
        table.append(row)
    completed_calls = [c for c in read_rows(output / 'calls.jsonl') if c['status'] == 'completed']
    costs = [(c['response'].get('usage') or {}).get('cost') for c in completed_calls]
    return {'experiment': run['config']['experiment'], 'model': run['model'],
            'complete': len(scores) == len(requests), 'partial_selection': run['plan']['partial'],
            'candidate_calls_planned': len(requests), 'candidate_calls_scored_or_unscored_history': len(scores),
            'observed_cost': sum(v for v in costs if v is not None) if any(v is not None for v in costs) else None,
            'cost_telemetry_calls': sum(v is not None for v in costs), 'completed_provider_calls': len(costs),
            'metrics': sorted(table, key=lambda r: str((r['scope'],r['condition'],r['horizon'],r['position'],r['native_length']))),
            'note': 'Descriptive rates only. Incomplete groups have no accuracy; partial selections are not full benchmark results.'}
