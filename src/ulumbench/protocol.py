"""Discover installed experiments and construct a repeatable dependency-safe schedule."""
import importlib
import importlib.util
import json
import random
from pathlib import Path
from .util import digest

NAMES = ('academic', 'answer_format', 'context', 'state_assistance',
         'dependency_depth', 'error_recovery', 'reasoning_effort')


def available():
    return [n for n in NAMES if importlib.util.find_spec('ulumbench.experiments.' + n)]


def load_config(path):
    config = json.loads(Path(path).read_text())
    keys = {'experiment', 'repetitions', 'max_tokens', 'temperature', 'reasoning_effort',
            'schedule_seed', 'structured_output'}
    if set(config) - keys:
        raise ValueError('Unknown configuration fields: ' + str(set(config) - keys))
    if config['experiment'] not in available():
        raise ValueError('Experiment is not installed: ' + config['experiment'])
    for key in ['repetitions', 'max_tokens']:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if config.get('reasoning_effort') not in {None, 'low', 'medium', 'high'}:
        raise ValueError('Unknown reasoning effort')
    temperature = config.get('temperature')
    if temperature is not None and (isinstance(temperature, bool) or not 0 <= temperature <= 2):
        raise ValueError('temperature must be null or between 0 and 2')
    if config.get('structured_output') and config['experiment'] != 'dependency_depth':
        raise ValueError('Native structured output is only implemented for dependency_depth')
    return config


def make_plan(data, config, limit=None):
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    module = importlib.import_module('ulumbench.experiments.' + config['experiment'])
    base = module.build(data, config)
    if len({r['request_id'] for r in base}) != len(base):
        raise ValueError('Duplicate request IDs')
    all_requests = []
    for repeat in range(config['repetitions']):
        for request in base:
            all_requests.append({**request, 'experiment': config['experiment'], 'repetition': repeat,
                                 'call_id': request['request_id'] + f':rep{repeat}',
                                 'depends_on_calls': [s + f':rep{repeat}' for s in request.get('dependencies', [])]})
    # Seeded topological ordering: generated history always precedes its consumers.
    index = {r['call_id']: r for r in all_requests}
    children = {key: [] for key in index}
    degree = {}
    for key, request in index.items():
        deps = set(request['depends_on_calls'])
        if not deps.issubset(index):
            raise ValueError('Missing history prerequisites: ' + str(deps - index.keys()))
        degree[key] = len(deps)
        for dep in deps:
            children[dep].append(key)
    rng = random.Random(config.get('schedule_seed', 314159))
    ready = sorted(key for key, n in degree.items() if not n)
    result = []
    while ready:
        i = rng.randrange(len(ready)); key = ready[i]
        ready[i] = ready[-1]; ready.pop()
        result.append(index[key])
        for child in children[key]:
            degree[child] -= 1
            if not degree[child]:
                ready.append(child)
    if len(result) != len(index):
        raise ValueError('Cyclic history dependencies')
    selected = result[:limit] if limit is not None else result
    return selected, {'experiment': config['experiment'], 'full_candidate_calls': len(result),
                      'selected_candidate_calls': len(selected), 'partial': len(selected) != len(result),
                      'plan_sha256': digest(selected), 'dataset_sha256': data.fingerprint,
                      'maximum_judge_calls': sum(r['scored'] and r['scorer'] == 'academic' and
                                                r['question_type'] != 'multiple_choice' for r in selected)}
