"""Matched operation pools with independent versus dependent execution."""
from ulumbench.prompts import DEPTH
from ._depth_design import build_requests


def build(data, config):
    protocol = {'panel': {'block_size': 10, 'matching_seed': 271828, 'horizons': [2, 4, 6, 8]},
                'generation': {'system_prompt': DEPTH}}
    requests = build_requests(list(data.families.values()), protocol)
    for request in requests:
        request.update(scorer='dependency_depth', scored=True,
                       primary_panel=request['native_length'] == 8)
    return requests


def response_format(request):
    return {'type': 'json_schema', 'json_schema': {'name': 'ulumbench_answers', 'strict': True,
            'schema': {'type': 'object', 'additionalProperties': False,
                       'required': ['request_id', 'answers'], 'properties': {
                           'request_id': {'type': 'string', 'enum': [request['request_id']]},
                           'answers': {'type': 'array', 'items': {
                               'type': 'object', 'additionalProperties': False,
                               'required': ['task_id', 'answer'], 'properties': {
                                   'task_id': {'type': 'string', 'enum': [t['task_id'] for t in request['tasks']]},
                                   'answer': {'type': 'string'}}}}}}}}
