"""Original, stripped, independently answerable and generated-history contexts."""
from ulumbench.prompts import ACADEMIC
from ._context_prompts import trajectory_items


def build(data, config):
    items = data.rows('metadata/context.jsonl')
    requests = []
    def request(item, arm, scored=True):
        r = {'request_id': item['id'] + ':' + arm, 'record_id': item['id'],
             'condition': arm, 'group_id': item['group_id'], 'position': item['position'],
             'scorer': 'academic', 'system_prompt': ACADEMIC, 'assets': item['assets'],
             'scored': scored, 'primary_panel': arm != 'C' and item.get('has_retained_gold_history', False),
             'topic': data.records[item['id']]['topic'], 'question_type': item['question_type']}
        if arm == 'D':
            r['prompt_segments'] = item['prompt_segments']
            r['dependencies'] = [s['record_id'] + ':D' for s in item['prompt_segments'] if s['kind'] == 'answer']
        else:
            r['user_prompt'] = item['prompt_' + arm.lower()]
        return r
    for item in items:
        for arm in ['A', 'B']:
            requests.append(request(item, arm))
        if item['independently_answerable']:
            requests.append(request(item, 'C'))
    for node in trajectory_items(items):
        r = request(node, 'D', node['is_scored_target'])
        r['primary_panel'] = node['is_scored_target']
        requests.append(r)
    return requests
