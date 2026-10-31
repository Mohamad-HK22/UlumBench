"""Within-model low, medium and high effort on the fixed CS panel."""
from ulumbench.prompts import academic_request


def build(data, config):
    requests = []
    for rid in data.read('metadata/reasoning_effort_panel.json'):
        record = data.records[rid]
        if not record['evaluation_eligible']:
            raise ValueError('Panel includes an ineligible item: ' + rid)
        for effort in ['low', 'medium', 'high']:
            requests.append({**academic_request(record, 'effort_' + effort), 'reasoning_effort': effort})
    return requests
