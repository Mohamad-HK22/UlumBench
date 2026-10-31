"""Continue or verify a correct or perturbed intermediate state."""
from ._recovery import prompt


def build(data, config):
    requests = []
    for injection in data.rows('diagnostic/error_injections.jsonl'):
        if not injection['live_eligible']:
            continue
        family = data.families[injection['family_id']]
        for arm in 'ABCD':
            requests.append({'request_id': f"{injection['id']}:{arm}",
                             'injection_id': injection['id'], 'family_id': family['family_id'],
                             'condition': arm, 'position': injection['position'],
                             'primary_panel': injection['primary_eligible'],
                             'terminal_repair_inference_eligible': injection['terminal_repair_inference_eligible'],
                             'system_prompt': 'Solve the supplied academic problem. Follow the requested JSON output format.',
                             'user_prompt': prompt(family, injection, arm),
                             'scorer': 'error_recovery', 'scored': True})
    return requests
