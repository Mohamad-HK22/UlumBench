"""Atomic, oracle-assisted and composed execution of nested prefixes."""
from ulumbench.prompts import STATE
from ._state_prompts import _atomic_prompt, _oracle_bundle_prompt, _composed_prompt


def build(data, config):
    result = []
    for f in sorted(data.families.values(), key=lambda f: f['family_id']):
        if not f['primary_panel']:
            continue
        common = {'family_id': f['family_id'], 'quality_flags': f['quality_flags'],
                  'scorer': 'state_assistance', 'system_prompt': STATE, 'scored': True}
        for position in range(1, 9):
            sid = f'step_{position}'
            result.append({**common, 'request_id': f"{f['family_id']}:atomic:{sid}",
                           'condition': 'atomic', 'step_id': sid, 'horizon': position,
                           'user_prompt': _atomic_prompt(f, position)})
        for condition, builder in [('oracle_bundle', _oracle_bundle_prompt), ('composed', _composed_prompt)]:
            for horizon in [2, 4, 6, 8]:
                result.append({**common, 'request_id': f"{f['family_id']}:{condition}:h{horizon}",
                               'condition': condition, 'horizon': horizon,
                               'user_prompt': builder(f, horizon)})
    return result
