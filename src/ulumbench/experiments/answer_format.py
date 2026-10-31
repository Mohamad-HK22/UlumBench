"""Two generated prompt arms; baseline is also rescored without a new call."""
from ulumbench.prompts import academic_request


def build(data, config):
    result = []
    for record in data.records.values():
        if not record['evaluation_eligible'] or record['question_type'] != 'multiple_choice':
            continue
        result += [academic_request(record, 'baseline'), academic_request(record, 'label_only', True)]
    return result
