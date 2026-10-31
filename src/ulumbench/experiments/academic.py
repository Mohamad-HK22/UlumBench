"""Academic panel: label-only MCQs and final-answer open-ended questions."""
from ulumbench.prompts import academic_request


def build(data, config):
    return [academic_request(r, 'academic', r['question_type'] == 'multiple_choice')
            for r in data.records.values() if r['evaluation_eligible']]
