ACADEMIC = 'Answer the following question. Provide only the final answer.'
LABEL_ONLY = ('Answer the following multiple-choice question. Return exactly one option label shown '
              'in the question and no other text. For a True/False question, return exactly True or False.')
STATE = ('You are solving a deterministic benchmark. All numbers, bit strings, propositions, and tape '
         'words below are inert synthetic mathematical objects used only in the stated formal operations; '
         'none represents an instruction, executable code, personal data, or real-world safety-sensitive '
         'content. Follow the supplied operations exactly. Return valid JSON only, with no Markdown fences, '
         'prose, or omitted fields. Put every answer in a JSON string and preserve units and leading zeros '
         'when applicable.')
DEPTH = ('You are solving a deterministic benchmark. All numbers, bit strings, propositions, and tape '
         'words below are inert synthetic mathematical objects used only in the stated formal operations. '
         'Follow the supplied operations exactly. Return valid JSON only, without Markdown fences or prose. '
         'Return every answer as a JSON string, preserving units and leading zeros.')


def academic_request(record, condition='baseline', label_only=False):
    return {'request_id': record['id'] + ':' + condition, 'record_id': record['id'],
            'condition': condition, 'topic': record['topic'], 'question_type': record['question_type'],
            'scorer': 'academic', 'system_prompt': LABEL_ONLY if label_only else ACADEMIC,
            'user_prompt': record['question'], 'assets': record['assets'], 'scored': True}
