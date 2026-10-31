"""One OpenRouter attempt per call; retries are an explicit operator decision."""
import os
import time
import requests


class OpenRouter:
    def __init__(self, timeout=300):
        self.key = os.environ.get('OPENROUTER_API_KEY')
        if not self.key:
            raise ValueError('Set OPENROUTER_API_KEY before running live evaluations')
        self.timeout = timeout

    def __call__(self, payload):
        start = time.monotonic()
        response = requests.post('https://openrouter.ai/api/v1/chat/completions',
                                 json=payload, timeout=self.timeout,
                                 headers={'Authorization': 'Bearer ' + self.key,
                                          'HTTP-Referer': 'https://ulumbench.com',
                                          'X-Title': 'UlumBench'})
        if response.status_code != 200:
            raise RuntimeError(f'OpenRouter HTTP {response.status_code}; no automatic retry')
        data = response.json()
        if data.get('error') or not data.get('choices'):
            raise RuntimeError('Provider returned an error or no choices')
        choice = data['choices'][0]
        message = choice.get('message', {})
        content = message.get('content') or ''
        if not isinstance(content, str):
            raise RuntimeError('Provider returned unsupported content format')
        return {'text': content, 'finish_reason': choice.get('finish_reason'),
                'refusal': bool(message.get('refusal')), 'usage': data.get('usage', {}),
                'provider': data.get('provider'), 'returned_model': data.get('model'),
                'generation_id': data.get('id'), 'latency_seconds': time.monotonic() - start}
