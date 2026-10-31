"""Durable generation and grading journal, bound to code, data, plan and settings."""
import json
import os
from pathlib import Path
from filelock import FileLock
from .util import canonical, digest, implementation_hash, runtime_versions
from .scoring.dispatch import score
from .scoring.academic import parse_judge_label


def append(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(canonical(row) + '\n'); stream.flush(); os.fsync(stream.fileno())


def read_rows(path):
    if not path.exists():
        return []
    try:
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    except ValueError as exc:
        raise ValueError(f'Incomplete journal at {path}; preserve and repair its last line before resuming') from exc


class Journal:
    def __init__(self, path, transport, retry_errors=False):
        self.path, self.transport, self.retry_errors = path, transport, retry_errors
        self.latest = {r['call_id']: r for r in read_rows(path)}

    def reject(self, key, reason):
        row = {**self.latest[key], 'status': 'error', 'error_type': reason}
        append(self.path, row)
        self.latest[key] = row

    def call(self, key, payload):
        fingerprint = digest(payload)
        old = self.latest.get(key)
        if old:
            if old['payload_sha256'] != fingerprint:
                raise ValueError('Changed request payload for ' + key)
            if old['status'] == 'completed':
                return old['response']
            if not self.retry_errors:
                raise RuntimeError('Unresolved provider attempt for ' + key +
                                   '; use --retry-errors only after checking possible charges')
        base = {'call_id': key, 'payload_sha256': fingerprint}
        append(self.path, {**base, 'status': 'pending'})
        try:
            response = self.transport(payload)
        except Exception as exc:
            row = {**base, 'status': 'error', 'error_type': type(exc).__name__}
            append(self.path, row); self.latest[key] = row
            raise
        row = {**base, 'status': 'completed', 'response': response}
        append(self.path, row); self.latest[key] = row
        return response


def render(request, generated):
    if 'prompt_segments' not in request:
        return request['user_prompt']
    result = []
    for segment in request['prompt_segments']:
        if segment['kind'] == 'text':
            result.append(segment['text'])
        elif segment['kind'] == 'answer':
            key = segment['record_id'] + f":D:rep{request['repetition']}"
            value = generated[key]['text']
            result.append(segment['prefix'] + ('' if not value or value[0].isspace() else ' ') + value)
        else:
            raise ValueError('Invalid history segment')
    return ''.join(result)


def run(data, config, requests, plan, output, model, judge_model, transport,
        provider=None, retry_errors=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.run.lock'), timeout=0):
        identity = {'config': config, 'plan': plan, 'model': model, 'judge_model': judge_model,
                    'provider': provider, 'implementation_sha256': implementation_hash(), 'runtime': runtime_versions()}
        manifest = output / 'run.json'
        if manifest.exists():
            if json.loads(manifest.read_text(encoding='utf-8')) != identity:
                raise ValueError('Run identity changed; use a new output directory')
        else:
            manifest.write_text(canonical(identity) + '\n', encoding='utf-8')
            (output / 'requests.jsonl').write_text(''.join(canonical(r) + '\n' for r in requests), encoding='utf-8')
        journal = Journal(output / 'calls.jsonl', transport, retry_errors)
        prior = {r['call_id']: r for r in read_rows(output / 'scores.jsonl')}
        generated = {}
        for i, request in enumerate(requests, 1):
            key = request['call_id']
            if key in prior:
                generated[key] = journal.latest[key]['response']
                continue
            prompt = render(request, generated)
            routing = {'allow_fallbacks': False, 'require_parameters': True}
            if provider:
                routing['only'] = [provider]
            payload = {'model': model, 'messages': [
                {'role': 'system', 'content': request['system_prompt']},
                {'role': 'user', 'content': data.content(prompt, request.get('assets', []))}],
                'max_tokens': config['max_tokens'], 'provider': routing, 'stream': False}
            effort = request.get('reasoning_effort', config.get('reasoning_effort'))
            if effort:
                payload['reasoning'] = {'effort': effort, 'exclude': True}
            if config.get('temperature') is not None:
                payload['temperature'] = config['temperature']
            if config.get('structured_output'):
                from .experiments.dependency_depth import response_format
                payload['response_format'] = response_format(request)
            response = journal.call(key, payload)
            generated[key] = response
            def judge(messages):
                judge_key = key + ':judge'
                judged = journal.call(judge_key, {'model': judge_model, 'messages': messages,
                                    'max_tokens': 8, 'temperature': 0, 'stream': False,
                                    'provider': {'allow_fallbacks': False, 'require_parameters': True}})
                if judged.get('finish_reason') != 'stop' or parse_judge_label(judged['text']).status != 'valid':
                    journal.reject(judge_key, 'invalid_judge_output')
                    raise ValueError('Invalid judge output saved; --retry-errors can retry grading only')
                return judged
            result = score(request, response, data, judge if judge_model else None) if request['scored'] else None
            append(output / 'scores.jsonl', {'call_id': key, 'score': result})
            print(f'{i}/{len(requests)} {key}', flush=True)
    from .reporting import summarize
    report = summarize(output)
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report
