"""Stable serialization and strict response parsing."""
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_hash():
    root = Path(__file__).parent
    return digest({p.relative_to(root).as_posix(): file_hash(p)
                   for p in sorted(root.rglob('*')) if p.suffix in {'.py', '.json'}})


def runtime_versions():
    import platform
    from importlib.metadata import version
    return {'python': platform.python_version(), 'packages': {
        name: version(name) for name in ['requests', 'huggingface-hub', 'filelock', 'numpy', 'scipy']}}
