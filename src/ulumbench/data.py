"""Load an immutable HF snapshot; no benchmark content is shipped with the code."""
import base64
import json
from importlib.resources import files
from pathlib import Path, PurePosixPath
from .util import digest, file_hash


def download(revision=None, local_dir=None):
    from huggingface_hub import snapshot_download
    spec = json.loads(files('ulumbench').joinpath('dataset.json').read_text())
    return Path(snapshot_download(repo_id=spec['repo_id'], repo_type='dataset',
                                  revision=revision or spec['revision'], local_dir=local_dir or Path('datasets/ulumbench')))


class Dataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest['schema_version'] != '1.0':
            raise ValueError('Unsupported dataset schema version')
        for name, expected in self.manifest['files'].items():
            if file_hash(self.path(name)) != expected:
                raise ValueError('Dataset checksum mismatch: ' + name)
        self.fingerprint = digest(self.manifest)
        self.records = {r['id']: r for r in self.rows('academic/records.jsonl')}
        self.families = {r['family_id']: r for r in self.rows('diagnostic/families.jsonl')}
        self.mcq = {r['id']: r for r in self.rows('metadata/mcq_targets.jsonl')}

    def path(self, name):
        relative = PurePosixPath(name)
        path = self.root.joinpath(*relative.parts).resolve()
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(self.root):
            raise ValueError('Path outside the dataset: ' + name)
        return path

    def rows(self, name):
        return [json.loads(s) for s in self.path(name).read_text().splitlines() if s.strip()]

    def read(self, name):
        return json.loads(self.path(name).read_text())

    def content(self, question, assets):
        content = [{'type': 'text', 'text': question}]
        for asset in assets:
            path = self.path(asset['path'])
            if path.suffix.lower() != '.png':
                raise ValueError('Expected a packaged PNG image')
            if asset.get('choice') is not None:
                content.append({'type': 'text', 'text': f"Option {asset['choice']}:"})
            encoded = base64.b64encode(path.read_bytes()).decode('ascii')
            content.append({'type': 'image_url', 'image_url':
                            {'url': 'data:image/png;base64,' + encoded, 'detail': 'auto'}})
        return content
