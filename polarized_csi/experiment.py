"""Framework-independent experiment provenance helpers."""
import hashlib
import json
from pathlib import Path


def describe_file(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def describe_pairs(pairs):
    return [{kind: describe_file(path)
             for kind, path in zip(('measurement', 'label'), pair)} for pair in pairs]


def save_record(output, record):
    path = Path(str(output) + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')
    return path
