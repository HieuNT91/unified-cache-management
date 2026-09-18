"""Materialize repository-packaged RULER assets without network access."""
import json
import os
from pathlib import Path
import zipfile


def valid_hotpot(data):
    return (isinstance(data, list) and bool(data) and
            all(isinstance(row, dict) and {'_id', 'question', 'answer', 'context'} <= row.keys()
                for row in data))


def materialize_hotpot(directory):
    """Atomically extract and validate hotpotqa.json.zip when JSON is absent."""
    directory = Path(directory)
    destination = directory / 'hotpotqa.json'
    if destination.is_file():
        if not valid_hotpot(json.loads(destination.read_text())):
            raise ValueError(f'Invalid HotpotQA data: {destination}')
        return destination
    archive = directory / 'hotpotqa.json.zip'
    if not archive.is_file():
        return None
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) != 1 or members[0].filename != 'hotpotqa.json' or members[0].is_dir():
            raise ValueError('hotpotqa.json.zip must contain exactly one top-level hotpotqa.json')
        raw = bundle.read(members[0])
    data = json.loads(raw)
    if not valid_hotpot(data):
        raise ValueError('Packaged HotpotQA JSON has an unexpected structure')
    temporary = destination.with_name(f'{destination.name}.{os.getpid()}.tmp')
    try:
        temporary.write_bytes(raw)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
