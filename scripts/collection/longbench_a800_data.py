"""Immutable run metadata helpers."""
import json
from pathlib import Path
from runner.cache.setups import atomic_json

def read(path):
    return json.loads(Path(path).read_text())

def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen metadata changed: {path}; use a new experiment directory')
    else:
        atomic_json(path, value)
