"""Resolve result artifacts without allowing paths outside their run."""
from pathlib import Path


def relative(root,name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('Expected a relative artifact path')
    path=Path(root)/p
    if not path.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Artifact escapes corpus')
    return path
