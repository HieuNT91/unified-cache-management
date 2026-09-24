"""Load the private port's CPU oracle without importing installed UCM."""
import sys
from pathlib import Path


def expansion_reference(scores, prefix, parameters):
    import torch
    directory = str(Path(__file__).resolve().parent / 'method')
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from prophetkv.selection import ExpansionConfig, select_expansion_reference
    selected, details = select_expansion_reference(
        torch.tensor(scores, dtype=torch.float64), prefix,
        ExpansionConfig(**parameters), diagnostics=True)
    return selected.tolist(), details
