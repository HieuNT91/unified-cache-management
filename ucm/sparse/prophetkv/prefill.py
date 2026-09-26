"""Original-position prefill progress, independent of vLLM and CUDA."""
from dataclasses import dataclass

PREFILL_BUDGET = 16384


@dataclass
class PrefillProgress:
    prefix: int
    suffix: int
    length: int
    cursor: int = None
    selected: tuple = None

    def __post_init__(self):
        self.cursor = self.prefix

    def select_once(self, selected):
        selected = tuple(selected)
        if self.selected is not None:
            raise RuntimeError('Global selection may only run once')
        if selected != tuple(sorted(set(selected))) or any(
                p < self.prefix or p >= self.suffix for p in selected):
            raise RuntimeError('Invalid global selection')
        self.selected = selected

    def step(self, start, count):
        if self.selected is None or start != self.cursor:
            raise RuntimeError('Missing selection or noncontiguous prefill progress')
        end = start + count
        if not 0 < count <= PREFILL_BUDGET or end > self.length:
            raise RuntimeError('Invalid scheduled prefill range')
        positions = tuple(p for p in self.selected if start <= p < end)
        positions += tuple(range(max(start, self.suffix), end))
        self.cursor = end
        return positions


def prefill_step(start, count, prompt_length):
    """One token is still prefill when its original position is in the prompt."""
    if start >= prompt_length:
        return False, False
    if not 0 < count <= PREFILL_BUDGET or start + count > prompt_length:
        raise RuntimeError('Invalid prefill scheduling budget')
    return True, start + count == prompt_length
