"""Exact token-tail repetition detection for Qwen3 thinking, CPU only.

Semantics follow vLLM's check_sequence_repetition / _has_repeating_pattern:
https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/utils.py
Inspected 2026-11-05. Only the thinking phase is observed; never stop an answer.
"""

KEY = 'thinking_loop_breaker'
ADATREE_CONFIG = dict(protocol='qwen3-thinking-loop-break-32-128-4-v1',
                      min_pattern_size=32, max_pattern_size=128, min_count=4)


def repetition_pattern_size(token_ids, min_pattern_size=32, max_pattern_size=128, min_count=4):
    """Return the first matching tail period, or None (vLLM detection semantics)."""
    minimum = max(1, min_pattern_size)
    if max_pattern_size <= 0 or min_count < 2 or minimum > max_pattern_size:
        return None
    for size in range(minimum, min(max_pattern_size, len(token_ids) // min_count) + 1):
        # Each offset must match in all adjacent copies of the trailing block.
        if all(token_ids[-offset] == token_ids[-offset - copy * size]
               for offset in range(1, size + 1) for copy in range(1, min_count)):
            return size
    return None


class ThinkingLoop:
    def __init__(self):
        self.config = dict(ADATREE_CONFIG)
        self.tail = []
        self.pattern_size = self.token_index = None

    @property
    def broken(self):
        return self.pattern_size is not None

    def observe(self, token, thinking_count):
        if self.broken:
            return
        self.tail.append(token)
        # Keep only the largest candidate period repeated the requested count.
        if len(self.tail) > self.config['max_pattern_size'] * self.config['min_count']:
            del self.tail[0]
        self.pattern_size = repetition_pattern_size(self.tail, **{
            k: self.config[k] for k in ('min_pattern_size', 'max_pattern_size', 'min_count')})
        if self.broken:
            self.token_index = thinking_count

    def metrics(self):
        return {KEY: dict(self.config), 'thinking_broken': self.broken,
                'thinking_break_pattern_size': self.pattern_size,
                'thinking_break_token_index': self.token_index,
                'stop_reason': 'thinking_broken' if self.broken else None}


def validate_metrics(value):
    """Check recorded intervention provenance without replaying accepted outputs."""
    broken = value.get('thinking_closure') == 'repetition'
    if KEY not in value:
        if broken or value.get('thinking_broken') or value.get('stop_reason') == 'thinking_broken':
            raise ValueError('Missing thinking loop breaker provenance')
        return
    config = value[KEY]
    if config != ADATREE_CONFIG:
        raise ValueError('Invalid thinking loop breaker configuration')
    size, index = value.get('thinking_break_pattern_size'), value.get('thinking_break_token_index')
    if (type(value.get('thinking_broken')) is not bool
            or value['thinking_broken'] != broken
            or value.get('stop_reason') != ('thinking_broken' if broken else None)
            or (broken and (type(size) is not int or not config['min_pattern_size'] <= size <= config['max_pattern_size']
                            or type(index) is not int or index != value.get('thinking_tokens')
                            or index < size * config['min_count']))
            or (not broken and (size is not None or index is not None))):
        raise ValueError('Inconsistent thinking loop breaker outcome')
