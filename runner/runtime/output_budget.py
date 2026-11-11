"""Only generation policy: 8192 thinking / 10000 total output tokens."""
import math
from runner.inputs.prepared_policy import KEY, RULER_PROTOCOL, LONGBENCH_PROTOCOL

PROTOCOL = 'qwen3-thinking-8192-total10000-v1'
THINKING_CAP = 8192
OUTPUT_CAP = 10000
SAMPLING = dict(temperature=.6, top_p=.95, top_k=20, min_p=0.)


def validate_policy(p):
    thinking_cap, output_cap = THINKING_CAP, OUTPUT_CAP
    if (p.get('protocol') != PROTOCOL or p.get('thinking_cap') != thinking_cap
            or type(p.get('thinking_cap')) is not int or p.get('output_reserve') != output_cap
            or type(p.get('output_reserve')) is not int or 'answer_cap' in p
            or type(p.get('opening_in_prompt')) is not bool
            or p.get('sampling') != SAMPLING
            or p.get('seed_source') != 'answer-record-seed'):
        raise ValueError('Invalid total output budget protocol')
    for key in ('eos_ids', 'control_ids', 'transition_ids', 'prefix_ids'):
        if (not isinstance(p.get(key), list) or (key != 'prefix_ids' and not p[key])
                or any(type(t) is not int or t < 0 for t in p[key])):
            raise ValueError('Invalid total output policy tokens')
    controls = set(p['control_ids'])
    if (any(type(p.get(k)) is not int or p[k] < 0 for k in ('open_id', 'close_id'))
            or p['open_id'] == p['close_id']
            or not {p['open_id'], p['close_id'], *p['eos_ids']} <= controls
            or {p['open_id'], p['close_id']} & set(p['eos_ids'])
            or controls & set(p['transition_ids'] + p['prefix_ids'])
            or thinking_cap + 2 + len(p['transition_ids']) + len(p['prefix_ids']) >= output_cap):
        raise ValueError('Invalid total output transition/reserve')
    return p


def validate_sample_policy(sample):
    p = validate_policy(sample[KEY])
    ruler = sample.get('evaluation_protocol') == RULER_PROTOCOL
    if (sample.get('evaluation_protocol') not in (RULER_PROTOCOL, LONGBENCH_PROTOCOL)
            or sample.get('thinking') is not True or sample.get('max_output_tokens') != p['output_reserve']
            or len(sample['token_ids']) + p['output_reserve'] > (82304 if ruler else 131072)
            or (ruler and (sample.get('execution_profile') != 'ruler-thinking' or not p['prefix_ids']))
            or (not ruler and p['prefix_ids'])):
        raise ValueError('Invalid total output sample policy')
    return p


class BudgetState:
    """Request-owned AdaTree thinking, forced transition and answer decoding."""
    def __init__(self, policy):
        from copy import deepcopy
        self.policy = deepcopy(validate_policy(policy))
        self.thinking_cap, self.output_cap = THINKING_CAP, OUTPUT_CAP
        from runner.runtime.thinking_loop import ThinkingLoop
        self.loop = ThinkingLoop()
        self.seen = self.thinking = self.answer = self.forced = self.opening_tokens = 0
        self.phase = 'thinking'
        self.closure = self.first_answer_index = None
        self.queue = []
        self.answer_ids = []
        self.answer_eos = False

    def forced_next(self):
        if self.phase == 'thinking' and (self.thinking == self.thinking_cap or self.loop.broken):
            return self.policy['close_id']
        return self.queue[0] if self.queue else None

    def consume(self, tokens):
        if len(tokens) < self.seen:
            raise ValueError('Thinking request output rewound')
        p = self.policy
        for i in range(self.seen, len(tokens)):
            token = tokens[i]
            forced = self.forced_next()
            if self.phase == 'done':
                raise ValueError('Token after answer termination')
            if forced is not None:
                if token != forced:
                    raise ValueError('Thinking forced token mismatch')
                self.forced += 1
            if self.phase == 'thinking':
                if token == p['close_id']:
                    self.closure = 'repetition' if self.loop.broken else 'forced' if forced is not None else 'natural'
                    self.queue = list(p['transition_ids'] + p['prefix_ids'])
                    self.phase = 'transition' if self.queue else 'answer'
                elif token == p['open_id'] and i == 0 and not p['opening_in_prompt']:
                    self.opening_tokens = 1
                elif token in p['control_ids']:
                    raise ValueError('Premature EOS or control token in thinking')
                else:
                    self.thinking += 1
                    self.loop.observe(token, self.thinking)
            elif self.phase == 'transition':
                self.queue.pop(0)
                if not self.queue:
                    self.phase = 'answer'
            elif token in p['eos_ids']:
                self.answer_eos = True
                self.phase = 'done'
            elif token in p['control_ids']:
                raise ValueError('Unexpected control token in answer')
            else:
                if self.first_answer_index is None:
                    self.first_answer_index = i
                self.answer_ids.append(token)
                self.answer += 1
            self.seen += 1
            if self.seen == self.output_cap:
                self.phase = 'done'
        return self

    def metrics(self, tokenizer):
        return dict(generated_answer_tokens=self.answer, forced_tokens=self.forced,
                    generated_thinking_open_tokens=self.opening_tokens,
                    thinking_cap_reached=self.thinking == self.thinking_cap,
                    answer_cap_reached=self.seen == self.output_cap and not self.answer_eos,
                    thinking_closure=self.closure, **self.loop.metrics(),
                    scored_text=tokenizer.decode(self.answer_ids, skip_special_tokens=False))


def validate_outcome(value, policy):
    validate_policy(policy)
    thinking_cap, output_cap = THINKING_CAP, OUTPUT_CAP
    from runner.runtime.thinking_loop import validate_metrics
    validate_metrics(value)
    for key in ('generated_answer_tokens', 'generated_thinking_open_tokens', 'forced_tokens',
                'thinking_tokens', 'answer_tokens', 'control_tokens', 'output_tokens'):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError('Invalid thinking token accounting')
    n, opening, closure = value['generated_answer_tokens'], value['generated_thinking_open_tokens'], value.get('thinking_closure')
    transition = len(policy['transition_ids']) + len(policy['prefix_ids'])
    latency = value.get('first_answer_content_seconds')
    capped = value.get('answer_cap_reached')
    if (closure not in ('natural', 'forced', 'repetition') or value['thinking_tokens'] > thinking_cap
            or value['forced_tokens'] != transition + (closure in ('forced', 'repetition'))
            or value['answer_tokens'] != n + transition
            or value['thinking_tokens'] + value['answer_tokens'] + value['control_tokens'] != value['output_tokens']
            or value['output_tokens'] > output_cap or opening not in (0, 1)
            or (policy['opening_in_prompt'] and opening != 0)
            or type(capped) is not bool or (capped and value['output_tokens'] != output_cap)
            or value['control_tokens'] != 1 + opening + (not capped)
            or type(value.get('thinking_cap_reached')) is not bool
            or value['thinking_cap_reached'] != (value['thinking_tokens'] == thinking_cap)
            or (closure != 'repetition' and (closure == 'forced') != (value['thinking_tokens'] == thinking_cap))
            or value.get('output_cap_reached') != capped
            or (n == 0 and latency is not None)
            or (n > 0 and (type(latency) not in (int, float) or not math.isfinite(latency) or latency < value['ttft_seconds']))):
        raise ValueError('Inconsistent total output budgets/timing')
