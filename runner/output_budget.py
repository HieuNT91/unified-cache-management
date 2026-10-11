"""Versioned 15K thinking / 16K total output policy for collection v3."""
import math

PROTOCOL = 'qwen3-thinking-15000-total16384-v2'
RULER_PROTOCOL = 'nvidia-ruler-c3f5e3b-qwen3-thinking-v2'
LONGBENCH_PROTOCOL = 'longbench-v2-qwen3-thinking-v2'
THINKING_CAP = 15000
OUTPUT_CAP = 16384


def make_policy(tokenizer, prefix='', opening_in_prompt=False):
    from runner.thinking_budget import SAMPLING
    p = dict(protocol=PROTOCOL, thinking_cap=THINKING_CAP, output_reserve=OUTPUT_CAP,
             opening_in_prompt=opening_in_prompt, sampling={k: v for k, v in SAMPLING.items() if k != 'seed'},
             seed_source='answer-record-seed',
             open_id=tokenizer.convert_tokens_to_ids('<think>'),
             close_id=tokenizer.convert_tokens_to_ids('</think>'),
             eos_ids=[tokenizer.eos_token_id],
             transition_ids=tokenizer.encode('\n\n', add_special_tokens=False),
             prefix_ids=tokenizer.encode(prefix, add_special_tokens=False))
    for key, text in (('open_id', '<think>'), ('close_id', '</think>')):
        if tokenizer.convert_ids_to_tokens(p[key]) != text:
            raise ValueError('Expected native Qwen3 thinking delimiters')
    p['control_ids'] = sorted(set(tokenizer.all_special_ids) | {p['open_id'], p['close_id']})
    return validate_policy(p)


def validate_policy(p):
    from runner.thinking_budget import SAMPLING
    if (p.get('protocol') != PROTOCOL or p.get('thinking_cap') != THINKING_CAP
            or type(p.get('thinking_cap')) is not int or p.get('output_reserve') != OUTPUT_CAP
            or type(p.get('output_reserve')) is not int or 'answer_cap' in p
            or type(p.get('opening_in_prompt')) is not bool
            or p.get('sampling') != {k: v for k, v in SAMPLING.items() if k != 'seed'}
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
            or THINKING_CAP + 2 + len(p['transition_ids']) + len(p['prefix_ids']) >= OUTPUT_CAP):
        raise ValueError('Invalid total output transition/reserve')
    return p


def validate_sample_policy(sample):
    from runner.thinking_budget import KEY
    p = validate_policy(sample[KEY])
    ruler = sample.get('evaluation_protocol') == RULER_PROTOCOL
    if (sample.get('evaluation_protocol') not in (RULER_PROTOCOL, LONGBENCH_PROTOCOL)
            or sample.get('thinking') is not True or sample.get('max_output_tokens') != OUTPUT_CAP
            or len(sample['token_ids']) + OUTPUT_CAP > (82304 if ruler else 131072)
            or (ruler and (sample.get('execution_profile') != 'ruler-thinking' or not p['prefix_ids']))
            or (not ruler and p['prefix_ids'])):
        raise ValueError('Invalid total output sample policy')
    return p


class TotalBudgetState:
    """Same logits/scheduler interface as the legacy separately capped decoder."""
    def __init__(self, policy):
        from copy import deepcopy
        self.policy = deepcopy(validate_policy(policy))
        self.seen = self.thinking = self.answer = self.forced = self.opening_tokens = 0
        self.phase = 'thinking'
        self.closure = self.first_answer_index = None
        self.queue = []
        self.answer_ids = []
        self.answer_eos = False

    def forced_next(self):
        if self.phase == 'thinking' and self.thinking == THINKING_CAP:
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
                    self.closure = 'forced' if forced is not None else 'natural'
                    self.queue = list(p['transition_ids'] + p['prefix_ids'])
                    self.phase = 'transition' if self.queue else 'answer'
                elif token == p['open_id'] and i == 0 and not p['opening_in_prompt']:
                    self.opening_tokens = 1
                elif token in p['control_ids']:
                    raise ValueError('Premature EOS or control token in thinking')
                else:
                    self.thinking += 1
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
            if self.seen == OUTPUT_CAP:
                self.phase = 'done'
        return self

    def metrics(self, tokenizer):
        return dict(generated_answer_tokens=self.answer, forced_tokens=self.forced,
                    generated_thinking_open_tokens=self.opening_tokens,
                    thinking_cap_reached=self.closure == 'forced',
                    answer_cap_reached=self.seen == OUTPUT_CAP and not self.answer_eos,
                    thinking_closure=self.closure,
                    scored_text=tokenizer.decode(self.answer_ids, skip_special_tokens=False))


def validate_outcome(value, policy):
    validate_policy(policy)
    for key in ('generated_answer_tokens', 'generated_thinking_open_tokens', 'forced_tokens',
                'thinking_tokens', 'answer_tokens', 'control_tokens', 'output_tokens'):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError('Invalid thinking token accounting')
    n, opening, closure = value['generated_answer_tokens'], value['generated_thinking_open_tokens'], value.get('thinking_closure')
    transition = len(policy['transition_ids']) + len(policy['prefix_ids'])
    latency = value.get('first_answer_content_seconds')
    capped = value.get('answer_cap_reached')
    if (closure not in ('natural', 'forced') or value['thinking_tokens'] > THINKING_CAP
            or value['forced_tokens'] != transition + (closure == 'forced')
            or value['answer_tokens'] != n + transition
            or value['thinking_tokens'] + value['answer_tokens'] + value['control_tokens'] != value['output_tokens']
            or value['output_tokens'] > OUTPUT_CAP or opening not in (0, 1)
            or (policy['opening_in_prompt'] and opening != 0)
            or type(capped) is not bool or (capped and value['output_tokens'] != OUTPUT_CAP)
            or value['control_tokens'] != 1 + opening + (not capped)
            or type(value.get('thinking_cap_reached')) is not bool
            or value['thinking_cap_reached'] != (closure == 'forced')
            or (closure == 'forced') != (value['thinking_tokens'] == THINKING_CAP)
            or value.get('output_cap_reached') != capped
            or (n == 0 and latency is not None)
            or (n > 0 and (type(latency) not in (int, float) or not math.isfinite(latency) or latency < value['ttft_seconds']))):
        raise ValueError('Inconsistent total output budgets/timing')
