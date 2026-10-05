"""Request-owned, two-phase RULER decoding; no CUDA imports or global state."""
KEY = 'ruler_thinking_budget'
PROFILE = 'ruler-thinking'
PROTOCOL = 'nvidia-ruler-c3f5e3b-qwen3-thinking-v1'
THINKING_CAP = 16384
WINDOW = 82304
SAMPLING = dict(temperature=.6, top_p=.95, top_k=20, min_p=0., seed=0)


def make_policy(tokenizer, prefix, answer_cap, opening_in_prompt=False):
    def special(text):
        token = tokenizer.convert_tokens_to_ids(text)
        if type(token) is not int or tokenizer.convert_ids_to_tokens(token) != text:
            raise ValueError('Expected native Qwen3 thinking delimiters')
        return token
    open_id, close_id = special('<think>'), special('</think>')
    policy = dict(protocol=PROTOCOL, thinking_cap=THINKING_CAP, answer_cap=answer_cap,
                  open_id=open_id, close_id=close_id, opening_in_prompt=opening_in_prompt,
                  eos_ids=[tokenizer.eos_token_id], control_ids=sorted(set(tokenizer.all_special_ids) | {open_id,close_id}),
                  transition_ids=tokenizer.encode('\n\n', add_special_tokens=False),
                  prefix_ids=tokenizer.encode(prefix, add_special_tokens=False), sampling=SAMPLING)
    policy['output_reserve'] = THINKING_CAP + 1 + (not opening_in_prompt) + len(policy['transition_ids']) + len(policy['prefix_ids']) + answer_cap
    validate_policy(policy)
    return policy


def validate_policy(p):
    from scripts.ruler import CAPS
    if not isinstance(p,dict):
        raise ValueError('Missing thinking budget policy')
    if (p.get('protocol') != PROTOCOL or p.get('thinking_cap') != THINKING_CAP
            or type(p.get('thinking_cap')) is not int or type(p.get('answer_cap')) is not int
            or type(p.get('output_reserve')) is not int or type(p.get('opening_in_prompt')) is not bool or p.get('answer_cap') not in CAPS.values() or p.get('sampling') != SAMPLING):
        raise ValueError('Invalid thinking budget protocol')
    for key in ('eos_ids', 'control_ids', 'transition_ids', 'prefix_ids'):
        if not isinstance(p.get(key), list) or not p[key] or any(type(t) is not int or t < 0 for t in p[key]):
            raise ValueError('Invalid thinking policy tokens')
    controls = set(p['control_ids'])
    if (type(p.get('open_id')) is not int or type(p.get('close_id')) is not int
            or p['open_id'] == p['close_id'] or not {p['open_id'], p['close_id'], *p['eos_ids']} <= controls
            or {p['open_id'], p['close_id']} & set(p['eos_ids'])
            or controls & set(p['transition_ids'] + p['prefix_ids'])
            or p.get('output_reserve') != THINKING_CAP + 1 + (not p['opening_in_prompt']) + len(p['transition_ids']) + len(p['prefix_ids']) + p['answer_cap']):
        raise ValueError('Invalid thinking transition/reserve')
    return p


def validate_sample_policy(sample):
    from scripts.ruler import CAPS
    p = validate_policy(sample.get(KEY))
    if (sample.get('evaluation_protocol') != PROTOCOL or sample.get('execution_profile') != PROFILE or sample['thinking'] is not True
            or p['answer_cap'] != CAPS.get(sample.get('task'))
            or sample['max_output_tokens'] != p['output_reserve']
            or len(sample['token_ids']) + p['output_reserve'] > WINDOW):
        raise ValueError('Invalid thinking RULER sample or output reserve; retain raw and halt')
    return p


class BudgetState:
    """Incrementally consumes emitted IDs. Attached to the request, freed with it."""
    def __init__(self, policy):
        from copy import deepcopy
        self.policy = deepcopy(validate_policy(policy))
        self.seen = self.thinking = self.answer = self.forced = 0
        self.phase = 'thinking'
        self.closure = None
        self.queue = []
        self.answer_ids = []
        self.first_answer_index = None
        self.answer_eos = False
        self.opening_tokens = 0

    def forced_next(self):
        if self.phase == 'thinking' and self.thinking == self.policy['thinking_cap']:
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
                    self.phase = 'transition'
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
                if self.answer == p['answer_cap']:
                    self.phase = 'done'
            self.seen += 1
        return self

    def metrics(self, tokenizer):
        return dict(generated_answer_tokens=self.answer, forced_tokens=self.forced,
                    generated_thinking_open_tokens=self.opening_tokens,
                    thinking_cap_reached=self.thinking == self.policy['thinking_cap'],
                    answer_cap_reached=self.answer == self.policy['answer_cap'],
                    thinking_closure=self.closure, scored_text=tokenizer.decode(self.answer_ids, skip_special_tokens=False))


def request_state(request):
    params = request.sampling_params
    policy = (getattr(params, 'extra_args', None) or {}).get(KEY)
    if policy is None:
        return None
    state = getattr(request, '_ruler_thinking_state', None)
    if state is None:
        state = request._ruler_thinking_state = BudgetState(policy)
    elif state.policy != policy:
        raise ValueError('Thinking policy changed during request')
    return state.consume(request.output_token_ids)


def enforce_logits(model_runner, logits):
    """Runs for baseline and sparse paths immediately before the native sampler."""
    for rid, index in model_runner.input_batch.req_id_to_index.items():
        state = request_state(model_runner.requests[rid])
        if state is None:
            continue
        if model_runner.speculative_config:
            raise ValueError('Thinking budget requires ordinary autoregressive decoding')
        forced = state.forced_next()
        if forced is not None:
            logits[index].fill_(float('-inf'))
            logits[index, forced] = 0.
        else:
            if state.phase not in ('thinking', 'answer'):
                raise ValueError('Sampling a finished thinking request')
            allowed = {state.policy['close_id']} if state.phase == 'thinking' else set(state.policy['eos_ids'])
            if state.phase == 'thinking' and state.seen == 0 and not state.policy['opening_in_prompt']:
                allowed.add(state.policy['open_id'])
            logits[index, sorted(set(state.policy['control_ids']) - allowed)] = float('-inf')


def stopping_hook(original, statuses):
    def check_stop(request, max_model_len, pooler_output=None):
        state = request_state(request)
        if state is None:
            return original(request, max_model_len, pooler_output)
        if state.phase == 'done':
            request.status = statuses.FINISHED_STOPPED if state.answer_eos else statuses.FINISHED_LENGTH_CAPPED
            return True
        if request.num_tokens >= max_model_len or request.num_output_tokens >= request.max_tokens:
            raise ValueError('Thinking request exhausted reserved capacity before answer completion')
        return False
    return check_stop


def install_scheduler_hook():
    from vllm.v1.core.sched import scheduler, utils
    from vllm.v1.request import RequestStatus
    if not getattr(utils.check_stop, '_ruler_budget_hook', False):
        hook = stopping_hook(utils.check_stop, RequestStatus)
        hook._ruler_budget_hook = True
        utils.check_stop = scheduler.check_stop = hook


def validate_thinking_outcome(value, policy):
    """Validate committed and portable phase counts without attention replay."""
    import math
    for key in ('generated_answer_tokens', 'generated_thinking_open_tokens', 'forced_tokens', 'thinking_tokens', 'answer_tokens', 'control_tokens', 'output_tokens'):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError('Invalid thinking token accounting')
    n = value['generated_answer_tokens']
    closure = value.get('thinking_closure')
    forced = len(policy['transition_ids']) + len(policy['prefix_ids']) + (closure == 'forced')
    latency = value.get('first_answer_content_seconds')
    opening = value['generated_thinking_open_tokens']
    if (closure not in ('natural', 'forced') or value['thinking_tokens'] > policy['thinking_cap']
            or n > policy['answer_cap'] or value['forced_tokens'] != forced
            or value['answer_tokens'] != n + forced - (closure == 'forced')
            or value['thinking_tokens'] + value['answer_tokens'] + value['control_tokens'] != value['output_tokens']
            or opening not in (0, 1) or (policy['opening_in_prompt'] and opening != 0)
            or value['control_tokens'] != 1 + opening + (n < policy['answer_cap'])
            or type(value.get('thinking_cap_reached')) is not bool
            or value['thinking_cap_reached'] != (value['thinking_tokens'] == policy['thinking_cap'])
            or (closure == 'forced') != value['thinking_cap_reached']
            or type(value.get('answer_cap_reached')) is not bool
            or value['answer_cap_reached'] != (n == policy['answer_cap'])
            or value['output_cap_reached'] != (value['thinking_cap_reached'] or value['answer_cap_reached'])
            or (n == 0 and latency is not None)
            or (n > 0 and (type(latency) not in (int,float) or not math.isfinite(latency) or latency < value['ttft_seconds']))):
        raise ValueError('Inconsistent thinking outcome budgets/timing')

