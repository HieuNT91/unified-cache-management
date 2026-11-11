"""Original prepared-data metadata; these budgets are never used for generation."""

PROTOCOL = 'qwen3-thinking-15000-total16384-v2'
RULER_PROTOCOL = 'nvidia-ruler-c3f5e3b-qwen3-thinking-v2'
LONGBENCH_PROTOCOL = 'longbench-v2-qwen3-thinking-v2'
THINKING_CAP = 15000
OUTPUT_CAP = 16384
KEY = 'ruler_thinking_budget'
WINDOW = 82304
SAMPLING = dict(temperature=.6, top_p=.95, top_k=20, min_p=0., seed=0)


def make_policy(tokenizer, prefix='', opening_in_prompt=False):
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
    thinking_cap, output_cap = THINKING_CAP, OUTPUT_CAP
    if (p.get('protocol') != PROTOCOL or p.get('thinking_cap') != thinking_cap
            or type(p.get('thinking_cap')) is not int or p.get('output_reserve') != output_cap
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

