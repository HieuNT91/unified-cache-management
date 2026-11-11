"""Nine-action collection identity, deterministic traversal and answer seeds."""
from runner.inputs.prepared_policy import PROTOCOL as GENERATION_PROTOCOL

VERSION = 'tp2-data-v3'
RATIOS = (1, 5, 10, 20, 30, 40, 60, 80)
ACTIONS = [dict(id='nocache', method='baseline', ratio=None)] + [
    dict(id=f'prophetkv-{p}', method='prophetkv', ratio=p/100) for p in RATIOS]
PHASES = ('baseline', 'features', 'cached')


def enabled(value):
    return value.get('collection_protocol') == VERSION


def seeds(value, case=None):
    if case == 'probe' or not enabled(value):
        return [None]
    n = value.get('answers_per_question')
    selected = value.get('answer_seeds')
    tree_longbench = (value.get('schema') == 'tree-prophetkv-thinking-collection-v1'
                      and value.get('dataset') == 'longbench-v2')
    if type(n) is not int or n <= 0:
        raise ValueError('Invalid frozen answer seed schedule')
    if tree_longbench:
        valid = (isinstance(selected, list) and len(selected) == n
                 and all(type(seed) is int and seed >= 0 for seed in selected)
                 and len(set(selected)) == len(selected))
    else:
        valid = selected == list(range(n))
    if not valid:
        raise ValueError('Invalid frozen answer seed schedule')
    return selected


def declaration(dataset, answers, thinking=False):
    if dataset not in ('ruler', 'longbench-v2'):
        raise ValueError('Unknown collection dataset')
    if type(answers) is not int or answers <= 0 or (dataset == 'ruler' and answers != 1):
        raise ValueError('ANSWERS_PER_QUESTION must be a positive integer; RULER requires one answer')
    return dict(collection_protocol=VERSION, answers_per_question=answers, answer_seeds=list(range(answers)),
                generation_protocol=GENERATION_PROTOCOL if thinking or dataset == 'longbench-v2' else 'ruler-nonthinking-v1',
                phase_order=list(PHASES), traversal='round-robin-subtask-sequence-method-seed-v1',
                shard_assignment='source-ordinal-modulo-groups-v1')


def ordered_rows(rows, protocol, group):
    rows = [r for r in rows if r['ordinal'] % len(protocol['groups']) == group]
    if not enabled(protocol):
        return rows
    # Fixed task order; round-robin queues retain each task's source ordinal order.
    from collections import defaultdict, deque
    from scripts.data.ruler import TASKS
    queues = defaultdict(deque)
    for row in sorted(rows, key=lambda r: (r['ordinal'], r['id'])):
        queues[row['subtask']].append(row)
    tasks = list(TASKS) if protocol['dataset'] == 'ruler' else sorted(queues)
    result = []
    while any(queues.values()):
        for task in tasks:
            if queues[task]:
                result.append(queues[task].popleft())
    return result






def answer_metrics(record):
    """Keep export/report memory bounded; full outputs remain in committed records."""
    names = ('sequence_id', 'seed', 'prompt_id', 'subtask', 'scoring', 'accuracy', 'timings',
             'thinking_tokens', 'answer_tokens', 'control_tokens', 'output_tokens', 'forced_tokens',
             'generated_answer_tokens', 'generated_thinking_open_tokens', 'thinking_closure',
             'thinking_loop_breaker', 'thinking_broken', 'thinking_break_pattern_size',
             'thinking_break_token_index', 'finish_reason', 'stop_reason',
             'thinking_cap_reached', 'answer_cap_reached', 'output_cap_reached', 'unfinished_thinking',
             'token_sha256', 'evaluation_protocol', 'gpu_uuids', 'input_sha256')
    result = {k: record[k] for k in names if k in record}
    # metrics() needs only emptiness for the RULER null-prediction counter.
    from runner.reporting.aggregate import ruler_preprocess
    result['null_prediction'] = not ruler_preprocess(record.get('scored_text', record.get('prediction', '')))
    return result
