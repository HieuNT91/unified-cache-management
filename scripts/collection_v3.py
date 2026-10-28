"""Nine-action collection identity, deterministic traversal and answer seeds."""
from runner.output_budget import PROTOCOL as GENERATION_PROTOCOL

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
                 and selected == sorted(set(selected)))
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
    from scripts.ruler import TASKS
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


def validate_declaration(settings):
    expected = declaration(settings['dataset'], settings.get('answers_per_question'),
                           settings.get('execution_profile') == 'ruler-thinking')
    if any(settings.get(k) != v for k, v in expected.items()):
        raise ValueError('Collection v3 declaration changed')
    if settings.get('naive_reuse') or 'extension' in settings or settings.get('tp') != 2:
        raise ValueError('Collection v3 requires a fresh TP2 run')
    import re
    groups = settings.get('groups', [])
    count = 2 if settings['dataset'] == 'longbench-v2' else 4 if settings.get('hardware_profile') == 'l40-tp2' else 5
    flat = [gpu for group in groups for gpu in group]
    if (len(groups) != count or any(len(group) != 2 for group in groups) or len(set(flat)) != count*2
            or any(not isinstance(gpu, str) or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', gpu) for gpu in flat)):
        raise ValueError('Invalid frozen TP2 shard assignment')
    return expected


class Snapshot:
    """Pin committed receipts once; collection may continue publishing new ones."""
    def __init__(self, root, include_probes=True):
        from pathlib import Path
        from runner.setups import file_hash
        from scripts.longbench_a800_data import load, stage, roles, device_role
        self.root = Path(root).resolve()
        self.settings, self.plan, self.rows = load(self.root)
        validate_declaration(self.settings)
        self.protocols = {role: stage(self.root, role)[0] for role in roles(self.settings)}
        self.pins = {name: file_hash(self.root/name) for name in ('settings.json', 'plan.json', 'rows.json')}
        self.receipts = {}
        for role, protocol in self.protocols.items():
            self.pins[f'{role}/protocol.json'] = file_hash(self.root/role/'protocol.json')
            name = f'{device_role(self.settings, role)}/devices.json'
            self.pins[name] = file_hash(self.root/name)
            if role == 'features' and not include_probes:
                continue
            path = self.root/role
            for case in protocol['scheduled_actions']:
                pattern = '*/validated.json' if case == 'probe' else '*/seed-*/validated.json'
                # Atomic validated.json publication is the commit boundary. This
                # list is fixed before reading records; new receipts are ignored.
                receipts = list((path/'records'/case).glob(pattern))
                for receipt in receipts:
                    sequence = receipt.parent.name if case == 'probe' else receipt.parent.parent.name
                    seed = None if case == 'probe' else int(receipt.parent.name.removeprefix('seed-'))
                    key = (role, case, sequence, seed)
                    if key in self.receipts:
                        raise ValueError('Duplicate committed answer identity')
                    self.receipts[key] = receipt
                    self.pins[str(receipt.relative_to(self.root))] = file_hash(receipt)
        row_ids = {r['id'] for r in self.rows}
        for role, case, sequence, seed in self.receipts:
            if sequence not in row_ids or seed not in seeds(self.protocols[role], case):
                raise ValueError('Unexpected committed answer identity')

    def record(self, role, case, row, seed=None):
        from runner.corpus_records import accepted, record_dir
        from scripts.longbench_a800_data import read
        if (role, case, row['id'], seed) not in self.receipts:
            return None
        result = accepted(self.root/role, case, row, self.protocols[role], seed=seed)
        if result is None:
            raise ValueError('Snapshot receipt disappeared')
        folder = record_dir(self.root/role, case, row['id'], seed)
        receipt = read(folder/'validated.json')
        self.pins[str((folder/'result.json').relative_to(self.root))] = receipt['files']['result.json']
        self.pins[str((self.root/role/result['initialization']).relative_to(self.root))] = result['initialization_sha256']
        return result

    def unchanged(self):
        from runner.setups import file_hash
        if any(file_hash(self.root/name) != sha for name, sha in self.pins.items()):
            raise ValueError('Snapshot provenance changed during read')


def answer_metrics(record):
    """Keep export/report memory bounded; full outputs remain in committed records."""
    names = ('sequence_id', 'seed', 'prompt_id', 'subtask', 'scoring', 'accuracy', 'timings',
             'thinking_tokens', 'answer_tokens', 'control_tokens', 'output_tokens', 'forced_tokens',
             'generated_answer_tokens', 'generated_thinking_open_tokens', 'thinking_closure',
             'thinking_cap_reached', 'answer_cap_reached', 'output_cap_reached', 'unfinished_thinking',
             'token_sha256', 'evaluation_protocol', 'gpu_uuids', 'input_sha256')
    result = {k: record[k] for k in names if k in record}
    # metrics() needs only emptiness for the RULER null-prediction counter.
    from runner.reporting import ruler_preprocess
    result['null_prediction'] = not ruler_preprocess(record.get('scored_text', record.get('prediction', '')))
    return result
