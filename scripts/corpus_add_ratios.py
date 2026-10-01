#!/usr/bin/env python3
"""Collect only 5/10% into an existing corpus using this checkout's runtime."""
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

EXTRA = ('prophetkv-5', 'prophetkv-10')
STATE = Path('extensions/add5-10')
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
os.environ.update(PYTHONPATH=str(CODE_ROOT), PYTHONDONTWRITEBYTECODE='1',
                  OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def contained(root, name):
    name = Path(name)
    if name.is_absolute() or '..' in name.parts or not name.parts:
        raise ValueError('Expected relative artifact path')
    path = Path(root)/name
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Artifact escapes its root')
    return path


@contextmanager
def locked(path):
    with Path(path).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def check_files(root, pins):
    for name, expected in pins.items():
        path = contained(root, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Pinned file changed or missing: '+str(path))


def completion(root, protocol, count):
    """The scheduled tranche, not the full reserved 2,600-row inventory, is final."""
    name = 'final-validation.json' if count == 2600 else f'partial-{count}-validation.json'
    done = read(root/name)
    expected = {a['id']: count for a in protocol['actions']}
    if (done.get('status') not in ('complete', 'partial-complete') or
            done.get('tranche_complete') is not True or done.get('scheduled_samples') != count or
            done.get('accepted_answers') != expected or done.get('accepted_probes') != count or
            done.get('answers') != count * len(expected)):
        raise ValueError('Original tranche is incomplete')
    cleanup = read(root/'cleanup.json')
    state = read(root/'supervisor.json')
    if (cleanup.get('owned_engines_exited') is not True or cleanup.get('complete') is not True or
            state.get('state') not in ('complete', 'partial-complete')):
        raise ValueError('Original collection has not successfully exited')
    return name


def environment_check(protocol):
    from importlib.metadata import version
    from runner.config import VERSIONS
    from scripts.corpus_inputs import spec
    from run import check_model
    model = Path(protocol['model'])
    check_model(model)
    for package, wanted in VERSIONS.items():
        if version(package).split('+')[0] != wanted:
            raise ValueError('Pinned runtime environment changed: '+package)
    for name, expected in spec()['model_fingerprints'].items():
        if digest(model/name) != expected:
            raise ValueError('Original model/tokenizer changed: '+name)
    index = read(model/'model.safetensors.index.json')
    if any(not (model/name).is_file() for name in set(index['weight_map'].values())):
        raise ValueError('Missing model weight shard')


def selected_rows(base, limit, tasks):
    rows = [r for r in base.rows.values() if r['ordinal'] < limit]
    if (not 1 <= limit <= 200 or Counter(r['subtask'] for r in rows) != Counter({t: limit for t in tasks}) or
            any(sorted(r['ordinal'] for r in rows if r['subtask'] == t) != list(range(limit)) for t in tasks)):
        raise ValueError('Missing or invalid prepared rows for original tranche')
    return rows


def combined_actions(base):
    from runner.tree_policy import actions
    result = [*base, dict(id=EXTRA[0], method='prophetkv', ratio=.05),
              dict(id=EXTRA[1], method='prophetkv', ratio=.10)]
    actions(result)  # Reject duplicate IDs AND duplicate ratios.
    return sorted(result, key=lambda a: a['ratio'] or 0)


def derive(attention, sample, inventory):
    from runner.corpus import selection
    return dict(scores=attention['scores'], selections={a['id']: selection(
        attention['scores'], sample['boundaries'][1], a['ratio']) for a in inventory if a['id'] != 'nocache'})


def prepare(root):
    """Caller owns the base run lock; expensive CPU replay precedes any GPU work."""
    import numpy as np
    from runner.corpus import Corpus, match_answer
    from runner.corpus_records import accepted, record_dir, protocol_identity
    from runner.setups import atomic_json
    from runner.tree_profiles import validate
    from scripts.corpus_inputs import TASKS
    from scripts.corpus_control import groups
    from scripts.router_control import code_hashes
    from run import verify_diagnostics
    state = root/STATE
    if (state/'protocol.json').exists():
        corpus = Extension(root)
        corpus.verify_sources()
        environment_check(corpus.protocol)
        corpus.validate_new()
        return corpus
    base = Corpus(root)
    protocol = base.protocol
    if protocol['kind'] != 'collection' or protocol['dataset'] != 'ruler':
        raise ValueError('Expected an original RULER collection')
    groups(*[','.join(g) for g in protocol['groups']])
    limit = read(root/'tranche.json')['limit_per_task']
    rows = selected_rows(base, limit, TASKS)
    done = completion(root, protocol, len(rows))
    inventory = combined_actions(protocol['actions'])
    for case in EXTRA:
        if (root/'records'/case).exists():
            raise ValueError('Unregistered added-action records already exist: '+case)
    environment_check(protocol)
    pins = {str(root/name): digest(root/name) for name in
            ('protocol.json', 'settings.json', 'tranche.json', 'cleanup.json', 'supervisor.json', done)}
    pins[str(base.prepared/'plan.json')] = protocol['plan_sha256']
    derived = {}
    (state/'derived').mkdir(parents=True, exist_ok=True)
    for n, row in enumerate(rows):
        pid = row['id']
        sample = base.inputs(pid)
        validate(sample, 'ruler')
        for key, hash_key in (('prepared', 'sha256'), ('raw', 'raw_sha256')):
            path = contained(base.prepared, row[key])
            if digest(path) != row[hash_key]:
                raise ValueError('Original input changed: '+pid)
            pins[str(path)] = row[hash_key]
        receipt_path = base.prepared/'generation'/pid/'receipt.json'
        pins[str(receipt_path)] = digest(receipt_path)
        records = {}
        for case in ['probe', *base.actions]:
            record = accepted(root, case, row, protocol)
            if record is None:
                raise ValueError('Missing original record: '+case+'/'+pid)
            records[case] = record
            folder = record_dir(root, case, pid)
            receipt = folder/'validated.json'
            pins[str(receipt)] = digest(receipt)
            for name, expected in read(receipt)['files'].items():
                pins[str(contained(folder, name))] = expected
            pins[str(contained(root, record['initialization']))] = record['initialization_sha256']
        attention = base.attention(pid)  # Original archive inventory, never the enlarged inventory.
        for case in ['probe', *base.actions]:
            ds = read(record_dir(root, case, pid)/'diagnostics.json')
            if case == 'probe':
                verify_diagnostics(ds, sample, 'prophetkv', .01, 4, range(64))
                probe_attention = derive(attention, sample, [dict(id='prophetkv-1', ratio=.01)])
                match_answer(ds, sample, 'prophetkv-1', probe_attention,
                             [dict(id='nocache', method='baseline', ratio=None),
                              dict(id='prophetkv-1', method='prophetkv', ratio=.01)])
            elif case == 'nocache':
                verify_diagnostics(ds, sample, 'baseline', None, 4, range(64))
            else:
                match_answer(ds, sample, case, attention, protocol['actions'])
        arrays = derive(attention, sample, inventory)
        path = state/'derived'/f'{pid}.npz'
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, scores=arrays['scores'], **arrays['selections'])
        temporary.replace(path)
        derived[pid] = dict(archive_sha256=digest(path), base_probe_sha256=digest(record_dir(root, 'probe', pid)/'validated.json'))
        if (n+1) % 10 == 0 or n+1 == len(rows):
            print(f'Validated original inputs, answers and attention: {n+1}/{len(rows)}', flush=True)
    atomic_json(state/'sources.json', dict(files=pins, rows=rows))
    extended = dict(protocol, schema='ruler-inplace-actions-v2', actions=inventory, added_actions=list(EXTRA),
                    base_protocol_sha256=protocol_identity(protocol), sources_sha256=digest(state/'sources.json'),
                    code=code_hashes(), derived=derived)
    atomic_json(state/'protocol.json', extended)
    return Extension(root)


class Extension:
    def __init__(self, root):
        from runner.corpus_records import protocol_identity
        from runner.tree_policy import actions
        self.root = Path(root)
        self.state = self.root/STATE
        self.protocol = read(self.state/'protocol.json')
        self.base_protocol = read(self.root/'protocol.json')
        if protocol_identity(self.base_protocol) != self.protocol['base_protocol_sha256']:
            raise ValueError('Original protocol changed')
        if digest(self.state/'sources.json') != self.protocol['sources_sha256']:
            raise ValueError('Source inventory changed')
        self.sources = read(self.state/'sources.json')
        self.rows = self.sources['rows']
        self.actions = actions(self.protocol['actions'])

    def verify_sources(self):
        from runner import corpus_progress as progress
        for name, expected in progress.track(self.sources['files'].items(), 'Validating original checksums', 'files', lambda item: item[0]):
            if not Path(name).is_file() or digest(name) != expected:
                raise ValueError('Original artifact changed: '+name)
        for row in progress.track(self.rows, 'Validating derived attention', 'samples', lambda row: row['id']):
            self.attention(row)

    def attention(self, row, replay=False):
        import numpy as np
        from runner.corpus import load_attention
        from runner.corpus_records import accepted, record_dir
        pid = row['id']
        path = self.state/'derived'/f'{pid}.npz'
        provenance = self.protocol['derived'][pid]
        if digest(path) != provenance['archive_sha256']:
            raise ValueError('Derived attention changed')
        receipt = record_dir(self.root, 'probe', pid)/'validated.json'
        if digest(receipt) != provenance['base_probe_sha256']:
            raise ValueError('Original probe receipt changed')
        sample_path = contained(self.protocol['prepared'], row['prepared'])
        if digest(sample_path) != row['sha256']:
            raise ValueError('Original input changed')
        with np.load(path, allow_pickle=False) as data:
            attention = dict(scores=data['scores'].copy(), selections={a: data[a].copy() for a in self.actions if a != 'nocache'})
        if replay:
            probe = accepted(self.root, 'probe', row, self.base_protocol)
            if probe is None:
                raise ValueError('Missing original probe')
            original = load_attention(receipt.parent, probe, read(sample_path), self.base_protocol['actions'])
            expected = derive(original, read(sample_path), self.protocol['actions'])
            if (not np.array_equal(expected['scores'], attention['scores']) or
                    any(not np.array_equal(mask, attention['selections'][a]) for a, mask in expected['selections'].items())):
                raise ValueError('Derived attention differs from original native replay')
        return attention

    def outcome(self, row, case, full=False):
        from runner.corpus_records import accepted, record_dir
        from runner.corpus import match_answer
        record = accepted(self.root, case, row, self.protocol)
        if record is not None:
            if (record.get('derived_mask_provenance') != self.protocol['derived'][row['id']] or
                    record.get('cache_deletion', {}).get('deleted_shards', 0) <= 0 or
                    not record['initialization'].startswith(str(STATE/'sessions')+'/')):
                raise ValueError('Missing extension provenance or cache deletion evidence')
            if full:
                attention = self.attention(row)
                ds = read(record_dir(self.root, case, row['id'])/'diagnostics.json')
                match_answer(ds, read(contained(self.protocol['prepared'], row['prepared'])), case,
                             attention, self.protocol['actions'])
        return record

    def pending(self, group):
        return [r for r in self.rows if r['ordinal'] % len(self.protocol['groups']) == group
                and any(self.outcome(r, a) is None for a in EXTRA)]

    def validate_new(self, full=True):
        from runner import corpus_progress as progress
        allowed = {r['id'] for r in self.rows}
        for case in EXTRA:
            if any(p.parent.name not in allowed for p in (self.root/'records'/case).glob('*/validated.json')):
                raise ValueError('Unexpected added-action prompt')
            for row in progress.track(self.rows, 'Validating '+case+' outcomes', 'records', lambda row: row['id']):
                self.outcome(row, case, full=full)


def collect(root, group, attempt):
    from runner.corpus_runtime import Engine
    from runner.corpus_records import record_dir, publish
    from runner.setups import atomic_json, check_environment
    from scripts.router_control import code_hashes
    corpus = Extension(root)
    pending = corpus.pending(group)
    if not pending:
        return
    if check_environment(4) != corpus.protocol['groups'][group]:
        raise ValueError('Original GPU UUID assignment changed')
    with locked(corpus.state/f'group{group}.lock'):
        engine = Engine(corpus.state, corpus.protocol, group, 'cached', attempt, pending)
        try:
            initialization = engine.session/'initialization.json'
            atomic_json(initialization, dict(read(initialization), execution_code=code_hashes()))
            for row in pending:
                attention = corpus.attention(row, replay=True)
                engine.begin(row)
                staged = []
                for case in EXTRA:
                    if corpus.outcome(row, case) is not None:
                        continue
                    folder = record_dir(root, case, row['id'])
                    if folder.exists():
                        history = corpus.state/'incomplete'/f'{case}-{row["id"]}-{uuid.uuid4().hex}'
                        history.parent.mkdir(exist_ok=True)
                        folder.rename(history)
                    record, ds = engine.answer(corpus.actions[case], case, attention)
                    record['initialization'] = str(STATE/record['initialization'])
                    record['derived_mask_provenance'] = corpus.protocol['derived'][row['id']]
                    staged.append((case, record, ds))
                deletion = engine.end()
                if deletion.get('deleted_shards', 0) <= 0:
                    raise ValueError('Cache deletion failed; answers cannot be committed')
                for case, record, ds in staged:
                    record['cache_deletion'] = deletion
                    publish(root, case, row, corpus.protocol, record, ds)
                    corpus.outcome(row, case, full=True)
                    atomic_json(corpus.state/f'progress-group{group}.json',
                                dict(prompt_id=row['id'], case=case, accepted_at=time.time()))
                    print(f'{case} {row["id"]}: accepted', flush=True)
                startup = corpus.state/f'startup-group{group}.json'
                if not startup.exists():
                    atomic_json(startup, dict(prompt_id=row['id'], group=group, complete=True,
                        actions=list(EXTRA), original_attention_replayed=True, all_rank_scores_masks=True,
                        all64_selection=True, retirement_validated=True, cache_deletion=deletion,
                        initialization=str(STATE/engine.session.relative_to(corpus.state)/'initialization.json')))
        finally:
            engine.close()


def summarize(root, same_count=False, final=False):
    from runner.corpus_records import protocol_identity
    from runner.matched_status import committed_records, match_records
    from runner.reporting import aggregate, COLUMNS, render_markdown
    from runner.router_process import alive
    from runner.setups import atomic_json
    if not (Path(root)/STATE/'protocol.json').exists():
        state = Path(root)/STATE/'supervisor.json'
        if not final and state.exists():
            saved = read(state)
            result = dict(status=saved['state'], supervisor=dict(saved, alive=alive(saved)),
                          message='Extension preparation is pending; see extensions/add5-10/supervisor.log')
            print(json.dumps(result, indent=2))
            return result
        raise ValueError('No prepared extension; run detach or verify first')
    corpus = Extension(root)
    expected = [dict(prompt_id=r['id'], subtask=r['subtask']) for r in corpus.rows]
    old = [a['id'] for a in corpus.base_protocol['actions']]
    records = committed_records(root, old+['probe'], protocol_identity(corpus.base_protocol))
    records.update(committed_records(root, EXTRA, protocol_identity(corpus.protocol)))
    probes = records.pop('probe')
    if {r['prompt_id'] for r in probes} != {r['prompt_id'] for r in expected}:
        raise ValueError('Original probe cohort changed')
    _, matching = match_records(records, expected)
    counts = {a: len(records[a]) for a in corpus.actions}
    if final and any(n != len(expected) for n in counts.values()):
        raise ValueError('Missing answers; cannot finalize extension')
    wanted = set(matching['prompt_ids']) if same_count else {r['prompt_id'] for r in expected}
    reports = {}
    for case in corpus.actions:
        values = [dict(r, timings=dict(r['timings'], ttft_seconds=r['timings']['answer_engine_ttft_seconds']))
                  for r in records[case] if r['prompt_id'] in wanted]
        reports[case] = aggregate(values, [r for r in expected if r['prompt_id'] in wanted],
                                  dict(method=case), 'completed' if final else 'live')
    report = dict(status='complete' if final else 'live', scheduled_prompts=len(expected),
        accepted_answers=sum(counts.values()), expected_answers=len(expected)*len(counts),
        new_answers=sum(counts[a] for a in EXTRA), expected_new_answers=len(expected)*len(EXTRA),
        accepted_probes=len(probes), available_counts=counts, reports=reports,
        timing='Fixed-action answer-engine TTFT; original and added actions were measured in different sessions.',
        validation='Result hashes and per-action protocol checked; live reports do not certify completion.')
    if same_count:
        report['matching'] = matching
    supervisor = corpus.state/'supervisor.json'
    if supervisor.exists():
        saved = read(supervisor)
        report['supervisor'] = dict(saved, alive=alive(saved))
    stem = 'report' if final else 'same_count_summary' if same_count else 'live_summary'
    with locked(corpus.state/'report.lock'):
        text = f"# RULER original + 5%/10%\n\nAnswers: {report['accepted_answers']}/{report['expected_answers']}; added: {report['new_answers']}/{report['expected_new_answers']}.\n\n{report['timing']}\n"
        if same_count:
            text += f"\nMatched prompt IDs: {matching['matched_samples']}.\n"
        table = []
        for case, value in reports.items():
            text += '\n'+render_markdown(value)+'\n'
            table.append(dict(method=case, subtask='overall', **value['overall']))
            table.extend(dict(method=case, subtask=t, **m) for t, m in value['subtasks'].items())
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=('method', 'subtask', *COLUMNS))
        writer.writeheader()
        writer.writerows(table)
        for suffix, content in (('md', text), ('csv', stream.getvalue())):
            path = corpus.state/(stem+'.'+suffix)
            tmp = path.with_suffix(path.suffix+'.tmp')
            tmp.write_text(content)
            tmp.replace(path)
        atomic_json(corpus.state/(stem+'.json'), report)
    print(f"{report['status']}: {report['new_answers']}/{report['expected_new_answers']} added answers; {corpus.state/(stem+'.md')}", flush=True)
    return report


def assert_engines_exited(state):
    from runner.router_process import alive, group_alive
    for path in [*(state/'processes').glob('*.json'), *(state/'sessions').glob('*/ownership.json')]:
        receipt = read(path)
        if alive(receipt) or group_alive(receipt['pid']):
            raise ValueError('Owned extension engines remain alive')


def finalize(root):
    from runner.setups import atomic_json
    corpus = Extension(root)
    assert_engines_exited(corpus.state)
    if read(corpus.state/'cleanup.json').get('owned_engines_exited') is not True:
        raise ValueError('Missing extension engine-exit receipt')
    corpus.verify_sources()
    corpus.validate_new()
    if (corpus.state/'complete.json').exists():
        done = read(corpus.state/'complete.json')
        check_files(corpus.state, done['reports'])
        check_files(root, done['new_receipts'])
        return done
    report = summarize(root, final=True)
    done = dict(complete=True, owned_engines_exited=True, prompts=len(corpus.rows),
                new_answers=report['new_answers'], answers=report['accepted_answers'],
                reused_probes=report['accepted_probes'],
                new_receipts={str(Path('records')/case/row['id']/'validated.json'):
                              digest(root/'records'/case/row['id']/'validated.json')
                              for row in corpus.rows for case in EXTRA},
                reports={f'report.{s}': digest(corpus.state/f'report.{s}') for s in ('json', 'md', 'csv')})
    atomic_json(corpus.state/'complete.json', done)
    return done


@contextmanager
def base_exclusive(root):
    """Coordinate with the original launcher's own locks, without replacing state."""
    from scripts.corpus_control import idle
    with locked(root/'launch.lock'):
        idle(root)
        handle = (root/'run.lock').open('a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            handle.close()
            raise
    try:
        yield
    finally:
        handle.close()


def supervise(root):
    from runner.router_process import identity, group_alive
    from runner.setups import atomic_json
    from scripts.corpus_control import check_hardware
    from scripts.router_control import environment, terminate_owned, cleanup_caches
    state_dir = root/STATE
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with base_exclusive(root), locked(state_dir/'run.lock'):
        state = dict(pid=os.getpid(), identity=identity(os.getpid()), state='verifying', started_at=time.time())
        atomic_json(state_dir/'supervisor.json', state)
        running = {}

        def launch(corpus, group, retry):
            attempt = uuid.uuid4().hex
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--root', str(root),
                   '--group', str(group), '--attempt', attempt]
            with (state_dir/f'group{group}-{attempt}.log').open('a') as log:
                child = subprocess.Popen(cmd, cwd=CODE_ROOT, env=environment(','.join(corpus.protocol['groups'][group])),
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            receipt = dict(pid=child.pid, identity=identity(child.pid), group=group, retry=retry, command=cmd, started_at=time.time())
            atomic_json(state_dir/'processes'/f'{attempt}.json', receipt)
            running[group] = dict(child=child, receipt=receipt, last=time.monotonic(), mtime=time.time())

        try:
            corpus = prepare(root)
            check_hardware(corpus.protocol)
            cleanup_caches(state_dir, corpus.protocol)
            for group in range(len(corpus.protocol['groups'])):
                if corpus.pending(group):
                    launch(corpus, group, 0)
            state['state'] = 'collecting'
            atomic_json(state_dir/'supervisor.json', state)
            while running:
                for group, job in list(running.items()):
                    child, receipt = job['child'], job['receipt']
                    progress = state_dir/f'progress-group{group}.json'
                    if progress.exists() and progress.stat().st_mtime > job['mtime']:
                        job.update(last=time.monotonic(), mtime=progress.stat().st_mtime)
                    code = child.poll()
                    if code is None and time.monotonic()-job['last'] > 900:
                        terminate_owned(receipt)
                        child.wait(timeout=10)
                        code = 75
                    if code is None:
                        continue
                    if group_alive(child.pid):
                        terminate_owned(receipt)
                    if group_alive(child.pid):
                        raise RuntimeError('Owned engines remain alive')
                    del running[group]
                    cleanup_caches(state_dir, corpus.protocol)
                    atomic_json(state_dir/'exits'/f'{child.pid}.json', dict(**receipt, exit_code=code, owned_engines_exited=True))
                    if code == 75 and receipt['retry'] == 0:
                        launch(corpus, group, 1)
                    elif code:
                        raise RuntimeError(f'Group {group} failed ({code}); validation failures are not retried')
                if running:
                    time.sleep(2)
            assert_engines_exited(state_dir)
            atomic_json(state_dir/'cleanup.json', dict(owned_engines_exited=True, at=time.time()))
            finalize(root)
            state.update(state='complete', finished_at=time.time())
            atomic_json(state_dir/'supervisor.json', state)
        except BaseException as error:
            for job in running.values():
                terminate_owned(job['receipt'])
                job['child'].wait(timeout=10)
            state.update(state='failed', error=str(error), finished_at=time.time())
            atomic_json(state_dir/'supervisor.json', state)
            raise


def detach(root, resume=False):
    from runner.router_process import identity
    from runner.setups import atomic_json
    from scripts.corpus_control import idle, check_hardware
    from scripts.router_control import environment
    state = root/STATE
    protocol = read(root/'protocol.json')
    count = 13 * read(root/'tranche.json')['limit_per_task']
    with locked(root/'launch.lock'):
        idle(root)
        completion(root, protocol, count)
        state.mkdir(parents=True, exist_ok=True)
    with locked(state/'launch.lock'):
        idle(state)
        if (state/'complete.json').exists():
            raise ValueError('Completed extension cannot restart')
        if (state/'supervisor.json').exists() and not resume:
            raise ValueError('Use resume for missing new records')
        command = ['nohup', sys.executable, '-u', str(Path(__file__).resolve()), 'supervise',
                   '--root', str(root)]
        with locked(root/'launch.lock'):
            idle(root)
            completion(root, protocol, count)
            check_hardware(protocol)
            helper = """import json,subprocess,sys
c=json.load(sys.stdin)
with open(c['log'],'a') as log:
 p=subprocess.Popen(c['command'],cwd=c['cwd'],env=c['env'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid,flush=True)
"""
            result = subprocess.run([sys.executable, '-c', helper], input=json.dumps(dict(command=command,
                cwd=str(CODE_ROOT), env=environment(), log=str(state/'supervisor.log'))), text=True, capture_output=True, check=True)
            pid = int(result.stdout.strip())
            atomic_json(state/'launch.json', dict(pid=pid, identity=identity(pid), command=command, resume=resume))
        for _ in range(200):
            if (state/'supervisor.json').exists() and read(state/'supervisor.json')['pid'] == pid:
                break
            if identity(pid) is None:
                raise RuntimeError('Supervisor exited; inspect extension supervisor.log')
            time.sleep(.1)
        else:
            raise RuntimeError('Supervisor startup timed out; inspect extension before retrying')
        proc = Path('/proc')/str(pid)
        fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        ignored = int(next(s.split()[1] for s in (proc/'status').read_text().splitlines() if s.startswith('SigIgn:')), 16)
        if (int(fields[1]) != 1 or int(fields[3]) != pid or not ignored & 1 or
                os.readlink(proc/'fd/0') != '/dev/null' or
                b'CUDA_VISIBLE_DEVICES=' not in (proc/'environ').read_bytes().split(b'\0')):
            raise RuntimeError('Detachment verification failed; inspect extension before retrying')
        receipt = dict(pid=pid, identity=identity(pid), parent_pid=1, session_id=pid,
                       sighup_ignored=True, stdin='/dev/null', cpu_only=True, resume=resume)
        atomic_json(state/'detachment.json', receipt)
        print(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify', 'detach', 'resume', 'status', 'status_same_count', 'report', 'supervise', 'worker'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--group', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--attempt', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root/'protocol.json').is_file():
        raise ValueError('Missing existing collection protocol: '+str(root))
    if args.command != 'worker' and os.environ.get('CUDA_VISIBLE_DEVICES', ''):
        raise ValueError('Control commands require CUDA_VISIBLE_DEVICES empty')
    if args.command == 'verify':
        from scripts.corpus_control import idle
        with base_exclusive(root):
            idle(root/STATE)
            corpus = prepare(root)
        print(f'Verified {len(corpus.rows)} original prompts; {len(corpus.rows)*2} new answers. No GPU work launched.')
    elif args.command in ('detach', 'resume'):
        detach(root, args.command == 'resume')
    elif args.command == 'supervise':
        supervise(root)
    elif args.command == 'worker':
        try:
            collect(root, args.group, args.attempt)
        except (TimeoutError, ConnectionError):
            import traceback
            traceback.print_exc()
            raise SystemExit(75)
        except Exception:
            import traceback
            traceback.print_exc()
            raise SystemExit(76)
    elif args.command == 'report':
        from scripts.corpus_control import idle
        with base_exclusive(root):
            idle(root/STATE)
            print(json.dumps(finalize(root), indent=2))
    else:
        summarize(root, same_count=args.command == 'status_same_count')


if __name__ == '__main__':
    main()
