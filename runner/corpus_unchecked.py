"""Explicit offline fast path: trust saved features/results, without artifact replay."""
import json
from pathlib import Path

from runner import corpus_progress as progress
from runner.corpus import relative
from runner.corpus_records import protocol_identity


class UncheckedCorpus:
    skip_validation = True

    def __init__(self, root, prepared=None, action_scope='all', tree=None):
        self.root = Path(root).resolve()
        self.action_scope = action_scope
        original = self.read(self.root/'protocol.json')
        from runner.corpus_extensions import discover,joined_protocol
        from runner.corpus_training_data import normalize_action_scope
        self.action_scope=normalize_action_scope(action_scope)
        if tree is not None and self.action_scope=='all':
            self.action_scope=normalize_action_scope([a['id'] for a in tree['actions']])
        requested=None if isinstance(self.action_scope,str) else self.action_scope
        extensions=[] if self.action_scope=='original' else discover(self.root,requested)
        self.protocol=joined_protocol(original,extensions)
        self.prepared = Path(prepared or self.protocol['prepared'])
        self.actions = {a['id']: dict(a) for a in self.protocol['actions']}
        self.source = ('add5-10-saved-records' if len(extensions)==1 and str(extensions[0][0])=='extensions/add5-10'
                       else 'multi-extension-saved-records' if extensions else 'original')
        if extensions:
            # Use union of source rows; snapshot requires all selected outcomes.
            by_id={}
            for state,_ in extensions:
                for row in self.read(self.root/state/'sources.json')['rows']:
                    by_id[row['id']]=row
            rows=list(by_id.values())
        elif (self.prepared/'manifest.jsonl').is_file():
            rows = [json.loads(line) for line in (self.prepared/'manifest.jsonl').read_text().splitlines() if line.strip()]
        else:
            plan = self.read(self.prepared/'plan.json')
            paths = [self.prepared/'generation'/e['id']/'receipt.json' for e in plan['entries']]
            rows = [self.read(p)['row'] for p in paths if p.is_file()]
        self.rows = {r['id']: r for r in rows}

    @staticmethod
    def read(path):
        return json.loads(Path(path).read_text())

    def inputs(self, prompt_id):
        return self.read(relative(self.prepared, self.rows[prompt_id]['prepared']))

    def outcome(self, prompt_id, action):
        if action not in self.actions:
            raise ValueError('Unknown corpus action')
        return self.read(relative(self.root, f'records/{action}/{prompt_id}/result.json'))

    def probe(self, prompt_id):
        return self.read(relative(self.root, f'records/probe/{prompt_id}/result.json'))

    def features(self, prompt_id):
        # Never reconstruct attention in this mode. Absent saved features fail
        # explicitly rather than silently falling back to expensive archive I/O.
        return dict(self.probe(prompt_id)['features'])

    def snapshot(self):
        cases = ['probe', *self.actions]
        samples = []
        counts = dict.fromkeys(cases, 0)
        for pid, row in progress.track(self.rows.items(), 'Listing saved samples (validation skipped)',
                                      'samples', lambda item: item[0]):
            present = []
            for case in cases:
                folder = relative(self.root, f'records/{case}/{pid}')
                found = (folder/'validated.json').is_file() and (folder/'result.json').is_file()
                counts[case] += int(found)
                present.append(found)
            if all(present):
                samples.append(dict(id=pid, task=row['subtask'], ordinal=row['ordinal'], input_sha256=row['sha256']))
        return dict(schema='ruler-corpus-snapshot-v1', protocol_sha256=protocol_identity(self.protocol),
                    plan_sha256=self.protocol['plan_sha256'], actions=self.protocol['actions'], samples=samples,
                    acceptance_hashes={}, source=self.source, dataset_validation='skipped',
                    counts=dict(planned=len(self.rows), prepared=len(self.rows), complete=len(samples),
                                incomplete=len(self.rows)-len(samples), unprepared=0, accepted=counts))
