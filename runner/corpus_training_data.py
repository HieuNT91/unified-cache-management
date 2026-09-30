"""CPU training/replay view joining original and completed added-action records."""
import json
from pathlib import Path

from runner.corpus import Corpus, load_attention, relative
from runner.corpus_records import accepted, protocol_identity, record_dir
from runner.setups import file_hash
from scripts.corpus_add_ratios import EXTRA, STATE, Extension, combined_actions


def open_corpus(root, prepared=None, tree=None):
    """Default to all collected actions; old trees keep their original evidence."""
    root = Path(root)
    if (root/STATE).exists():
        original = json.loads((root/'protocol.json').read_text())
        if (tree is not None and tree['actions'] == original['actions'] and
                tree['provenance']['corpus_protocol_sha256'] == protocol_identity(original)):
            return Corpus(root, prepared)
        return ExtendedCorpus(root, prepared)
    return Corpus(root, prepared)


def training_view(corpus):
    # Also cover direct Python calls to train(Corpus(...), ...).
    if not isinstance(corpus, ExtendedCorpus) and (Path(corpus.root)/STATE).exists():
        return ExtendedCorpus(corpus.root, getattr(corpus, 'prepared', None))
    return corpus


class ExtendedCorpus(Corpus):
    """Read-only six-action view; collection protocols/receipts are never rewritten."""
    def __init__(self, root, prepared=None):
        self.root = Path(root).resolve()
        self.state = self.root/STATE
        completion = self.state/'complete.json'
        if not completion.is_file():
            raise ValueError('5%/10% extension is incomplete; finish collection and final reporting before training')
        from scripts.corpus_control import idle
        idle(self.root)
        idle(self.state)
        self.extension = Extension(self.root)
        self.protocol = self.extension.protocol
        self.actions = self.extension.actions
        self.prepared = Path(self.protocol['prepared'])
        if prepared is not None and Path(prepared).resolve() != self.prepared.resolve():
            raise ValueError('Extension training must use its original pinned prepared directory')
        if self.protocol['actions'] != combined_actions(self.extension.base_protocol['actions']):
            raise ValueError('Extension action inventory changed')
        self.rows = {r['id']: r for r in self.extension.rows}
        if len(self.rows) != len(self.extension.rows) or not self.rows:
            raise ValueError('Duplicate or empty extension cohort')
        self.done = json.loads(completion.read_text())
        n = len(self.rows)
        if (self.done.get('complete') is not True or self.done.get('owned_engines_exited') is not True or
                self.done.get('prompts') != n or self.done.get('new_answers') != len(EXTRA)*n or
                self.done.get('answers') != len(self.actions)*n or self.done.get('reused_probes') != n):
            raise ValueError('Extension completion counts or engine-exit gate changed')
        if json.loads((self.state/'cleanup.json').read_text()).get('owned_engines_exited') is not True:
            raise ValueError('Missing extension engine-exit evidence')
        expected = {f'records/{a}/{pid}/validated.json' for pid in self.rows for a in EXTRA}
        if set(self.done.get('new_receipts', {})) != expected:
            raise ValueError('Incomplete extension receipt inventory')
        for name, digest in self.done['new_receipts'].items():
            if file_hash(relative(self.root, name)) != digest:
                raise ValueError('Completed extension receipt changed: '+name)
        for name, digest in self.done['reports'].items():
            if file_hash(relative(self.state, name)) != digest:
                raise ValueError('Completed extension report changed: '+name)
        if file_hash(self.prepared/'plan.json') != self.protocol['plan_sha256']:
            raise ValueError('Original prepared plan changed')
        self.metadata_hashes = {name: file_hash(relative(self.root, name)) for name in (
            'protocol.json', str(STATE/'protocol.json'), str(STATE/'sources.json'),
            str(STATE/'complete.json'), str(STATE/'cleanup.json'), str(STATE/'frozen-code/runtime.json'))}
        self._features = {}

    def _receipt(self, prompt_id, case):
        name = f'records/{case}/{prompt_id}/validated.json'
        path = relative(self.root, name)
        pins = self.done['new_receipts'] if case in EXTRA else self.extension.sources['files']
        expected = pins.get(name if case in EXTRA else str(path))
        if expected is None or not path.is_file() or file_hash(path) != expected:
            raise ValueError('Frozen training receipt changed or missing: '+name)
        return expected

    def probe(self, prompt_id):
        self._receipt(prompt_id, 'probe')
        return accepted(self.root, 'probe', self.rows[prompt_id], self.extension.base_protocol,
                        prepared=self.prepared)

    def outcome(self, prompt_id, action):
        if action not in self.actions:
            raise ValueError('Unknown corpus action')
        self._receipt(prompt_id, action)
        if action in EXTRA:
            return self.extension.outcome(self.rows[prompt_id], action)
        return accepted(self.root, action, self.rows[prompt_id], self.extension.base_protocol,
                        prepared=self.prepared)

    def attention(self, prompt_id):
        import numpy as np
        # Original archives contain only the original masks. Replay those first,
        # then compare added masks against the same native FP32 scores.
        sample = self.inputs(prompt_id)
        original = load_attention(record_dir(self.root, 'probe', prompt_id), self.probe(prompt_id),
                                  sample, self.extension.base_protocol['actions'])
        derived = self.extension.attention(self.rows[prompt_id])
        from runner.corpus import selection
        if (not np.array_equal(original['scores'], derived['scores']) or
                any(not np.array_equal(mask, selection(original['scores'], sample['boundaries'][1],
                                                       self.actions[a]['ratio']))
                    for a, mask in derived['selections'].items())):
            raise ValueError('Added masks differ from original attention')
        original['selections'] = derived['selections']
        return original

    def features(self, prompt_id):
        self._receipt(prompt_id, 'probe')
        if prompt_id not in self._features:
            features = super().features(prompt_id)
            if features != self.probe(prompt_id)['features']:
                raise ValueError('Saved probe features differ from original attention replay')
            self._features[prompt_id] = features
        return dict(self._features[prompt_id])

    def snapshot(self):
        import fcntl
        self.extension.verify_sources()
        self.extension.validate_new(full=True)
        pins = {}
        with (self.root/'publication.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            for pid, row in self.rows.items():
                self.inputs(pid)
                if file_hash(relative(self.prepared, row['raw'])) != row['raw_sha256']:
                    raise ValueError('Original raw input changed')
                for case in ['probe', *self.actions]:
                    record = self.probe(pid) if case == 'probe' else self.outcome(pid, case)
                    if record is None:
                        raise ValueError('Completed extension lost a measured outcome')
                    pins[f'records/{case}/{pid}/validated.json'] = self._receipt(pid, case)
        for name, digest in self.metadata_hashes.items():
            if file_hash(relative(self.root, name)) != digest:
                raise ValueError('Training source metadata changed')
        # Keep the ordinary record path shape for frozen held-out evidence.
        n = len(self.rows)
        return dict(schema='ruler-corpus-snapshot-v1', protocol_sha256=protocol_identity(self.protocol),
            plan_sha256=self.protocol['plan_sha256'], actions=self.protocol['actions'],
            samples=[dict(id=pid, task=r['subtask'], ordinal=r['ordinal'], input_sha256=r['sha256'])
                     for pid, r in self.rows.items()], acceptance_hashes=pins,
            metadata_hashes=dict(self.metadata_hashes), source='completed-add5-10-extension',
            counts=dict(planned=n, prepared=n, complete=n, incomplete=0, unprepared=0,
                        accepted={case: n for case in ['probe', *self.actions]}))
