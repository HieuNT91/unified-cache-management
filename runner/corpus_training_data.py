"""CPU training/replay view joining original and completed added-action records."""
import json
from pathlib import Path

from runner import corpus_progress as progress
from runner.corpus import Corpus, load_attention, relative
from runner.corpus_records import accepted, protocol_identity, record_dir
from runner.setups import file_hash
from scripts.corpus_add_ratios import STATE, Extension, combined_actions
from runner.corpus_extensions import discover,joined_protocol


def normalize_action_scope(value):
    """Accept named scopes or explicit IDs/percentages; always retain dense fallback."""
    values=[value] if isinstance(value,str) else list(value)
    tokens=[token.strip() for item in values for token in item.split(',') if token.strip()]
    if len(tokens)==1 and tokens[0] in ('all','original'):return tokens[0]
    if not tokens or any(t in ('all','original') for t in tokens):
        raise ValueError('Use all, original, or individual actions; do not mix named scopes with actions')
    selected={'nocache'}
    for token in tokens:
        value=token.rstrip('%')
        selected.add('prophetkv-'+str(int(value)) if value.isdigit() else token)
    if selected=={'nocache'}:raise ValueError('Select at least one sparse action; nocache is retained automatically')
    return tuple(sorted(selected))


def select_actions(corpus,scope):
    if isinstance(scope,str):return corpus
    unknown=set(scope)-set(corpus.actions)
    if unknown:raise ValueError('Actions absent from corpus inventory: '+', '.join(sorted(unknown)))
    return SelectedCorpus(corpus,scope)


def open_corpus(root, prepared=None, tree=None, action_scope='all', skip_validation=False):
    """Open the chosen inventory without rewriting collection protocols or receipts."""
    scope=normalize_action_scope(action_scope)
    root=Path(root)
    # An exported subset tree carries its inventory; default testing inherits it.
    if scope=='all' and tree is not None:
        scope=normalize_action_scope([a['id'] for a in tree['actions']])
    base_scope=scope if isinstance(scope,str) else 'all'
    if not isinstance(scope,str):
        original=json.loads((root/'protocol.json').read_text())
        if set(scope)<=set(a['id'] for a in original['actions']):base_scope='original'
    if skip_validation:
        from runner.corpus_unchecked import UncheckedCorpus
        corpus=UncheckedCorpus(root,prepared,scope)
    elif base_scope=='all':
        extensions=discover(root,None if isinstance(scope,str) else scope)
        if len(extensions)>1:corpus=MultiExtensionCorpus(root,prepared,extensions)
        elif extensions:corpus=ExtendedCorpus(root,prepared,extensions[0][0])
        elif (root/STATE).exists():corpus=ExtendedCorpus(root,prepared)
        else:corpus=Corpus(root,prepared)
    else:
        corpus=Corpus(root,prepared)
    return select_actions(corpus,scope)


def training_view(corpus, action_scope='all'):
    scope=normalize_action_scope(action_scope)
    if isinstance(corpus,SelectedCorpus):
        if corpus.action_scope==scope:return corpus
        return open_corpus(corpus.root,corpus.prepared,action_scope=scope,
                           skip_validation=getattr(corpus,'skip_validation',False))
    if not isinstance(scope,str):
        if isinstance(corpus,(ExtendedCorpus,MultiExtensionCorpus)) or getattr(corpus,'source',None) in ('add5-10-saved-records','multi-extension-saved-records'):
            return open_corpus(corpus.root,corpus.prepared,action_scope=scope,
                               skip_validation=getattr(corpus,'skip_validation',False))
        # Direct callers may have supplied an original reader for added actions.
        if not set(scope)<=set(corpus.actions):
            return open_corpus(corpus.root,corpus.prepared,action_scope=scope,
                               skip_validation=getattr(corpus,'skip_validation',False))
        return select_actions(corpus,scope)
    if getattr(corpus, 'skip_validation', False):
        if corpus.action_scope == scope:return corpus
        return open_corpus(corpus.root,corpus.prepared,action_scope=scope,skip_validation=True)
    if scope == 'original':
        return Corpus(corpus.root, corpus.prepared) if isinstance(corpus,(ExtendedCorpus,MultiExtensionCorpus)) else corpus
    extensions=discover(corpus.root)
    if len(extensions)>1 and not isinstance(corpus,MultiExtensionCorpus):
        return MultiExtensionCorpus(corpus.root,getattr(corpus,'prepared',None),extensions)
    if not isinstance(corpus,(ExtendedCorpus,MultiExtensionCorpus)):
        if extensions:return ExtendedCorpus(corpus.root,getattr(corpus,'prepared',None),extensions[0][0])
        if (Path(corpus.root)/STATE).exists():return ExtendedCorpus(corpus.root,getattr(corpus,'prepared',None))
    return corpus


class SelectedCorpus(Corpus):
    """Read-only subset; underlying readers retain original acceptance identities."""
    def __init__(self,source,scope):
        self._source=source
        self.action_scope=scope
        self.actions={name:value for name,value in source.actions.items() if name in scope}
        self.protocol=dict(source.protocol,actions=list(self.actions.values()))
        self.root=source.root
        self.prepared=source.prepared
        self.rows=source.rows

    def __getattr__(self,name):
        return getattr(self._source,name)

    def inputs(self,prompt_id):return self._source.inputs(prompt_id)
    def probe(self,prompt_id):return self._source.probe(prompt_id)
    def features(self,prompt_id):return self._source.features(prompt_id)
    def attention(self,prompt_id):return self._source.attention(prompt_id)

    def outcome(self,prompt_id,action):
        if action not in self.actions:raise ValueError('Action excluded by action-scope: '+action)
        return self._source.outcome(prompt_id,action)

    def snapshot(self):
        if self.actions==self._source.actions:
            result=self._source.snapshot()
        elif getattr(self,'skip_validation',False):
            from runner.corpus_unchecked import UncheckedCorpus
            result=UncheckedCorpus.snapshot(self)
        else:
            result=Corpus.snapshot(self)
            if isinstance(self._source,(ExtendedCorpus,MultiExtensionCorpus)):
                result.update(source=self._source.source,metadata_hashes=dict(self._source.metadata_hashes))
        result['action_scope']=list(self.actions)
        return result


class ExtendedCorpus(Corpus):
    """Read-only single-batch view; collection protocols/receipts are never rewritten."""
    def __init__(self, root, prepared=None, state=STATE):
        self.root = Path(root).resolve()
        self.relative_state=Path(state)
        self.state = relative(self.root,self.relative_state)
        self.source='completed-add5-10-extension' if self.relative_state==STATE else 'completed-multi-extension'
        completion = self.state/'complete.json'
        if not completion.is_file():
            raise ValueError('Action extension is incomplete; finish collection and final reporting before training')
        from scripts.corpus_control import idle
        idle(self.root)
        idle(self.state)
        self.extension = Extension(self.root,self.relative_state)
        self.extra=self.extension.added_actions
        self.protocol = self.extension.protocol
        self.actions = self.extension.actions
        self.prepared = Path(self.protocol['prepared'])
        if prepared is not None and Path(prepared).resolve() != self.prepared.resolve():
            raise ValueError('Extension training must use its original pinned prepared directory')
        if self.protocol['actions'] != combined_actions(self.extension.base_protocol['actions'],[int(a.removeprefix('prophetkv-')) for a in self.extra]):
            raise ValueError('Extension action inventory changed')
        self.rows = {r['id']: r for r in self.extension.rows}
        if len(self.rows) != len(self.extension.rows) or not self.rows:
            raise ValueError('Duplicate or empty extension cohort')
        self.done = json.loads(completion.read_text())
        n = len(self.rows)
        if (self.done.get('complete') is not True or self.done.get('owned_engines_exited') is not True or
                self.done.get('prompts') != n or self.done.get('new_answers') != len(self.extra)*n or
                self.done.get('answers') != len(self.actions)*n or self.done.get('reused_probes') != n):
            raise ValueError('Extension completion counts or engine-exit gate changed')
        if json.loads((self.state/'cleanup.json').read_text()).get('owned_engines_exited') is not True:
            raise ValueError('Missing extension engine-exit evidence')
        expected = {f'records/{a}/{pid}/validated.json' for pid in self.rows for a in self.extra}
        if set(self.done.get('new_receipts', {})) != expected:
            raise ValueError('Incomplete extension receipt inventory')
        for name, digest in progress.track(self.done['new_receipts'].items(), 'Checking extension receipts', 'files', lambda item: item[0]):
            if file_hash(relative(self.root, name)) != digest:
                raise ValueError('Completed extension receipt changed: '+name)
        for name, digest in progress.track(self.done['reports'].items(), 'Checking extension reports', 'files', lambda item: item[0]):
            if file_hash(relative(self.state, name)) != digest:
                raise ValueError('Completed extension report changed: '+name)
        if file_hash(self.prepared/'plan.json') != self.protocol['plan_sha256']:
            raise ValueError('Original prepared plan changed')
        self.metadata_hashes = {name: file_hash(relative(self.root, name)) for name in (
            'protocol.json', str(self.relative_state/'protocol.json'), str(self.relative_state/'sources.json'),
            str(self.relative_state/'complete.json'), str(self.relative_state/'cleanup.json'))}
        self._features = {}

    def _receipt(self, prompt_id, case):
        name = f'records/{case}/{prompt_id}/validated.json'
        path = relative(self.root, name)
        pins = self.done['new_receipts'] if case in self.extra else self.extension.sources['files']
        expected = pins.get(name if case in self.extra else str(path))
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
        if action in self.extra:
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
        progress.detail('waiting for corpus publication lock')
        with (self.root/'publication.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            for pid, row in progress.track(self.rows.items(), 'Validating snapshot', 'samples', lambda item: item[0]):
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
            metadata_hashes=dict(self.metadata_hashes), source=self.source,
            counts=dict(planned=n, prepared=n, complete=n, incomplete=0, unprepared=0,
                        accepted={case: n for case in ['probe', *self.actions]}))


class MultiExtensionCorpus(Corpus):
    """Join disjoint batches while validating each record under its own protocol."""
    def __init__(self,root,prepared=None,extensions=None):
        self.root=Path(root).resolve()
        extensions=discover(self.root) if extensions is None else extensions
        self.base=Corpus(self.root,prepared)
        self.batches=[ExtendedCorpus(self.root,prepared,state) for state,_ in extensions]
        self.protocol=joined_protocol(self.base.protocol,extensions)
        self.actions={a['id']:a for a in self.protocol['actions']}
        self.prepared=self.base.prepared
        self.rows={pid:row for batch in self.batches for pid,row in batch.rows.items()}
        self.owners={action:batch for batch in self.batches for action in batch.extra}
        self.metadata_hashes={name:value for batch in self.batches for name,value in batch.metadata_hashes.items()}
        self.source='completed-multi-extension'

    def inputs(self,prompt_id):return self.base.inputs(prompt_id)
    def probe(self,prompt_id):return self.base.probe(prompt_id)
    def features(self,prompt_id):return self.base.features(prompt_id)
    def attention(self,prompt_id):return self.base.attention(prompt_id)

    def outcome(self,prompt_id,action):
        if action not in self.actions:raise ValueError('Unknown corpus action')
        owner=self.owners.get(action,self.base)
        if prompt_id not in owner.rows:return None
        return owner.outcome(prompt_id,action)

    def snapshot(self):
        for batch in self.batches:
            batch.extension.verify_sources()
            batch.extension.validate_new(full=True)
        result=Corpus.snapshot(self)
        for name,expected in self.metadata_hashes.items():
            if file_hash(relative(self.root,name))!=expected:raise ValueError('Extension metadata changed')
        result.update(source=self.source,metadata_hashes=dict(self.metadata_hashes))
        return result
