"""Append-only action collection linked to an immutable, completed corpus."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from runner.corpus import Corpus, relative, selection, match_answer
from runner.corpus_records import accepted, record_dir, protocol_identity
from runner.router_policy import features_from_arrays
from runner.setups import atomic_json, file_hash, fingerprint
from runner.tree_policy import actions


def check_pins(pins):
    for name, digest in pins.items():
        path=Path(name)
        if not path.is_file() or file_hash(path)!=digest:
            raise ValueError('Pinned source changed or missing: '+name)


def inventory(base, added):
    extra=[dict(id=f'prophetkv-{p}',method='prophetkv',ratio=p/100) for p in added]
    combined=[*base,*extra];actions(combined)
    return sorted(combined,key=lambda a:a['ratio'] or 0),[a['id'] for a in extra]


def derive(attention, sample, definitions):
    return {a:selection(attention['scores'],sample['boundaries'][1],d['ratio'])
            for a,d in actions(definitions).items() if a!='nocache'}


class LinkedCorpus:
    def __init__(self, root):
        self.root=Path(root);self.protocol=json.loads((self.root/'protocol.json').read_text())
        self.sources=json.loads((self.root/'sources.json').read_text())
        if file_hash(self.root/'sources.json')!=self.protocol['sources_sha256']:
            raise ValueError('Extension source inventory changed')
        self.base=Corpus(self.protocol['base_corpus']);self.prepared=self.base.prepared
        if protocol_identity(self.base.protocol)!=self.protocol['base_protocol_sha256']:
            raise ValueError('Original protocol changed')
        self.rows=self.base.rows;self.actions=actions(self.protocol['actions'])
        if list(self.rows)!=self.protocol['prompt_ids']:raise ValueError('Original prompt order changed')

    def verify_sources(self):check_pins(self.sources['files'])

    def inputs(self,pid):return self.base.inputs(pid)

    def _original(self,pid,case):
        folder=record_dir(self.base.root,case,pid)
        receipt=folder/'validated.json'
        if not receipt.is_file() or file_hash(receipt)!=self.sources['files'].get(str(receipt)):raise ValueError('Linked receipt changed or missing')
        result=self.base.probe(pid) if case=='probe' else self.base.outcome(pid,case)
        if result is None:raise ValueError('Missing linked record')
        return result

    def probe(self,pid):return self._original(pid,'probe')

    def outcome(self,pid,action):
        if action not in self.actions:raise ValueError('Unknown action')
        if action in self.base.actions:return self._original(pid,action)
        result=accepted(self.root,action,self.rows[pid],self.protocol)
        if result is not None:
            expected=self.provenance(pid)
            if result['derived_mask_provenance']!=expected:raise ValueError('Derived-mask provenance changed')
            if not result.get('alignment_audit') or any(not a['normalized'] or a['delta_amplitude']!=1. or a['aliases_native_table'] for a in result['alignment_audit']):
                raise ValueError('Missing YaRN normalization evidence')
            if result.get('cache_deletion',{}).get('deleted_shards',0)<=0:raise ValueError('Cache was not deleted')
        return result

    def provenance(self,pid):
        p=self.root/'derived'/f'{pid}.json'
        expected=self.protocol['derived_receipts'][pid]
        if file_hash(p)!=expected:raise ValueError('Derived receipt changed')
        meta=json.loads(p.read_text());check_pins(meta['source_files'])
        path=self.root/'derived'/f'{pid}.npz'
        if file_hash(path)!=meta['sha256']:raise ValueError('Derived masks changed')
        return dict(receipt_sha256=expected,archive_sha256=meta['sha256'],base_probe_sha256=meta['base_probe_sha256'])

    def attention(self,pid,replay=False):
        self.provenance(pid)
        with np.load(self.root/'derived'/f'{pid}.npz',allow_pickle=False) as data:
            result=dict(scores=data['scores'].copy(),selections={a:data[a].copy() for a in self.actions if a!='nocache'})
        if replay:
            original=self.base.attention(pid);masks=derive(original,self.inputs(pid),self.protocol['actions'])
            if not np.array_equal(result['scores'],original['scores']) or any(not np.array_equal(result['selections'][a],v) for a,v in masks.items()):
                raise ValueError('Derived masks differ from original replay')
        return result

    def features(self,pid):return self.probe(pid)['features']

    def pending(self):
        return [r for pid,r in self.rows.items() if any(self.outcome(pid,a) is None for a in self.protocol['added_actions'])]

    def matrix(self, full=True):
        self.verify_sources();ids=list(self.rows);scores=[];times=[];features=[];probe=[];caps=[];pins={}
        for pid in ids:
            sample=self.inputs(pid);row=self.rows[pid];records=[]
            attention=self.attention(pid) if full else None
            for action in self.actions:
                record=self.outcome(pid,action)
                if record is None:raise ValueError('Incomplete extension collection')
                if full and action in self.protocol['added_actions']:
                    ds=json.loads((record_dir(self.root,action,pid)/'diagnostics.json').read_text())
                    match_answer(ds,sample,action,attention,self.protocol['actions'])
                origin=self.base.root if action in self.base.actions else self.root
                path=record_dir(origin,action,pid)/'validated.json';pins[str(path)]=file_hash(path)
                records.append(record)
            scores.append([r['accuracy'] for r in records]);times.append([r['timings']['answer_engine_ttft_seconds'] for r in records])
            caps.append([r['output_cap_reached'] for r in records]);features.append(self.features(pid));probe.append(self.probe(pid)['timings']['probe_ttft_seconds'])
        return dict(ids=ids,actions=self.protocol['actions'],features=features,scores=scores,answer_ttft=times,probe_ttft=probe,
                    output_caps=caps,input_hashes=[self.rows[p]['sha256'] for p in ids],tasks=[self.rows[p]['subtask'] for p in ids],
                    acceptance_hashes=pins,protocol_sha256=protocol_identity(self.protocol))


def prepare(root, base_root, added, previous):
    from scripts.corpus_control import idle
    from scripts.router_control import code_hashes
    root=Path(root);base_root=Path(base_root).resolve();previous=Path(previous).resolve()
    if root.exists():raise FileExistsError('Use a fresh extension directory')
    idle(base_root);base=Corpus(base_root)
    completion=base_root.parent/'collection-complete.json';done=json.loads(completion.read_text())
    if not done['complete'] or not done['owned_engines_exited']:raise ValueError('Base collection is not complete')
    if len(base.rows)!=260 or set(Counter(r['subtask'] for r in base.rows.values()).values())!={20}:raise ValueError('Expected original 260 prompts, 20/task')
    current=code_hashes()
    if any(current.get(name)!=digest for name,digest in base.protocol['code'].items()):
        raise ValueError('Original runtime behavior must remain unchanged in an action-only extension')
    combined,extra=inventory(base.protocol['actions'],added)
    if len(base.protocol['groups'])!=1:raise ValueError('Extension requires original single TP4 group')
    pins={str(completion):file_hash(completion),str(base_root/'protocol.json'):file_hash(base_root/'protocol.json')}
    for name,digest in done['files'].items():
        p=relative(base_root.parent,name)
        if file_hash(p)!=digest:raise ValueError('Base completion artifact changed')
        pins[str(p)]=digest
    for name,digest in base.protocol['code'].items():
        p=relative(base_root.parent/'frozen-code',name)
        if file_hash(p)!=digest:raise ValueError('Original runtime changed')
        pins[str(p)]=digest
    for p in base.prepared.rglob('*'):
        if p.is_file():pins[str(p)]=file_hash(p)
    from runner.tree_policy import load
    load(previous)
    for p in (previous,Path(str(previous)+'.sha256')):pins[str(p)]=file_hash(p)
    root.mkdir(parents=True);(root/'derived').mkdir();derived={}
    for n,(pid,row) in enumerate(base.rows.items()):
        sample=base.inputs(pid)
        from runner.config import validate_sample
        validate_sample(sample)
        if sample['thinking']:raise ValueError('Input profile changed')
        for case in ['probe',*base.actions]:
            record=base.probe(pid) if case=='probe' else base.outcome(pid,case)
            if record is None:raise ValueError('Incomplete original collection')
            folder=record_dir(base_root,case,pid);receipt=folder/'validated.json'
            pins[str(receipt)]=file_hash(receipt)
            for name,digest in json.loads(receipt.read_text())['files'].items():pins[str(relative(folder,name))]=digest
            p=relative(base_root,record['initialization']);pins[str(p)]=record['initialization_sha256']
        attention=base.attention(pid)
        features=features_from_arrays(attention['layers'].astype(np.float64).mean(0),attention['scores'],sample['boundaries'][1],sample['boundaries'][-2])
        probe=base.probe(pid)
        if features!=probe['features']:raise ValueError('Archived features differ from replay')
        path=root/'derived'/f'{pid}.npz';np.savez_compressed(path,scores=attention['scores'],**derive(attention,sample,combined))
        folder=record_dir(base_root,'probe',pid)
        source_files={str(folder/'validated.json'):pins[str(folder/'validated.json')]}
        source_files.update({str(relative(folder,a['path'])):a['sha256'] for a in probe['artifacts']})
        meta=dict(sha256=file_hash(path),source_files=source_files,base_probe_sha256=file_hash(folder/'validated.json'),
            original_inventory=base.protocol['actions'],derived_inventory=combined,features=features,
            arithmetic='Native FP32 scores; floor(N*ratio); stable ascending position ties')
        atomic_json(root/'derived'/f'{pid}.json',meta);derived[pid]=file_hash(root/'derived'/f'{pid}.json')
        if (n+1)%10==0:print(f'Pinned/replayed {n+1}/260 prompts',flush=True)
    atomic_json(root/'sources.json',dict(files=pins))
    protocol=dict(base.protocol,schema='ruler-action-extension-v1',kind='extension',actions=combined,added_actions=extra,
        base_corpus=str(base_root),base_protocol_sha256=protocol_identity(base.protocol),sources_sha256=file_hash(root/'sources.json'),
        derived_receipts=derived,prompt_ids=list(base.rows),previous_tree=str(previous),code=code_hashes(),cache_root=str(root/'cache'))
    atomic_json(root/'protocol.json',protocol)
    return dict(prompts=260,new_answers=260*len(extra),reused_probes=260)
