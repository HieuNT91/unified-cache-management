"""CPU interfaces for lossless attention archives and measured outcome lookup."""
import itertools
import json
import math
from pathlib import Path
import numpy as np
from runner.setups import file_hash
from runner.tree_policy import actions as action_inventory


def relative(root,name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('Expected a relative artifact path')
    path=Path(root)/p
    if not path.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Artifact escapes corpus')
    return path


def selection(scores,prefix,ratio):
    if scores.dtype!=np.float32 or scores.ndim!=1 or not np.isfinite(scores).all() or (scores<0).any():
        raise ValueError('Invalid FP32 attention scores')
    return np.sort(np.argsort(-scores,kind='stable')[:math.floor(len(scores)*ratio)]+prefix).astype(np.int64)


def reduction_matches(means,scores,prefix):
    # NCCL is free to select a reduction tree. Accept only exact FP32 results of
    # a TP-rank binary reduction, never a numerical tolerance. NCCL can use
    # different rank orders for different tensor slices.
    def trees(values):
        if len(values)==1:
            yield values[0];return
        for split in range(1,len(values)):
            for left in trees(values[:split]):
                for right in trees(values[split:]):yield np.add(left,right,dtype=np.float32)
    matched=np.zeros(scores.shape,dtype=bool)
    for order in itertools.permutations(means):
        for total in trees(order):
            matched |= (total/np.float32(len(means)))[prefix:]==scores
            if matched.all():return True
    return False


def load_attention(folder,receipt,sample,inventory=None,head_layers=None,tp=4):
    from runner.tensor_parallel import validate_tp
    validate_tp(tp)
    inventory=action_inventory(inventory)
    folder=Path(folder);artifacts=receipt['artifacts']
    if sorted(a['rank'] for a in artifacts)!=list(range(tp)):raise ValueError('Missing/duplicate attention rank')
    prefix,end=sample['boundaries'][1],sample['boundaries'][-2]
    layers=[];means=[];native=None;heads=[]
    for item in sorted(artifacts,key=lambda a:a['rank']):
        path=relative(folder,item['path'])
        if file_hash(path)!=item['sha256']:raise ValueError('Corrupt attention archive')
        with np.load(path,allow_pickle=False) as data:
            local=data['layers'];scores=data['scores'];mean=data['local_mean']
            if (local.dtype!=np.float32 or local.shape!=(64,end) or not np.isfinite(local).all()
                    or (local<0).any() or mean.dtype!=np.float32 or mean.shape!=(end,)
                    or scores.dtype!=np.float32 or scores.shape!=(end-prefix,)):
                raise ValueError('Invalid attention dimensions/dtype/values')
            replay=np.zeros(end,dtype=np.float32)
            for layer in local:np.add(replay,layer,out=replay)
            replay/=np.float32(64)
            if not np.array_equal(replay,mean):raise ValueError('Ascending FP32 layer replay mismatch')
            if native is not None and not np.array_equal(native,scores):raise ValueError('TP score disagreement')
            for key,value in dict(context_positions=np.arange(end),question_positions=sample['question_positions'],
                                  boundaries=sample['boundaries'],original_to_formatted=sample.get('original_to_formatted',list(range(len(sample['token_ids']))))).items():
                if not np.array_equal(data[key],value):raise ValueError('Position mapping mismatch')
            for action,definition in inventory.items():
                if action=='nocache':continue
                if not np.array_equal(data[action],selection(scores,prefix,definition['ratio'])):
                    raise ValueError('Selection floor/tie replay mismatch')
            layers.append(local.copy());means.append(mean.copy());native=scores.copy()
            if head_layers is not None:
                head=data['heads']
                if (not np.array_equal(data['head_layers'],head_layers) or head.dtype!=np.float32
                        or head.ndim!=3 or head.shape[0]!=len(head_layers) or head.shape[1]!=64//tp
                        or head.shape[2]!=end or not np.isfinite(head).all() or (head<0).any()):
                    raise ValueError('Invalid all-head capture (Qwen3-32B requires 64/TP Q heads/rank)')
                heads.append(head.copy())
    if not reduction_matches(means,native,prefix):raise ValueError('Native TP reduction cannot be replayed')
    result=dict(layers=np.stack(layers),scores=native,local_means=np.stack(means),
                selections={a:selection(native,prefix,d['ratio']) for a,d in inventory.items() if a!='nocache'})
    if head_layers is not None:result.update(heads=np.stack(heads),head_layers=list(head_layers))
    return result


def match_answer(diagnostics,sample,action,attention,inventory=None,tp=4):
    inventory=action_inventory(inventory)
    if action not in inventory or action=='nocache':raise ValueError('Unknown sparse action')
    from run import verify_diagnostics
    verify_diagnostics(diagnostics,sample,'prophetkv',inventory[action]['ratio'],tp,range(64))
    for worker in diagnostics:
        event=next(e for e in worker['diagnostics'] if e['kind']=='prophetkv_selection')
        if (not np.array_equal(np.asarray(event['scores'],dtype=np.float32),attention['scores']) or
                not np.array_equal(event['selected_positions'],attention['selections'][action])):
            raise ValueError('Sparse answer differs from independent probe replay')


class Corpus:
    """Lazy CPU loader; explicit prepared override supports relocation."""
    def __init__(self,root,prepared=None):
        self.root=Path(root)
        self.protocol=json.loads((self.root/'protocol.json').read_text())
        self.prepared=Path(prepared or self.protocol['prepared'])
        self.actions=action_inventory(self.protocol['actions'])
        from scripts.corpus_inputs import prepared_rows
        if file_hash(self.prepared/'plan.json')!=self.protocol['plan_sha256']:
            raise ValueError('Prepared corpus plan differs from collection protocol')
        self.rows={r['id']:r for r in prepared_rows(self.prepared)}

    def inputs(self,prompt_id):
        row=self.rows[prompt_id];path=relative(self.prepared,row['prepared'])
        if file_hash(path)!=row['sha256']:raise ValueError('Corrupt prepared input')
        return json.loads(path.read_text())

    def outcome(self,prompt_id,action):
        from runner.corpus_records import accepted
        if action not in self.actions:raise ValueError('Unknown corpus action')
        return accepted(self.root,action,self.rows[prompt_id],self.protocol,prepared=self.prepared)

    def probe(self,prompt_id):
        from runner.corpus_records import accepted
        return accepted(self.root,'probe',self.rows[prompt_id],self.protocol,prepared=self.prepared)

    def attention(self,prompt_id):
        from runner.corpus_records import record_dir
        receipt=self.probe(prompt_id)
        if receipt is None:raise ValueError('Probe is not accepted')
        return load_attention(record_dir(self.root,'probe',prompt_id),receipt,self.inputs(prompt_id),self.protocol['actions'])

    def features(self,prompt_id):
        from runner.router_policy import features_from_arrays
        sample=self.inputs(prompt_id);attention=self.attention(prompt_id)
        return features_from_arrays(attention['layers'].astype(np.float64).mean(0),attention['scores'],
                                    sample['boundaries'][1],sample['boundaries'][-2])

    def snapshot(self):
        # Freeze visible receipt names first. Publication is immutable and atomic.
        from runner.corpus_records import record_dir
        from runner.corpus_records import protocol_identity
        cases=['probe',*self.actions]
        visible={(pid,c):record_dir(self.root,c,pid)/'validated.json' for pid in self.rows for c in cases}
        import fcntl
        with (self.root/'publication.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_SH)
            visible={key:file_hash(p) for key,p in visible.items() if p.exists()}
        complete=[];counts={c:0 for c in cases};pins={}
        for pid,row in self.rows.items():
            for path_key,hash_key in (('prepared','sha256'),('raw','raw_sha256')):
                path=relative(self.prepared,row[path_key])
                if not path.is_file() or file_hash(path)!=row[hash_key]:raise ValueError('Corrupt accepted preparation')
            good=True
            for case in cases:
                if (pid,case) not in visible:good=False;continue
                record=self.probe(pid) if case=='probe' else self.outcome(pid,case)
                if record is None:raise ValueError('Accepted receipt disappeared')
                path=record_dir(self.root,case,pid)/'validated.json'
                if file_hash(path)!=visible[pid,case]:raise ValueError('Accepted receipt changed during snapshot')
                counts[case]+=1;pins[str(path.relative_to(self.root))]=visible[pid,case]
            if good:complete.append(dict(id=pid,task=row['subtask'],ordinal=row['ordinal'],input_sha256=row['sha256']))
        return dict(schema='ruler-corpus-snapshot-v1',protocol_sha256=protocol_identity(self.protocol),
            plan_sha256=self.protocol['plan_sha256'],actions=self.protocol['actions'],samples=complete,
            acceptance_hashes=pins,counts=dict(planned=2600,prepared=len(self.rows),complete=len(complete),
            incomplete=len(self.rows)-len(complete),unprepared=2600-len(self.rows),accepted=counts))
