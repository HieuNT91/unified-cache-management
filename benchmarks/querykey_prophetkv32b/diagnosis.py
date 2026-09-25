"""Post-inference evidence analysis; never imported by a GPU worker."""
import numpy as np
from common import load

def evidence_metrics(m,selected,scores,layer_scope):
    ann=load(m['evidence_path']);prefix,end=m['boundaries'][1],m['boundaries'][-2]
    mask=np.zeros(end-prefix,bool);mask[np.asarray(selected,dtype=int)-prefix]=True
    # Selected5 preserves native sum arithmetic; divide only here for comparable probability mass.
    probability=np.asarray(scores,dtype=np.float64)/(5 if layer_scope=='selected5' else 1)
    assert len(probability)==len(mask)
    out={};unions={};occurrences=[]
    for kind in ['key','answer','evidence']:
        bits=np.zeros(len(mask),bool)
        for span in ann['spans']:
            if span['kind']!=kind:continue
            idx=np.array([i-prefix for a,b in span['token_runs'] for i in range(max(prefix,a),min(end,b))],dtype=int)
            bits[idx]=True
            if kind=='evidence' and len(idx):
                occurrences.append(dict(label=span['label'],text=span['text'],tokens=len(idx),
                    coverage=float(mask[idx].mean()),complete=bool(mask[idx].all()),attention_mass=float(probability[idx].sum())))
        unions[kind]=bits
        out[kind+'_tokens']=int(bits.sum())
        out[kind+'_recall']=float(mask[bits].mean()) if bits.any() else None
        out[kind+'_attention_mass']=float(probability[bits].sum()) if bits.any() else None
    key,value=out['key_recall'],out['answer_recall']
    out['harmonic_recall']=(2*key*value/(key+value) if key+value else 0.) if key is not None and value is not None else None
    out['complete_statement_retention']=float(np.mean([o['complete'] for o in occurrences])) if m['label'].startswith('niah_') and occurrences else None
    out['eligible_attention_mass']=float(probability.sum())
    out['occurrences']=occurrences
    return out

def overlap(a,b):
    intersection=len(np.intersect1d(a,b));union=len(a)+len(b)-intersection
    return dict(mask_intersection=intersection,mask_union=union,mask_jaccard=intersection/union if union else 1.,
        mask_overlap_fraction=intersection/len(a) if len(a) else 1.,mask_symmetric_difference=len(a)+len(b)-2*intersection)
