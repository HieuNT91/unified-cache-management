"""Freeze exact existing prompts and explicit oracle-only sidecars."""
import hashlib,shutil,sys,time,re
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1]
D=R/'.results/oracle-highbudget-prophetkv32b-20260925'
OLD=R/'.results/querykey-prophetkv32b-mk2-mk3-cwe-5samples-output256-20260924'
COHORT=R/'.results/qwen3-32b-prophetkv-yarn2-ruler10x50-lb50-20260922'
VANILLA=R/'.results/vanilla-prophetkv32b-40samples-output256-20260924'
STUDY=R/'.analysis/qwen3-32b-layer-search-40samples-20260924'
TRANSITION=R/'.analysis/oracle-highbudget-transition-20260925'
sys.path.insert(0,str(H/'runtime'))
from common import load,dump
from query_config import *

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def annotate(tok,m,sample,row):
    content=row['input'];text=tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+row.get('answer_prefix','')
    enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];offsets=enc['offset_mapping'];mapping=[];start=0;b=m['boundaries']
    for i,base in enumerate(b[:-2]):
        marker=tok.encode(f'\n[Context chunk {i+1}]\n',add_special_tokens=False)
        n=min(4095-len(marker),len(ids)-m['fresh_suffix_tokens']-start)
        assert sample['token_ids'][base+len(marker):base+len(marker)+n]==ids[start:start+n]
        mapping.extend(range(base+len(marker),base+len(marker)+n));start+=n
    mapping.extend(range(b[-2],b[-1]));assert len(mapping)==len(ids)
    assert all(sample['token_ids'][absolute]==token for absolute,token in zip(mapping,ids))
    q=m['query']['text'];qstart=text.rfind(q);assert qstart>=0
    qpos=[mapping[i] for i,(a,z) in enumerate(offsets) if z>qstart and a<qstart+len(q)]
    assert qpos==m['query']['positions']
    key=re.search(r'\bfor (.*?) mentioned\b',q)[1]
    matches=[x for x in re.finditer(r'One of the special magic (?:numbers?|uuids?) for (.*?) is: ([^.]+)\.',content) if x[1]==key]
    assert matches and {x[2] for x in matches}==set(sample['source_metadata']['references'])
    base=text.index(content);spans=[]
    for match in matches:
        for group,kind in [(1,'key'),(2,'value')]:
            a,z=match.span(group);a+=base;z+=base
            indices=[i for i,(x,y) in enumerate(offsets) if y>a and x<z]
            spans.append(dict(kind=kind,text=match[group],formatted_char_start=a,formatted_char_end=z,
                token_char_offsets=[list(offsets[i]) for i in indices],positions=[mapping[i] for i in indices],token_ids=[ids[i] for i in indices]))
    all_positions=sorted({i for s in spans for i in s['positions']});eligible=[i for i in all_positions if b[1]<=i<b[-2]]
    prefix=[i for i in all_positions if i<b[1]];suffix=[i for i in all_positions if i>=b[-2]]
    assert not suffix,'Target statement unexpectedly in suffix'
    # Exact prefix is already computed with its full causal context. Record any such targets explicitly.
    return dict(spans=spans,all_positions=all_positions,eligible_positions=eligible,already_exact_prefix_positions=prefix,
        fresh_suffix_positions=suffix,input_sha256=m['input_sha256'],uses_reference_information=True,mapping_verified=True)

def prepare():
    from transformers import AutoTokenizer
    assert load(TRANSITION/'cancellation.json')['complete'];assert not D.exists();D.mkdir()
    source=load(COHORT/'protocol.json');base=load(VANILLA/'protocol.json');tok=AutoTokenizer.from_pretrained(base['model'],local_files_only=True)
    prior={m['id']:m for m in load(VANILLA/'protocol.json')['samples']};sources={}
    for stage in STAGES:
        root=D/stage;root.mkdir();(root/'private_ucm').symlink_to(H/'private_ucm',target_is_directory=True)
        samples=[]
        for index in range(COUNTS[stage]):
            for task in TASKS[stage]:
                m=next(m for m in source['samples'] if m['label']==task and m['source_row']==index)
                original=Path(m['input_path']);target=root/'inputs'/original.name;target.parent.mkdir(exist_ok=True);shutil.copy2(original,target)
                m=m|dict(input_path=str(target),input_sha256=sha(original),original_input_path=str(original))
                assert sha(target)==m['input_sha256'];sample=load(target)
                assert sample['tokens']+256<=65920 and sample['fresh_suffix_tokens']==256
                if m['id'] in prior:
                    assert m['input_sha256']==prior[m['id']]['input_sha256']
                    m['prior_capture_stem']=prior[m['id']]['prior_capture_stem']
                if stage=='oracle':
                    if m['source_path'] not in sources:sources[m['source_path']]=[__import__('json').loads(x) for x in Path(m['source_path']).read_text().splitlines()]
                    m['oracle']=annotate(tok,m,sample,sources[m['source_path']][m['source_row']])
                    dump(root/'oracle-spans'/(m['id']+'.json'),m['oracle'])
                samples.append(m)
        p=__import__('copy').deepcopy(base)
        for k in ['comparison_root','scoring_layers']:p.pop(k,None)
        cases=ORACLE_CASES if stage=='oracle' else HIGH_CASES
        p.update(study='Oracle target recomputation' if stage=='oracle' else 'High-budget original versus selected-layer ProphetKV',
            stage=stage,samples=samples,cases=list(cases),scope=[dict(length=65536,task=t) for t in TASKS[stage]],
            samples_per_task_length=COUNTS[stage],measured_requests=TARGETS[stage],
            selection='Target-only, then union(target key/value, native top floor(budget*eligible))' if stage=='oracle' else 'Native full-question scoring with all64 mean or selected5 sum',
            fusion='all64 ascending FP32 sum/64 or selected5 ascending FP32 sum; unchanged TP sum/4',
            query_scope='full_question',selected_layers=list(LAYERS),
            oracle_reference_information=stage=='oracle',budget_policy='union; oracle-only once per prompt' if stage=='oracle' else 'native exact floor',
            baseline_policy='No additional baseline inference; only requested methods are measured',created_at=time.time(),
            same_selection_cohort='First five rows/task overlap the layer-selection cohort; all rows0–49 remain descriptive')
        p['settings'].update(root=str(root),cache='/tmp/ucm-oracle-highbudget-prophetkv32b-20260925',samples=COUNTS[stage],scope=p['scope'])
        p['memory_adaptation']['applies_to']='all requested configurations'
        dump(root/'protocol.json',p)
        dump(root/'schedule.json',[m|dict(case=case,output=str(root/'records'/m['id']/(case+'.json'))) for case in cases for m in samples])
        dump(root/'progress.json',dict(state='prepared' if stage=='oracle' else 'queued_after_oracle',validated=0,target=TARGETS[stage]))
        print(stage,len(samples),'prompts',TARGETS[stage],'measurements','length range',min(m['tokens'] for m in samples),max(m['tokens'] for m in samples),flush=True)
        if stage=='oracle':print('Oracle prefix targets',[(m['id'],m['oracle']['already_exact_prefix_positions']) for m in samples if m['oracle']['already_exact_prefix_positions']],flush=True)
    dump(D/'pipeline.json',dict(stages=list(STAGES),targets=TARGETS,total=1230,order='oracle complete and final report validated, then highbudget',transition=str(TRANSITION/'cancellation.json')))

def preserve():
    pins=load(OLD/'preserved-prior.json')
    for name,h in pins.items():assert sha(name)==h,name
    for root in [OLD,R/'benchmarks/querykey_prophetkv32b',TRANSITION]:
        for f in root.rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.lock','.pyc'):pins[str(f)]=sha(f)
    for stage in STAGES:
        for m in load(D/stage/'protocol.json')['samples']:
            for name in [m['original_input_path'],m['source_path']]:pins[name]=sha(name)
    pins[str(COHORT/'protocol.json')]=sha(COHORT/'protocol.json')
    dump(D/'preserved-prior.json',pins);print('Pinned prior artifacts',len(pins),flush=True)

def pin():
    files=[f for f in H.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc']
    files += [D/'pipeline.json',D/'preserved-prior.json',D/'cpu-validation.json',D/'cpu-report-validation.json']
    for stage in STAGES:
        files += [D/stage/'protocol.json',D/stage/'schedule.json']
        files += [f for name in ['inputs','oracle-spans'] for f in (D/stage/name).glob('*')]
    dump(D/'implementation.json',{str(f):sha(f) for f in files})
if __name__=='__main__':{'prepare':prepare,'preserve':preserve,'pin':pin}[sys.argv[1]]()
