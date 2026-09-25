"""Inference span extraction uses only the final question, never answers/evidence."""
import re

def extract(task,question):
    if task=='cwe':
        matches=list(re.finditer(re.escape('10 most common words'),question))
        if len(matches)!=1:raise ValueError('Ambiguous CWE task phrase')
        m=matches[0];return m.span(),m.group(),'task_phrase'
    if task not in ('niah_multikey_2','niah_multikey_3'):raise ValueError(task)
    matches=list(re.finditer(r'\bfor (.*?) mentioned\b',question))
    if len(matches)!=1:raise ValueError('Ambiguous query key')
    m=matches[0];key=m.group(1)
    if not key or key!=key.strip():raise ValueError('Empty/ambiguous key')
    if task=='niah_multikey_3' and not re.fullmatch(r'[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}',key):
        raise ValueError('Incomplete queried UUID')
    return m.span(1),key,'entity_key'

def map_span(tokenizer,m,sample,row):
    content=row['input']
    text=tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)+row.get('answer_prefix','')
    q=m['query']['text'];qstart=text.rfind(q)
    if qstart<0:raise ValueError('Final question absent from original formatted prompt')
    (a,z),span,kind=extract(m['label'],q);a+=qstart;z+=qstart
    enc=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=enc['input_ids'];offsets=enc['offset_mapping'];mapping=[];start=0;b=m['boundaries']
    for i,base in enumerate(b[:-2]):
        marker=tokenizer.encode(f'\n[Context chunk {i+1}]\n',add_special_tokens=False)
        n=min(4095-len(marker),len(ids)-m['fresh_suffix_tokens']-start)
        assert sample['token_ids'][base+len(marker):base+len(marker)+n]==ids[start:start+n]
        mapping.extend(range(base+len(marker),base+len(marker)+n));start+=n
    mapping.extend(range(b[-2],b[-1]))
    assert len(mapping)==len(ids)
    assert all(sample['token_ids'][absolute]==token for absolute,token in zip(mapping,ids))
    qpositions=[mapping[i] for i,(x,y) in enumerate(offsets) if y>qstart and x<qstart+len(q)]
    assert qpositions==m['query']['positions'],'Final question mapping differs'
    indices=[i for i,(x,y) in enumerate(offsets) if y>a and x<z]
    absolute=[mapping[i] for i in indices]
    assert absolute and set(absolute)<=set(qpositions)
    assert min(absolute)>=b[-2] and max(absolute)<b[-1]
    return dict(span_text=span,kind=kind,question=q,formatted_char_start=a,formatted_char_end=z,
        question_char_start=qstart,question_char_end=qstart+len(q),
        token_char_offsets=[list(offsets[i]) for i in indices],original_token_positions=indices,
        token_ids=[ids[i] for i in indices],absolute_positions=absolute,
        mapping_verified=True,input_sha256=m['input_sha256'])
