"""Fresh datasets, full LongBench census, immutable prompts and source hashes."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from common import HERE, REPO, CASES, PARAMETERS, ENGINE_POLICY, TIMING, dump, load
from cacheblend_ruler import cacheblend_prompt
from prophetkv_common import load_sample


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tree_hashes(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def longbench_content(row):
    template = (REPO / 'benchmarks/vendor/LongBench-v2/prompts/0shot.txt').read_text().replace('\\n', '\n')
    before, separator, after = template.partition('$DOC$')
    if not separator:
        raise ValueError('Missing LongBench document marker')
    markers = {'$Q$': 'question', **{f'$C_{c}$': f'choice_{c}' for c in 'ABCD'}}
    tail = re.sub(r'\$(?:Q|C_[ABCD])\$', lambda match: row[markers[match[0]]], after)
    content = before + row['context'] + tail
    begin = len(before) + len(row['context']) + tail.index('What is the correct answer to this question:')
    end = len(before) + len(row['context']) + tail.rindex('\n\nFormat your response')
    return content, begin, end


def encode_longbench(tokenizer, row, ordinal, thinking=True, limit=65536):
    if not thinking:
        raise ValueError('This study is thinking only')
    content, begin, end = longbench_content(row)
    text = tokenizer.apply_chat_template([dict(role='user', content=content)], tokenize=False,
        add_generation_prompt=True, enable_thinking=thinking)
    # A bounded eligibility probe only. Accepted model inputs are encoded in full
    # below, never truncated. Exact excluded lengths are unnecessary for selection.
    probe = tokenizer(text, add_special_tokens=False, truncation=True, max_length=limit+1)['input_ids']
    if len(probe) > limit:
        return None, dict(eligible=False, reason='chat_prompt_over_limit', tokens_lower_bound=limit+1)
    offset = text.index(content)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    query = [i for i, (a,b) in enumerate(encoded['offset_mapping']) if b > offset+begin and a < offset+end]
    if not query:
        raise ValueError('Missing LongBench query')
    suffix = max(256, len(ids)-query[0])
    chunks, tokens = cacheblend_prompt(ids, tokenizer, tokenizer.pad_token_id, 4095, suffix)
    if len(tokens) > limit:
        return None, dict(eligible=False, reason='formatted_prompt_over_limit', tokens=len(tokens))
    if len(chunks) < 2:
        # Never silently drop an eligible sample because the method cannot run it.
        raise ValueError(f'Eligible LongBench row {ordinal} needs support for fewer than two chunks')
    bounds = [0]
    for chunk in chunks:
        bounds.append(bounds[-1]+len(chunk))
    bounds.append(len(tokens))
    assert tokens[-suffix:] == ids[-suffix:]
    mode = 'thinking' if thinking else 'non_thinking'
    sample = dict(id=f'longbench_v2-{ordinal:04d}-{mode}', dataset='longbench_v2',
        label='longbench_v2_'+mode, context_target=65536, chunk_size=4096, source_row=ordinal,
        source_index=row['_id'], thinking_enabled=thinking, max_output_tokens=16384,
        original_tokens=len(ids), tokens=len(tokens), token_ids=tokens, boundaries=bounds,
        fresh_suffix_tokens=suffix, query=dict(text=content[begin:end],
            positions=[i+len(tokens)-len(ids) for i in query]),
        source_metadata=dict(references=[row['answer']], answer=row['answer'], domain=row['domain'],
            sub_domain=row.get('sub_domain'), difficulty=row.get('difficulty'), length=row.get('length')))
    return sample, dict(eligible=True, tokens=len(tokens))


def longbench_census(tokenizer, path, sink):
    rows = load(path)
    if not isinstance(rows, list) or len(rows) != 503:
        raise ValueError('LONGBENCH_DATA must be the full 503-row data.json array')
    ids = [row['_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate LongBench source IDs')
    census = []
    eligible_ordinal = 0
    for ordinal, row in enumerate(rows):
        sample, eligibility = encode_longbench(tokenizer, row, ordinal, True)
        census.append(dict(source_row=ordinal, source_id=row['_id'], thinking_enabled=True, **eligibility))
        if sample is not None:
            sample['assigned_job'] = eligible_ordinal % 2
            sink(sample)
            eligible_ordinal += 1
        if ordinal % 20 == 0:
            print(f'LongBench census {ordinal+1}/{len(rows)}', flush=True)
    return dict(source_sha256=sha(path), source_rows=len(rows), input_limit=65536,
        policy='All eligible thinking rows after chat and chunk formatting; no inference truncation',
        counts={'thinking': sum(c['eligible'] for c in census)}, census=census)


def prepare_experiment(cfg, root):
    from suite import preflight, verify, copy_private_ucm, persistent_sources, progress
    if (root/'protocol.json').exists():
        return verify(cfg, root)
    runtime, ucm, _, devices, tokenizer = preflight(cfg)
    if (root/'preparation.json').exists() and load(root/'preparation.json') != cfg:
        raise ValueError('Partially prepared output has different settings; choose a new RESULT_ROOT')
    dump(root/'preparation.json', cfg)
    samples, datasets = [], []
    job = int(cfg['job_id'])

    def save(sample, source):
        if sample.get('assigned_job', sample['source_row'] % 2) != job:
            return
        sample['source_path'] = str(source)
        path = root/'samples'/(sample['id']+'.json')
        dump(path, sample)
        load_sample(path)
        samples.append({k:v for k,v in sample.items() if k!='token_ids'} |
                       dict(input_path=str(path), input_sha256=sha(path)))

    source = Path(cfg['longbench'])
    target = root/'datasets/longbench_v2/data.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    datasets.append(str(target))
    census = longbench_census(tokenizer, target, lambda sample: save(sample, target))
    dump(root/'longbench-census.json', census)
    private = root/'private_ucm/ucm'
    copy_private_ucm(ucm, private)
    for name in ('blend.py','selection.py'):
        shutil.copy2(REPO/'ucm/sparse/blend'/name, private/'sparse/blend'/name)
    method = private/'sparse/prophetkv'
    if method.exists():
        shutil.rmtree(method)
    shutil.copytree(HERE/'method/prophetkv', method, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for path, text in persistent_sources(private).items():
        path.write_text(text)
    with (private/'sparse/factory.py').open('a') as stream:
        stream.write('\nUcmSparseFactory.register_sparse_method("ProphetKV", "ucm.sparse.prophetkv.prophetkv", "ProphetKV")\n')
    shutil.copytree(HERE, root/'source', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    from common import rope_for_length
    protocol = dict(schema_version=2, study='Remote Qwen3-32B Original ProphetKV', settings=cfg,
        model=cfg['model'], runtime_versions=runtime, gpu_devices=devices, tensor_parallel_size=2,
        num_layers=64, gpu_memory_utilization=cfg['memory'], dtype='bfloat16', eager=True,
        cases=list(CASES), case_parameters=PARAMETERS, scope=cfg['scope'],
        samples=samples, datasets=datasets, measured_requests=len(samples)*len(CASES),
        input_limit=65536, max_model_len=81920, rope_scaling_64k=rope_for_length(65536),
        rope_extension='16384 appended positions with unchanged YaRN2 frequencies; first 65536 bitwise preserved',
        output_tokens=dict(thinking=16384), thinking_enabled=True,
        decoding=dict(thinking=dict(temperature=.6,top_p=.95,top_k=20,seed=0)),
        scoring="Only final text after closing </think>; unfinished reasoning scores zero",
        longbench_counts=census['counts'],
        assignment='LongBench eligible source ordinal modulo 2; same source/GPU pair across methods',
        methods_order=list(CASES) if job==0 else list(CASES[2:]+CASES[:2]),
        qualification='No separate qualification/smoke; normal warmup/readiness/measured validation retained',
        engine_lifetime=ENGINE_POLICY, timing_source=TIMING, created_at=time.time(),
        sources_sha256=tree_hashes(HERE), private_ucm_sha256=tree_hashes(root/'private_ucm'),
        datasets_sha256={str(p):sha(p) for p in datasets},
        tokenizer_sha256={p.name:sha(p) for p in Path(cfg['model']).iterdir()
                          if p.is_file() and p.suffix in ('.json','.jinja')})
    dump(root/'prompt_manifest.json', samples)
    dump(root/'protocol.json', protocol)
    progress(root, protocol, 'prepared')
    print(f'PREPARED {len(samples)} inputs / {protocol["measured_requests"]} measurements; LongBench census {census["counts"]}', flush=True)
    return protocol


def verify_frozen(root, protocol):
    if tree_hashes(HERE) != protocol['sources_sha256']:
        raise ValueError('Prepared runner source changed; use a new result directory')
    if tree_hashes(root/'private_ucm') != protocol['private_ucm_sha256']:
        raise ValueError('Private UCM changed')
    for sample in protocol['samples']:
        if sha(sample['input_path']) != sample['input_sha256']:
            raise ValueError('Frozen input changed')
    for path, digest in protocol['datasets_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Frozen dataset changed')
    for name, digest in protocol['tokenizer_sha256'].items():
        if sha(Path(protocol['model'])/name) != digest:
            raise ValueError('Model configuration/tokenizer changed')
