"""Freeze a seven-method expansion with immutable retained measurements."""
import hashlib
import os
from pathlib import Path
import shutil
import time

from common import HERE, REPO, CASES, dump, load, identity, rope_for_length, model_limit

ORIGINAL = REPO / '.results/qwen3-32b-prophetkv-vt-cwe10-64k-c4096-tp4-20260921'
EXTENSION = REPO / '.results/qwen3-32b-prophetkv-vt-cwe10-64k-c4096-tp4-extension-20260921'
PREVIOUS_SOURCE = REPO / 'benchmarks/prophetkv32b_tp4_yarn2'
PARENT = REPO / '.results/qwen3-32b-prophetkv-vt-cwe10-64k-c4096-tp4-yarn2-20260921'
NIAH = REPO / '.results/qwen3-32b-prophetkv-niahqa8x10-yarn2-no-smoke-20260921'
SMALL = Path('/home/thnguyen/unified-cache-management/.results/prophetkv-expanded5-ruler13x100-longbenchv2-20260918')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_hashes(hashes, base=Path('/')):
    for name, expected in hashes.items():
        if sha256(base / name) != expected:
            raise ValueError(f'Frozen artifact changed: {base / name}')


def verify_unchanged_inference():
    names = ['worker.py', 'reference.py', 'persistent_worker.py', 'cacheblend_ruler.py',
             'prophetkv_common.py', 'lifecycle.py', 'persistent_connector.py', 'memory_worker.py', 'rope_window.py']
    names += [str(p.relative_to(HERE)) for p in (HERE / 'method').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for name in names:
        if (HERE / name).read_bytes() != (PREVIOUS_SOURCE / name).read_bytes():
            raise ValueError(f'Inference implementation changed: {name}')
    return names


def verify_frozen(root):
    amendment = load(root / 'experiment.json')
    check_hashes(amendment['preserved_sha256'])
    check_hashes(load(root / 'provenance.json')['sha256'], root)
    for path in (root / 'source').rglob('*'):
        if path.is_file() and path.read_bytes() != (HERE / path.relative_to(root / 'source')).read_bytes():
            raise ValueError(f'Executed source differs from snapshot: {path}')
    p = load(root / 'protocol.json')
    if p['rope_scaling_64k'] != rope_for_length(65536) or p['rope_scaling_64k']['factor'] != 2:
        raise ValueError('Wrong RoPE factor')
    if os.environ.get('VLLM_ALLOW_LONG_MAX_MODEL_LEN') != '1':
        raise ValueError('Exact prompt preservation requires the documented length override')


def longbench_samples(tokenizer, cfg, root, small):
    import json
    import random
    import re
    from collections import defaultdict, deque
    from cacheblend_ruler import cacheblend_prompt
    candidates = [m for m in small['samples'] if m['dataset'] == 'longbench_v2']
    source = Path(candidates[0]['source_path'])
    template_path = source.parent / '0shot.txt'
    source_digest = sha256(source)
    if any(m['source_metadata']['source_sha256'] != source_digest for m in candidates):
        raise ValueError('LongBench source changed')
    directory = root / 'datasets/longbench_v2'
    directory.mkdir(parents=True, exist_ok=True)
    for path in (source, template_path):
        shutil.copyfile(path, directory / path.name)
    rows = json.loads(source.read_text())
    template = template_path.read_text().replace('\\n', '\n')
    groups = defaultdict(list)
    for meta in candidates:
        groups[meta['source_metadata']['domain']].append(meta)
    rng = random.Random(20260922)
    for domain in sorted(groups):
        groups[domain].sort(key=lambda m: m['source_row'])
        rng.shuffle(groups[domain])
        groups[domain] = deque(groups[domain])
    queue = []
    while any(groups.values()):
        for domain in sorted(groups):
            if groups[domain]:
                queue.append(groups[domain].popleft())
    selected, census = [], []
    for meta in queue:
        ordinal = meta['source_row']
        row = rows[ordinal]
        before, sep, after = template.partition('$DOC$')
        if not sep:
            raise ValueError('Missing LongBench document placeholder')
        markers = {'$Q$': 'question', **{f'$C_{c}$': f'choice_{c}' for c in 'ABCD'}}
        tail = re.sub(r'\$(?:Q|C_[ABCD])\$', lambda m: row[markers[m[0]]], after)
        content = before + row['context'] + tail
        begin = len(before) + len(row['context']) + tail.index('What is the correct answer to this question:')
        end = len(before) + len(row['context']) + tail.rindex('\n\nFormat your response')
        text = tokenizer.apply_chat_template([dict(role='user', content=content)], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        offset = text.index(content)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded['input_ids']
        query = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > offset+begin and a < offset+end]
        assert query
        suffix = max(256, len(ids)-query[0])
        chunks, tokens = cacheblend_prompt(ids, tokenizer, tokenizer.pad_token_id, cfg['chunk']-1, suffix)
        eligible = len(tokens) < 65536 and len(chunks) >= 2
        census.append(dict(source_row=ordinal, source_index=row['_id'], domain=row['domain'],
            original_tokens=len(ids), formatted_tokens=len(tokens), selected=eligible))
        if not eligible:
            continue
        boundaries = [0]
        for chunk in chunks:
            boundaries.append(boundaries[-1]+len(chunk))
        boundaries.append(len(tokens))
        positions = [i+len(tokens)-len(ids) for i in query]
        assert ids[-suffix:] == tokens[-suffix:] and positions[0] >= boundaries[-2]
        selected.append(dict(id=f'longbench-v2-{ordinal:04d}', dataset='longbench_v2', label='longbench_v2',
            context_target=65536, chunk_size=4096, source_row=ordinal, source_index=row['_id'],
            source_path=str(directory/'data.json'), source_metadata=dict(answer=row['answer'], references=[row['answer']],
                domain=row['domain'], sub_domain=row['sub_domain'], difficulty=row['difficulty'],
                length_category=row['length'], source_sha256=source_digest),
            original_tokens=len(ids), tokens=len(tokens), token_ids=tokens, boundaries=boundaries,
            fresh_suffix_tokens=suffix, thinking_enabled=False,
            query=dict(text=content[begin:end], positions=positions)))
        if len(selected) == 50:
            break
    if len(selected) != 50:
        raise ValueError('Insufficient eligible LongBench v2 prompts')
    dump(root/'longbench-selection.json', dict(seed=20260922, samples=50,
        policy='Round-robin domains after seeded per-domain shuffle of the preserved 184-row under-64K cohort',
        limit='Strictly fewer than 65536 fully formatted tokens, including chat/markers/padding',
        minimum='At least two context chunks, required by the existing independent-chunk reuse protocol',
        truncated=False, census=census))
    return selected, {str(path): sha256(path) for path in (source, template_path)}, [str(directory/'data.json'), str(directory/'0shot.txt')]


def prepare_experiment(cfg, root):
    import json
    from suite import check_orphan, progress, verify, preflight, encode, validate
    from prophetkv_common import load_sample
    if (root/'protocol.json').exists():
        verify_frozen(root)
        return verify(cfg, root)
    parents = (NIAH, PARENT)
    for parent in parents:
        check_orphan(parent)
        for name in ('supervisor.json', 'reporter.json'):
            state = load(parent/name)
            if identity(state['pid']) == state['identity']:
                raise RuntimeError(f'Parent process still alive: {parent/name}')
        if not load(parent/'final/integrity-validation.json')['complete']:
            raise ValueError('Parent experiment is incomplete')
    runtime, _, _, devices, tokenizer = preflight(cfg)
    unchanged = verify_unchanged_inference()
    preserved = {}
    for parent in parents:
        preserved.update(load(parent/'experiment.json')['preserved_sha256'])
        preserved.update({str(parent/name): digest for name, digest in load(parent/'provenance.json')['sha256'].items()})
        for folder in ('final',):
            for path in (parent/folder).rglob('*'):
                if path.is_file():
                    preserved[str(path)] = sha256(path)
    previous = load(NIAH/'protocol.json')
    for key in ('model', 'chunk', 'indices', 'memory'):
        if cfg[key] != previous['settings'][key]:
            raise ValueError(f'Unrequested inference setting change: {key}')
    small = load(SMALL/'protocol.json')
    preserved[str(SMALL/'protocol.json')] = sha256(SMALL/'protocol.json')
    samples, datasets, retained, source_hashes = [], [], {}, {}
    for unit in cfg['scope']:
        task = unit['task']
        if task == 'longbench_v2':
            continue
        parent = PARENT if task in ('vt', 'cwe') else NIAH
        selected = sorted([m for m in small['samples'] if m['dataset']=='ruler' and m['label']==task and m['source_row']<50], key=lambda m:m['source_row'])
        if [m['source_row'] for m in selected] != list(range(50)):
            raise ValueError(f'Expected 50 distinct source rows for {task}')
        source = Path(selected[0]['source_path'])
        digest = sha256(source)
        if any(m['source_path']!=str(source) or m['source_metadata']['source_sha256']!=digest for m in selected):
            raise ValueError('RULER dataset identity mismatch')
        source_hashes[str(source)] = digest
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        target = root/'datasets/65536'/task/'validation.jsonl'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        datasets.append(str(target))
        for ordinal, meta in enumerate(selected):
            row = rows[ordinal]
            assert row['outputs'] == meta['source_metadata']['references']
            sid = f'{task}-65536-{ordinal:04d}'
            path = root/'samples'/(sid+'.json')
            if ordinal < 10:
                original = parent/'samples'/(sid+'.json')
                sample = load_sample(original)
                assert sample['source_row']==ordinal and sample['source_metadata']['references']==row['outputs']
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, path)
                for case in previous['cases']:
                    record = parent/'records'/sid/(case+'.json')
                    retained[sid+'/'+case] = dict(root=str(parent), record=str(record))
                    for suffix in ('.json', '.log', '.diagnostics.json', '.validated.json'):
                        artifact = record.with_suffix(suffix)
                        preserved[str(artifact)] = sha256(artifact)
                        link = root/'records'/sid/artifact.name
                        link.parent.mkdir(parents=True, exist_ok=True)
                        link.symlink_to(artifact)
                    session = Path(load(record)['session_path'])
                    if str(session) not in preserved:
                        preserved[str(session)] = sha256(session)
            else:
                sample = encode(tokenizer, row, cfg, 65536, ordinal, task)
                sample['source_path'] = str(target)
                dump(path, sample)
            load_sample(path)
            samples.append({k:v for k,v in sample.items() if k!='token_ids'} | dict(input_path=str(path)))
        print(f'Prepared {task}: 10 original prompts/60 retained measurements plus 40 new prompts', flush=True)
    lb, lb_hashes, lb_paths = longbench_samples(tokenizer, cfg, root, small)
    source_hashes.update(lb_hashes)
    datasets.extend(lb_paths)
    for sample in lb:
        path = root/'samples'/(sample['id']+'.json')
        dump(path, sample)
        load_sample(path)
        samples.append({k:v for k,v in sample.items() if k!='token_ids'} | dict(input_path=str(path)))
    preserved.update(source_hashes)
    assert len(samples)==550 and len(retained)==600 and len(CASES)==7
    shutil.copytree(NIAH/'private_ucm', root/'private_ucm', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    protocol = previous | dict(study='Qwen3-32B YaRN 2x RULER 50/task and LongBench v2 50 total',
        settings=cfg, scope=cfg['scope'], cases=list(CASES), samples=samples, datasets=datasets,
        runtime_versions=runtime, gpu_devices=devices, samples_per_task_length=50,
        measured_requests=3850, retained_requests=600, new_requests=3250, retained_records=retained,
        qualification='Disabled by explicit user request; measured sessions only',
        longbench_selection=str(root/'longbench-selection.json'), created_at=time.time(),
        cohort_note='First ten RULER prompts/methods except 60% retain earlier timings; later cohorts run at different times')
    model_limit(protocol, samples[0])
    for path in (root/'records').glob('*/*.json'):
        if path.stem in previous['cases']:
            sid = path.parent.name
            validate(root, protocol, next(m for m in samples if m['id']==sid), path.stem, path)
    dump(root/'protocol.json', protocol)
    dump(root/'prompt_manifest.json', samples)
    dump(root/'preparation.json', cfg)
    dump(root/'experiment.json', dict(preserved_sha256=preserved, parent_roots=[str(p) for p in parents],
        new_requests=3250, retained_requests=600, total_requests=3850, unchanged_inference_sources=unchanged,
        source_dataset_sha256=source_hashes, changed_scope='50/task; add 60%; 50 LongBench v2 prompts <64K', created_at=time.time()))
    shutil.copytree(HERE, root/'source', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    hashes = {}
    for folder in ('source','samples','datasets','private_ucm'):
        for path in (root/folder).rglob('*'):
            if path.is_file():
                hashes[str(path.relative_to(root))] = sha256(path)
    for name in ('protocol.json','prompt_manifest.json','experiment.json','longbench-selection.json'):
        hashes[name] = sha256(root/name)
    dump(root/'provenance.json', dict(sha256=hashes, frozen_at=time.time()))
    verify(cfg, root)
    verify_frozen(root)
    progress(root, protocol, 'prepared')
    print(f'Prepared 3850 total / 600 retained / 3250 new; token range {min(m["tokens"] for m in samples)}–{max(m["tokens"] for m in samples)}', flush=True)
    return protocol

