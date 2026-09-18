"""CPU-safe common artifacts, timing and scoring utilities."""
import hashlib
import re
import json
import os
from pathlib import Path
import time
from cacheblend_ruler import prompt_digest,score_prediction

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def extract_answer(prediction):
    cleaned=prediction.replace('*','')
    for pattern in (r'The correct answer is \(([A-D])\)',r'The correct answer is ([A-D])'):
        match=re.search(pattern,cleaned)
        if match:return match.group(1)
    return None

def score(sample, prediction):
    meta = sample['source_metadata']
    if sample['dataset'] == 'ruler':
        return float(score_prediction(sample['label'], prediction, meta['references'])), None
    answer = extract_answer(prediction.strip())
    return float(answer == meta['answer']), answer

def load_sample(path):
    sample = json.loads(Path(path).read_text())
    ids, bounds = sample['token_ids'], sample['boundaries']
    if prompt_digest(ids) != sample['prompt_sha256'] or len(ids) != sample['tokens']:
        raise ValueError('Saved prompt hash/length mismatch')
    if bounds[0] != 0 or bounds[-1] != len(ids) or any(a >= b for a, b in zip(bounds, bounds[1:])):
        raise ValueError('Invalid saved chunk boundaries')
    if bounds[-1] - bounds[-2] != sample['fresh_suffix_tokens'] or any(x % 64 for x in bounds[:-1]):
        raise ValueError('Expected aligned context chunks and declared fresh suffix')
    end_id = ids[bounds[1] - 1]
    # The native connector recognizes markers at block ends. Check that this
    # exactly matches the saved boundaries, allowing adjacent padding blocks.
    for start, end in zip(bounds[:-2], bounds[1:-1]):
        if ids[end - 1] != end_id:
            raise ValueError('Chunk terminators differ')
        marker_started = False
        for p in range(start + 63, end, 64):
            if ids[p] == end_id:
                marker_started = True
            elif marker_started:
                raise ValueError('Internal chunk marker splits saved content')
    if any(ids[p] == end_id for p in range(bounds[-2]+63, bounds[-1], 64)):
        raise ValueError('Suffix contains a connector boundary')
    return sample

def cache_snapshot(path, content=False):
    rows = {}
    for f in sorted((Path(path)/'kv').glob('*/*')):
        if f.is_file():
            st = f.stat()
            rows[str(f.relative_to(path))] = [st.st_size, st.st_mtime_ns, st.st_ctime_ns]
            if content:
                rows[str(f.relative_to(path))].append(sha(f))
    return rows

def generate(engine, tokens, budget, request_id, before_submit=None):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    if engine.has_unfinished_requests():
        raise RuntimeError('Engine must be idle before measurement')
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=budget,
                              output_kind=RequestOutputKind.CUMULATIVE)
    start = time.perf_counter()
    if before_submit is not None:
        before_submit()
    engine.add_request(request_id, TokensPrompt(prompt_token_ids=tokens), sampling)
    first = result = None
    last_progress = start
    while engine.has_unfinished_requests():
        outputs = engine.step()
        now = time.perf_counter()
        for output in outputs:
            if output.request_id != request_id:
                raise RuntimeError('Unexpected concurrent request')
            if output.outputs and output.outputs[0].token_ids and first is None:
                first = now - start
            if output.finished:
                result = output
        if now - last_progress > 30:
            print(f'engine_progress request={request_id} elapsed={now-start:.1f}', flush=True)
            last_progress = now
    elapsed = time.perf_counter() - start
    if result is None or first is None:
        raise RuntimeError('Missing token-bearing first/final engine output')
    return result, first, elapsed
