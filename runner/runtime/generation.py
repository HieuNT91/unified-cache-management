"""Shared request generation and native attention diagnostic validation."""
import time


def generate(engine, tokens, request_id, *, sample=None, phase='read', seed=0, progress=None):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    if engine.has_unfinished_requests():
        raise RuntimeError('Concurrent requests are unsupported')
    started = time.perf_counter()
    from runner.inputs.layout import request_metadata, METADATA_KEY
    from runner.runtime.output_budget import KEY, SAMPLING, OUTPUT_CAP, BudgetState, validate_sample_policy
    from runner.runtime.thinking_loop import KEY as LOOP_KEY, ADATREE_CONFIG
    if phase not in ('read', 'populate') or (phase == 'populate' and sample is not None):
        raise ValueError('Expected an answer sample or independent KV population')
    metadata = request_metadata(request_id, tokens, phase, sample)
    extra = {METADATA_KEY: metadata}
    policy = validate_sample_policy(sample) if phase == 'read' else None
    if policy is not None:
        extra.update({KEY: policy, LOOP_KEY: dict(ADATREE_CONFIG)})
    state = BudgetState(policy) if policy is not None else None
    answer_first = None
    sampling = SAMPLING if policy is not None else dict(
        temperature=0., top_p=1., top_k=32, min_p=0.)
    params = SamplingParams(**sampling, seed=seed, max_tokens=OUTPUT_CAP if policy is not None else 1,
                            extra_args=extra, ignore_eos=policy is not None)
    # Preserve the existing one-token KV-population request on the wire.
    if phase == 'populate':
        params.top_k = 32
    engine.add_request(request_id, TokensPrompt(prompt_token_ids=tokens), params)
    first, result = None, None
    last_progress = None
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.request_id != request_id:
                raise RuntimeError('Unexpected request')
            if output.outputs and output.outputs[0].token_ids and first is None:
                first = time.perf_counter() - started
            if state is not None and output.outputs:
                state.consume(output.outputs[0].token_ids)
                if state.first_answer_index is not None and answer_first is None:
                    answer_first = time.perf_counter() - started
            if progress is not None and output.outputs:
                now = time.perf_counter()
                if last_progress is None or now-last_progress >= 30 or output.finished:
                    progress(dict(request_id=request_id, output_tokens=len(output.outputs[0].token_ids),
                                  elapsed_seconds=now-started, phase=state.phase if state else 'generation',
                                  finished=output.finished))
                    last_progress = now
            if output.finished:
                result = output
    if result is None or first is None:
        raise RuntimeError('Generation did not return a token-bearing result')
    if state is not None:
        if state.phase != 'done':
            raise ValueError('Thinking request finished before answer phase completion')
        result.ruler_thinking_state = state
        result.first_answer_content_seconds = answer_first
    return result, first, time.perf_counter() - started


def verify_diagnostics(workers, sample, method, ratio, tp):
    """Independent CPU replay of each TP rank's global scores and layer sets."""
    import math
    if method == 'adatree-v1':
        raise ValueError('AdaTreeV1 requires validate_routing before native TreeKV validation')
    if method not in ('baseline', 'tree-prophetkv'):
        raise ValueError('Only baseline and native TreeKV diagnostics are supported')
    if sorted(w['rank'] for w in workers) != list(range(tp)):
        raise RuntimeError('Missing tensor-parallel rank')
    common = None
    common_scores = None
    common_f3 = None
    for worker in workers:
        events = worker['diagnostics']
        steps = [e for e in events if e['kind'] == 'prefill_step']
        cursor = 0 if method == 'baseline' else sample['boundaries'][1]
        for step in steps:
            if (step['start'] != cursor or step['end'] - cursor != step['scheduled_tokens']
                    or not 0 < step['scheduled_tokens'] <= 16384
                    or step['end'] > sample['boundaries'][-1]
                    or step['prefill_complete'] != (step['end'] == sample['boundaries'][-1])
                    or step['no_forward'] != (step['recomputed_tokens'] == 0)):
                raise RuntimeError('Invalid prefill progress or scheduling budget')
            cursor = step['end']
        if not steps or cursor != sample['boundaries'][-1]:
            raise RuntimeError('Incomplete prefill progress')
        if method == 'baseline':
            if len(events) != len(steps) or any(e['scheduled_tokens'] != e['recomputed_tokens'] for e in steps):
                raise RuntimeError('Baseline used a sparse method')
            continue
        selections = [e for e in events if e['kind'] == 'prophetkv_selection']
        if len(selections) != 1 or any(e['kind'] == 'cache_miss' for e in events):
            raise RuntimeError('Expected exactly one successful cache selection')
        event = selections[0]
        prefix, end = sample['boundaries'][1], sample['boundaries'][-2]
        scores = event['scores']
        if len(scores) != end - prefix or not all(math.isfinite(s) for s in scores):
            raise RuntimeError('Invalid selection scores')
        if common_scores is not None and scores != common_scores:
            raise RuntimeError('TP ranks disagree on native attention scores')
        common_scores = scores
        from runner.tree.prophetkv import validate_selection
        reference = validate_selection(event, sample, ratio)
        tree_evidence = event['f3_links']
        if common_f3 is not None and tree_evidence != common_f3:
            raise RuntimeError('TP ranks disagree on F3 evidence')
        common_f3 = tree_evidence
        if (event['selected_positions'] != reference or event['scoring_layers'] != list(range(64))
                or event['fusion'] != 'mean_layers_fp32' or event['alignment_count'] != 64):
            raise RuntimeError('Selection replay, mean or cache alignment mismatch')
        if common is not None and common != reference:
            raise RuntimeError('TP ranks disagree on selected positions')
        common = reference
        layers = [e for e in events if e['kind'] == 'layer_counts']
        expected = {f'model.layers.{i}.self_attn.attn' for i in range(64)}
        required = reference + list(range(end, sample['boundaries'][-1]))
        if len(layers) != 64 * len(steps):
            raise RuntimeError('Missing all-layer selected-set verification')
        for step in steps:
            subset = [p for p in required if step['start'] <= p < step['end']]
            entries = [e for e in layers if (e['start'], e['end']) == (step['start'], step['end'])]
            if (len(entries) != 64 or {e['layer'] for e in entries} != expected
                    or step['recomputed_tokens'] != len(subset)):
                raise RuntimeError('Missing layer or incorrect per-step token count')
            for entry in entries:
                if (not entry['selected_set_verified'] or entry['selected_positions'] != subset
                        or any(entry[k] != len(subset) for k in
                               ('projection_tokens', 'attention_tokens', 'ffn_tokens'))):
                    raise RuntimeError('Per-layer selection coverage mismatch or duplicate positions')
