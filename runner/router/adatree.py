"""Fixed AdaTreeV1 routing receipts and native answer validation."""

SCHEMA = 'adatree-v1-native-tailspread-v1'
METHOD = 'adatree-v1'


def answer_details(result, sample, tokenizer, ttft_seconds):
    """Preserve the prepared thinking policy's answer channel and budget audit."""
    from runner.reporting.aggregate import OutputAnalyzer
    from runner.runtime.thinking_budget import KEY, validate_thinking_outcome
    from runner.runtime.output_budget import BUDGETS as TOTAL_BUDGETS
    state = getattr(result, 'ruler_thinking_state', None)
    if (KEY in sample) != (state is not None):
        raise RuntimeError('AdaTreeV1 answer is missing its prepared thinking-policy state')
    tokens = list(result.outputs[0].token_ids)
    metrics = OutputAnalyzer(tokenizer).analyze(tokens, [] if state is not None else sample['token_ids'], sample['thinking'])
    capped = len(tokens) >= sample['max_output_tokens']
    timing = {}
    if state is not None:
        if state.policy != sample[KEY] or state.phase != 'done':
            raise RuntimeError('AdaTreeV1 thinking-policy state is incomplete or changed')
        metrics.update(state.metrics(tokenizer))
        timing['first_answer_content_seconds'] = result.first_answer_content_seconds
        capped = (metrics['answer_cap_reached'] if sample[KEY]['protocol'] in TOTAL_BUDGETS
                  else metrics['thinking_cap_reached'] or metrics['answer_cap_reached'])
        validate_thinking_outcome(dict(metrics, output_cap_reached=capped,
            ttft_seconds=ttft_seconds, **timing), sample[KEY])
    return dict(metrics=metrics, output_cap_reached=capped, timings=timing)


def validate_routing(workers, sample, request_id, tp, chunk_size=4096):
    """Check all ranks and replay the selected native TreeKV action."""
    from runner.runtime.generation import verify_diagnostics
    from ucm.sparse.prophetkv.adatree import route_features, feature_names, protocol_for
    names, schema = feature_names(chunk_size), protocol_for(chunk_size)
    if sorted(worker.get('rank', -1) for worker in workers) != list(range(tp)):
        raise RuntimeError('Missing AdaTreeV1 tensor-parallel rank')
    common = None
    for worker in workers:
        events = [event for event in worker['diagnostics'] if event['kind'] == 'adatree_v1']
        selections = [event for event in worker['diagnostics'] if event['kind'] == 'prophetkv_selection']
        if len(events) != 1 or len(selections) != 1:
            raise RuntimeError('AdaTreeV1 requires exactly one integrated routing event and native probe')
        event = events[0]
        features = event.get('features')
        if not isinstance(features, dict) or set(features) != set(names):
            raise RuntimeError('Invalid AdaTreeV1 feature receipt')
        try:
            action = route_features(**features, chunk_size=chunk_size)
        except (TypeError, ValueError) as error:
            raise RuntimeError('Invalid AdaTreeV1 feature values') from error
        ratio = int(action.rsplit('-', 1)[1]) / 100
        if (event.get('schema') != schema or event.get('request_id') != request_id
                or event.get('chunk_size', 4096) != chunk_size
                or event.get('action') != action or event.get('ratio') != ratio
                or event.get('independent_probes') != 0 or event.get('feature_mask_only') is not True
                or selections[0].get('request_id') != request_id):
            raise RuntimeError('Invalid AdaTreeV1 decision receipt')
        current = dict(features=features, action=action, ratio=ratio)
        if common is not None and current != common:
            raise RuntimeError('Tensor-parallel ranks disagree on AdaTreeV1 features or action')
        common = current
    if common is None:
        raise RuntimeError('Missing AdaTreeV1 routing event')
    verify_diagnostics(workers, sample, 'tree-prophetkv', common['ratio'], tp, range(64))
    return dict(schema=schema, **common, **({'chunk_size': chunk_size} if chunk_size != 4096 else {}),
                request_id=request_id, independent_probes=0,
                feature_mask_only=True, included_in_native_probe=True)


def answer(engine, seed, *, baseline=False):
    import time
    import uuid
    from runner.cache.setups import atomic_json
    from runner.corpus.records import NATIVE_ANSWER_VALIDATION
    from runner.corpus.runtime import clean_capture
    from runner.runtime.generation import generate
    from runner.runtime.worker import arm, configure, drain
    from runner.router.measure import verify_retired
    from runner.reporting.aggregate import evaluation_metadata, score_answer, scoring_text
    from runner.runtime.thinking_budget import KEY, validate_thinking_outcome
    from runner.runtime.output_budget import BUDGETS as TOTAL_BUDGETS

    sample = generation_sample(engine.sample, engine.protocol)
    chunk_size = engine.protocol.get('kv_chunk_size') or 4096
    generation_args = {}
    if engine.protocol.get('answer_generation'):
        generation_args['loop_breaker'] = engine.protocol['answer_generation']['loop_breaker']
    method = 'nocache' if baseline else METHOD
    rid = engine.rid('read:' + method + '-' + uuid.uuid4().hex)
    sync = []; sync_seconds = 0.
    if not baseline:
        clean_capture(engine.llm, engine.tp)
        started = time.perf_counter()
        sync = engine.llm.collective_rpc(configure, kwargs=dict(method=METHOD, ratio=None,
                                                              adatree_chunk_size=chunk_size))
        if sorted(value.get('rank', -1) for value in sync) != list(range(engine.tp)):
            raise RuntimeError('Missing AdaTreeV1 configuration rank')
        engine.llm.collective_rpc(arm, kwargs=dict(request_id=rid,
            boundaries=sample['boundaries'], question_positions=sample['question_positions']))
        sync_seconds = time.perf_counter() - started

    def progress(value):
        atomic_json(engine.session/'answer-progress.json', dict(value, method=method, seed=seed, at=time.time()))

    output, ttft, elapsed = generate(engine.llm.llm_engine, sample['token_ids'],
        sample['max_output_tokens'], rid, sample['thinking'],
        sample=sample, seed=seed, progress=progress, **generation_args)
    diagnostics = engine.llm.collective_rpc(drain)
    if baseline:
        from runner.runtime.generation import verify_diagnostics
        verify_diagnostics(diagnostics, sample, 'baseline', None, engine.tp, range(64))
        if output.num_cached_tokens != 0:
            raise ValueError('No-cache answer reused KV')
        routing = None
    else:
        routing = validate_routing(diagnostics, sample, rid, engine.tp, chunk_size=chunk_size)
    retired = verify_retired(engine.llm, engine.tp, rid, None if baseline else engine.scheduler)
    if not baseline:
        engine.pc.unchanged()
        clean_capture(engine.llm, engine.tp)
    tokens = list(output.outputs[0].token_ids)
    metrics = engine.analyzer.analyze(tokens,
        [] if hasattr(output, 'ruler_thinking_state') else sample['token_ids'], sample['thinking'])
    if hasattr(output, 'ruler_thinking_state'):
        metrics.update(output.ruler_thinking_state.metrics(engine.analyzer.tokenizer))
    evaluation = evaluation_metadata({}, engine.row)
    record = engine.common(method, dict(answer_engine_ttft_seconds=ttft,
        ttft_seconds=sync_seconds+ttft, generation_seconds=elapsed,
        tp_sync_seconds=sync_seconds), retired)
    record.update(executed_action='nocache' if baseline else routing['action'], independent_probes=0,
        seed=seed, sequence_id=engine.row['id'], decision_sync=sync,
        output_token_ids=tokens, prediction=output.outputs[0].text,
        num_cached_tokens=output.num_cached_tokens,
        finish_reason=getattr(output.outputs[0], 'finish_reason', None),
        answer_validation=NATIVE_ANSWER_VALIDATION,
        max_output_tokens=sample['max_output_tokens'],
        output_cap_reached=len(tokens)>=sample['max_output_tokens'],
        **metrics, **evaluation,
        accuracy=score_answer(scoring_text(output.outputs[0].text, metrics, evaluation), evaluation))
    if routing is not None:
        record['routing'] = routing
    if 'scored_text' in metrics:
        record['timings']['first_answer_content_seconds'] = (sync_seconds+output.first_answer_content_seconds
            if output.first_answer_content_seconds is not None else None)
        record['output_cap_reached'] = (metrics['answer_cap_reached'] if sample[KEY]['protocol'] in TOTAL_BUDGETS
            else metrics['thinking_cap_reached'] or metrics['answer_cap_reached'])
        validate_thinking_outcome(dict(record, ttft_seconds=record['timings']['ttft_seconds'],
            first_answer_content_seconds=record['timings']['first_answer_content_seconds']), sample[KEY])
    if engine.protocol.get('answer_generation'):
        record.update(prepared_generation_policy=engine.sample[KEY], **{KEY: sample[KEY]})
    return record, diagnostics



def validate_record(protocol, result):
    """Cheap resume validation; native diagnostics are checked at execution."""
    from runner.tree.collection import inventory
    from ucm.sparse.prophetkv.adatree import route_features, feature_names, protocol_for
    chunk_size = protocol.get('kv_chunk_size') or 4096
    schema = protocol_for(chunk_size)
    if METHOD not in inventory(protocol) or protocol.get('adaptive_routing') != schema:
        raise ValueError('AdaTreeV1 record requires its declared collection protocol')
    routing = result.get('routing', {})
    features = routing.get('features', {})
    if set(features) != set(feature_names(chunk_size)):
        raise ValueError('Missing AdaTreeV1 features')
    action = route_features(**features, chunk_size=chunk_size)
    if (routing.get('schema') != schema or routing.get('action') != action
            or routing.get('chunk_size', 4096) != chunk_size
            or result.get('executed_action') != action
            or routing.get('ratio') != int(action.rsplit('-', 1)[1])/100
            or routing.get('independent_probes') != 0 or result.get('independent_probes') != 0
            or routing.get('feature_mask_only') is not True
            or routing.get('included_in_native_probe') is not True
            or not routing.get('request_id')):
        raise ValueError('Accepted AdaTreeV1 routing decision changed')


def answer_generation():
    """Explicit A800 override; original prepared generation policy stays intact."""
    from runner.runtime.output_budget import ADATREE_PROTOCOL
    from runner.runtime.thinking_loop import ADATREE_CONFIG
    return dict(protocol=ADATREE_PROTOCOL, thinking_cap=8192, output_reserve=10000,
                loop_breaker=dict(ADATREE_CONFIG))


def generation_policy(prepared_policy, protocol):
    from runner.runtime.output_budget import PROTOCOL as PREPARED, validate_policy
    from ucm.sparse.prophetkv.adatree import protocol_for
    override = protocol.get('answer_generation')
    if override is None:
        return prepared_policy
    if (protocol.get('adaptive_routing') != protocol_for(protocol.get('kv_chunk_size') or 4096)
            or override != answer_generation()
            or prepared_policy.get('protocol') != PREPARED):
        raise ValueError('Invalid AdaTreeV1 generation override')
    validate_policy(prepared_policy)
    result = dict(prepared_policy, **{k: override[k] for k in ('protocol', 'thinking_cap', 'output_reserve')})
    return validate_policy(result)


def generation_sample(prepared, protocol):
    from runner.runtime.thinking_budget import KEY
    if not protocol.get('answer_generation'):
        return prepared
    policy = generation_policy(prepared[KEY], protocol)
    return dict(prepared, max_output_tokens=policy['output_reserve'], **{KEY: policy})
