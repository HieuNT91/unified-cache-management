"""Per-request logits and scheduler hooks for AdaTree/no-cache decoding."""
from runner.runtime.output_budget import KEY, BudgetState
from runner.runtime.thinking_loop import ADATREE_CONFIG


def request_state(request):
    params = request.sampling_params
    from runner.runtime.thinking_loop import KEY as LOOP_KEY
    extra = getattr(params, 'extra_args', None) or {}
    policy = extra.get(KEY)
    loop_breaker = extra.get(LOOP_KEY)
    if policy is None:
        return None
    state = getattr(request, '_ruler_thinking_state', None)
    if state is None:
        if loop_breaker != ADATREE_CONFIG:
            raise ValueError('Expected AdaTree thinking loop breaker')
        state = request._ruler_thinking_state = BudgetState(policy)
    elif state.policy != policy or state.loop.config != loop_breaker:
        raise ValueError('Thinking policy changed during request')
    return state.consume(request.output_token_ids)


def enforce_logits(model_runner, logits):
    """Runs for baseline and sparse paths immediately before the native sampler."""
    for rid, index in model_runner.input_batch.req_id_to_index.items():
        state = request_state(model_runner.requests[rid])
        if state is None:
            continue
        if model_runner.speculative_config:
            raise ValueError('Thinking budget requires ordinary autoregressive decoding')
        forced = state.forced_next()
        if forced is not None:
            logits[index].fill_(float('-inf'))
            logits[index, forced] = 0.
        else:
            if state.phase not in ('thinking', 'answer'):
                raise ValueError('Sampling a finished thinking request')
            allowed = {state.policy['close_id']} if state.phase == 'thinking' else set(state.policy['eos_ids'])
            if state.phase == 'thinking' and state.seen == 0 and not state.policy['opening_in_prompt']:
                allowed.add(state.policy['open_id'])
            logits[index, sorted(set(state.policy['control_ids']) - allowed)] = float('-inf')


def stopping_hook(original, statuses):
    def check_stop(request, max_model_len, pooler_output=None):
        state = request_state(request)
        if state is None:
            return original(request, max_model_len, pooler_output)
        if state.phase == 'done':
            request.status = statuses.FINISHED_STOPPED if state.answer_eos else statuses.FINISHED_LENGTH_CAPPED
            if state.loop.broken:
                request.stop_reason = 'thinking_broken'
            return True
        if request.num_tokens >= max_model_len or request.num_output_tokens >= request.max_tokens:
            raise ValueError('Thinking request exhausted reserved capacity before answer completion')
        return False
    return check_stop


def install_scheduler_hook():
    from vllm.v1.core.sched import scheduler, utils
    from vllm.v1.request import RequestStatus
    if not getattr(utils.check_stop, '_ruler_budget_hook', False):
        hook = stopping_hook(utils.check_stop, RequestStatus)
        hook._ruler_budget_hook = True
        utils.check_stop = scheduler.check_stop = hook
