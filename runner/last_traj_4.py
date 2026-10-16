"""Fixed last-token/four-draft control, separate from learned router actions."""
import math

METHOD = 'last_traj_4'
SCHEMA = 'last-traj-4-comparison-v1'
FIXED_METHOD = 'last_traj_4f'
FIXED_SCHEMA = 'last-traj-4f-comparison-v1'
QUERY_METHOD = 'query_traj_4f'
QUERY_SCHEMA = 'query-traj-4f-mean-query-comparison-v2'
COLLECTION_SCHEMA = 'l20-ruler-thinking-traj4f-v1'
METHODS = (METHOD, FIXED_METHOD, QUERY_METHOD)
SCHEMAS = dict(zip(METHODS, (SCHEMA, FIXED_SCHEMA, QUERY_SCHEMA)))


def definition(ratio, method=METHOD):
    if method not in METHODS:
        raise ValueError('Unknown trajectory method')
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 <= ratio <= 1:
        raise ValueError('Invalid trajectory ratio')
    return dict(id=method, method=method, ratio=ratio)


def inventory(protocol):
    schema = protocol.get('schema')
    if schema == COLLECTION_SCHEMA:
        expected = control_actions(protocol['percentages'], FIXED_METHOD)[1:]
        if (protocol.get('actions') != expected or protocol.get('scheduled_actions') != [a['id'] for a in expected]
                or protocol.get('answer_validation') != 'native-answer-diagnostics-v1'):
            raise ValueError('Invalid last_traj_4f collection protocol')
        return {d['id']: d for d in expected}
    method = next((m for m, s in SCHEMAS.items() if s == schema), METHOD)
    expected = control_actions(protocol['percentages'], method)
    if (schema not in SCHEMAS.values() or protocol.get('actions') != expected
            or protocol.get('answer_validation') != 'native-answer-diagnostics-v1'):
        raise ValueError('Invalid last_traj_4 protocol')
    return {d['id']: d for d in expected}


def control_actions(percentages, method=METHOD):
    if (not percentages or len(set(percentages)) != len(percentages)
            or any(type(p) is not int or not 0 <= p <= 100 for p in percentages)):
        raise ValueError('Expected distinct integer percentages in [0, 100]')
    return [dict(id='nocache', method='baseline', ratio=None)] + [
        dict(definition(p / 100, method), id=f'{method}_{p}pct') for p in percentages]


def validate_selection(event, sample, ratio, method=METHOD):
    """Validate native evidence; copying does not claim independent DP replay."""
    prefix, end = sample['boundaries'][1], sample['boundaries'][-2]
    selected = event['selected_positions']
    z = event.get('copy_z')
    if method not in METHODS:
        raise ValueError('Unknown trajectory method')
    fixed = method in (FIXED_METHOD, QUERY_METHOD)
    probe_positions = sample['question_positions'] if method == QUERY_METHOD else [sample['boundaries'][-1] - 1]
    if method == QUERY_METHOD and (event.get('trajectory_head_positions') != sample['question_positions']
            or event.get('trajectory_head_fusion') != 'mean_query_tokens'):
        raise RuntimeError('Query trajectory must start at the mean over all original question tokens')
    if (event.get('selection_method') != method or event.get('drafts') != 4
            or len(event.get('draft_tokens', [])) != 4
            or any(type(t) is not int or t < 0 for t in event['draft_tokens'])
            or type(event.get('copying')) is not bool
            or not isinstance(z, (int, float)) or not math.isfinite(z)
            or event['copying'] != (z >= 2.0) or event.get('copy_z_threshold') != 2.0
            or event.get('trajectory_lambda') != 16384.0 or event.get('trajectory_lam_rel') != (1.0 if fixed else 0.0)
            or event.get('traj_unbiased') is not fixed
            or event.get('probe_question_positions') != probe_positions
            or event.get('question_positions') != sample['question_positions']
            or event.get('ratio') != ratio or event.get('eligible_count') != end - prefix
            or event.get('selected_count') != len(selected)
            or len(selected) != math.floor((end - prefix) * ratio)
            or selected != sorted(set(selected))
            or any(type(p) is not int or not prefix <= p < end for p in selected)):
        raise RuntimeError('Invalid last_traj_4 native selection evidence')
    if event['copying']:
        return selected
    scores = event['scores']
    return sorted(i + prefix for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:len(selected)])
