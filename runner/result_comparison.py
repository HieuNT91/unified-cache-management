"""Compare new answers with explicitly selected historical outputs, without replay."""
import json
from pathlib import Path
from runner.setups import file_hash

IDENTITY_FIELDS=('token_sha256','template_sha256','prompt_protocol','evaluation_protocol',
                 'max_output_tokens','executed_action','gpu_uuids','subtask','references','scoring')
OUTPUT_FIELDS=('output_token_ids','prediction','accuracy','thinking_tokens','answer_tokens',
               'control_tokens','output_cap_reached','unfinished_thinking')


def load_reference(protocol):
    reference=protocol.get('comparison_reference')
    if reference is None:return None
    path=Path(reference['path'])
    if file_hash(path)!=reference['sha256']:raise ValueError('Historical comparison reference changed')
    return json.loads(path.read_text())['records']


def compare_result(result,reference):
    old=reference.get(result['token_sha256'],{}).get(result['executed_action'])
    if old is None:return dict(status='new_input')
    mismatches=[k for k in IDENTITY_FIELDS if old.get(k)!=result.get(k)]
    if mismatches:return dict(status='incompatible',fields=mismatches,old_prompt_id=old['prompt_id'])
    fields={k:result[k]==old[k] for k in OUTPUT_FIELDS}
    return dict(status='matched' if all(fields.values()) else 'different',fields=fields,
        old_prompt_id=old['prompt_id'],old_result_sha256=old['result_sha256'],
        old_ttft_seconds=old['ttft_seconds'])


def summarize(records):
    comparisons=[r['historical_comparison'] for r in records if 'historical_comparison' in r]
    return dict(compared=sum(c['status'] in ('matched','different') for c in comparisons),
        matched=sum(c['status']=='matched' for c in comparisons),different=sum(c['status']=='different' for c in comparisons),
        incompatible=sum(c['status']=='incompatible' for c in comparisons),new_inputs=sum(c['status']=='new_input' for c in comparisons))
