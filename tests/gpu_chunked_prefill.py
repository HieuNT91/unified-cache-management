"""Opt-in integration matrix. Never invoked by CPU unittest discovery."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner.config import validate_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuids', nargs='+', required=True)
    args = parser.parse_args()
    devices = args.gpu_uuids
    if len(devices) not in (1, 2, 4, 8) or len(set(devices)) != len(devices) or any(
            not d.startswith('GPU-') for d in devices):
        parser.error('Supply distinct, explicitly assigned GPU UUIDs (TP1/2/4/8)')
    sample = validate_sample(json.loads(args.input.read_text()))
    if len(sample['token_ids']) - sample['boundaries'][1] <= 32768:
        parser.error('Use a prepared input requiring at least three cached prefill steps')
    args.output.mkdir(parents=True, exist_ok=False)
    # Preserve every prepared input token; use the deterministic per-task
    # output cap for this integration check, leaving the source file untouched.
    # Original RULER task caps remain unchanged; one-token probes are internal only.
    prepared = args.output.resolve() / 'input.json'
    prepared.write_text(json.dumps(sample))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(devices), PYTHON_BIN=sys.executable)
    cases = [('baseline', 'baseline', [])]
    for ratio in ('0', '0.2', '1'):
        cases.append((f'prophetkv-{ratio}', 'prophetkv', ['--ratio', ratio]))
        cases.append((f'selective-{ratio}', 'selective_prophetkv',
                      ['--ratio', ratio, '--layers', '45', '48', '50', '56', '58']))
    cases.append(('selective-last5', 'selective_prophetkv', ['--ratio', '.2', '--num-layers', '5']))
    results = {}
    for name, method, extra in cases:
        output = args.output.resolve() / name
        subprocess.run(['bash', str(ROOT / 'run.sh'), 'run', '--model', str(args.model.resolve()),
            '--input', str(prepared), '--output', str(output), '--method', method,
            '--tp', str(len(devices)), *extra], cwd=ROOT, env=env, check=True)
        record = json.loads((output / 'result.json').read_text())
        events = json.loads((output / 'diagnostics.json').read_text())[0]['diagnostics']
        steps = [e for e in events if e['kind'] == 'prefill_step']
        if len(steps) < 3 or any(e['scheduled_tokens'] > 16384 for e in steps):
            raise RuntimeError(f'{name}: expected actual 16K chunking')
        if name.endswith('-0') and not any(e['no_forward'] for e in steps):
            raise RuntimeError(f'{name}: missing empty recomputation range')
        results[name] = record['output_token_ids']
    for name in ('prophetkv-1', 'selective-1'):
        if results[name] != results['baseline']:
            raise RuntimeError(f'{name}: 100% repair output differs from baseline; investigate numerical drift')
    (args.output / 'validation.json').write_text(json.dumps(dict(passed=True, first_tokens=results), indent=2))


if __name__ == '__main__':
    main()
