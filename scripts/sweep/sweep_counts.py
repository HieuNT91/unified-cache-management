#!/usr/bin/env python3
"""Quick, read-only count of saved answers; no token loading or diagnostic replay."""
import argparse
import json
from pathlib import Path


def saved_counts(output, manifest, percentages, shards=2, excluded_percentages=None):
    output, manifest = Path(output), Path(manifest)
    if not output.is_dir():
        raise FileNotFoundError(output)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    ids = [row['id'] for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Manifest prompt IDs must be nonempty and unique')
    if not percentages or len(set(percentages)) != len(percentages) or any(not 0 < p <= 100 for p in percentages):
        raise ValueError('Provide distinct percentages in [1,100]')
    if shards < 1:
        raise ValueError('shards must be positive')
    n = len(ids)
    if excluded_percentages is None:
        marker = output/'continuation.json'
        excluded_percentages = json.loads(marker.read_text()).get('excluded_percentages', []) if marker.exists() else []
    if not set(excluded_percentages) <= set(percentages) or len(set(excluded_percentages)) == len(percentages):
        raise ValueError('Invalid excluded percentages')
    all_vanilla = [f'prophetkv-{p}' for p in percentages]
    vanilla = [f'prophetkv-{p}' for p in percentages if p not in excluded_percentages]
    selective = [f'selective-{p}' for p in percentages]
    found = {}
    print(f'Saved completed result files (not yet revalidated): {output}', flush=True)
    print(f'{"Configuration":<22} {"Saved/expected":>17} {"Remaining":>10}', flush=True)
    for method in ['baseline', *all_vanilla, *selective]:
        indices = {int(path.parent.name) for path in (output/method).glob('*/result.json')
                   if path.is_file() and path.parent.name.isdigit()
                   and 0 <= int(path.parent.name) < n
                   and path.parent.name == f'{int(path.parent.name):06d}'}
        found[method] = indices
        remaining = str(n-len(indices)) if method in ['baseline', *vanilla] else 'excluded'
        print(f'{method:<22} {str(len(indices))+"/"+str(n):>17} {remaining:>10}', flush=True)
    saved = sum(len(found[name]) for name in ['baseline', *vanilla])
    expected = n*(1+len(vanilla))
    complete = set.intersection(*(found[name] for name in vanilla))
    print(f'Baseline + ProphetKV: {saved}/{expected} saved; {expected-saved} remaining.', flush=True)
    print(f'Prompts complete at every ProphetKV ratio: {len(complete)}/{n}.', flush=True)
    for shard in range(shards):
        assigned = set(range(shard,n,shards))
        baseline = len(found['baseline'] & assigned)
        cached = sum(len(found[name] & assigned) for name in vanilla)
        print(f'Group {shard}: baseline {baseline}/{len(assigned)}, '
              f'ProphetKV {cached}/{len(assigned)*len(vanilla)} saved.', flush=True)
    print('Selective results are retained but excluded from the continuation.\n', flush=True)
    if excluded_percentages:
        print(f'Excluded ProphetKV percentages (saved results retained): {excluded_percentages}', flush=True)
    return dict(saved=saved, expected=expected, remaining=expected-saved,
                complete_prompts=len(complete), methods={name:len(values) for name,values in found.items()},
                excluded_percentages=excluded_percentages)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--percentages',type=int,nargs='+',default=[1,5,10,15,20,30])
    parser.add_argument('--shards',type=int,default=2)
    args=parser.parse_args()
    saved_counts(args.output,args.manifest,args.percentages,args.shards)
