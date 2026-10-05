#!/usr/bin/env python3
"""Offline CPU train/evaluate on RULER, LongBench v2, or their portable union."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.router_dataset import load as load_dataset
from scripts.longbench_a800_data import assign_folds, LENGTHS, ACTIONS
from runner.longbench_features import FEATURES, DEFINITIONS, SCHEMA
from runner.layout import PROMPT_PROTOCOL
from runner.setups import atomic_json, file_hash, fingerprint
from runner.tree_policy import actions, decide, export, load as load_tree
from runner.corpus_train import search, GRID, leaf, choose


def select_data(mode, ruler=None, longbench=None, requested=('common',)):
    paths = {'ruler': ruler, 'longbench-v2': longbench}
    datasets = ('ruler', 'longbench-v2') if mode == 'both' else (mode,)
    bundles = {}
    for name in datasets:
        if paths[name] is None:
            raise ValueError(f'Provide --{name if name == "ruler" else "longbench"}-data')
        bundle = load_dataset(paths[name])
        if bundle['dataset'] != name:
            raise ValueError('Dataset file does not match the requested mode')
        bundles[name] = bundle
    inventories = {name: actions(b['actions']) for name, b in bundles.items()}
    common = set.intersection(*(set(a) for a in inventories.values()))
    selected = common if list(requested) == ['common'] else set(requested)
    if selected != set(a['id'] for a in ACTIONS) or selected != common:
        raise ValueError('Training requires the complete canonical 12-action inventory; no subset or silent intersection')
    first = next(iter(inventories.values()))
    inventory = sorted((first[name] for name in selected), key=lambda a: (-1 if a['ratio'] is None else a['ratio']))
    for name in selected:
        if any(a[name] != first[name] for a in inventories.values()):
            raise ValueError('Same action ID has different method/ratio definitions')
    rows = [dict(row, sample_id=f"{dataset}:{row['id']}") for dataset, b in bundles.items() for row in b['rows']]
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate training identity')
    metadata = dict(mode=mode, dataset_hashes={n: b['payload_sha256'] for n, b in bundles.items()},
                    dataset_provenance={n:b['provenance'] for n,b in bundles.items()},
                    actions=inventory, excluded_actions={n: sorted(set(a)-selected) for n, a in inventories.items()},
                    evaluation='training', training_overlap=True, weighting='equal-prompt',
                    cost_normalization='(answer TTFT + probe overhead) / same-prompt nocache TTFT',
                    interpretation='All selected samples are used for fitting and resubstitution evaluation; no held-out performance claim',
                    timing='Offline fixed-action TTFT plus measured feature-probe overhead, not fresh router inference. Source hardware and sessions can differ.')
    return rows, inventory, metadata


def evaluate(tree, rows):
    decisions = []
    trained = set(tree['training_ids'])
    if tree.get('prompt_protocol') != PROMPT_PROTOCOL:
        raise ValueError('Tree prompt protocol is incompatible')
    for row in rows:
        decision = decide(tree, row['features']); case = decision['action']
        if case not in row['outcomes']:
            raise ValueError(f'Missing measured outcome for selected action: {case}')
        outcome, baseline = row['outcomes'][case], row['outcomes']['nocache']
        decisions.append(dict(sample_id=row['sample_id'], dataset=row['dataset'], task=row['subtask'],
            length=row.get('length'), training_overlap=(row['sample_id'] in trained and
                tree.get('training_input_hashes', {}).get(row['sample_id'], row['input_sha256']) == row['input_sha256']),
            action=case, fallback_reason=decision['fallback_reason'], accuracy=outcome['accuracy'],
            baseline_accuracy=baseline['accuracy'], answer_ttft_seconds=outcome['ttft_seconds'],
            estimated_router_ttft_seconds=outcome['ttft_seconds']+row['probe_overhead_seconds'],
            baseline_ttft_seconds=baseline['ttft_seconds'], probe_overhead_seconds=row['probe_overhead_seconds'],
            normalized_router_cost=(outcome['ttft_seconds']+row['probe_overhead_seconds'])/baseline['ttft_seconds'],
            thinking_tokens=outcome['thinking_tokens'],
            answer_tokens=outcome['answer_tokens'], output_cap_reached=outcome['output_cap_reached']))
    def metric(selected):
        if not selected:
            return dict(samples=0, accuracy_percent=None, estimated_speedup=None)
        mean = lambda key: float(np.mean([r[key] for r in selected]))
        return dict(samples=len(selected), accuracy_percent=100*mean('accuracy'),
            baseline_accuracy_percent=100*mean('baseline_accuracy'),
            accuracy_delta_pp=100*(mean('accuracy')-mean('baseline_accuracy')),
            mean_answer_ttft_seconds=mean('answer_ttft_seconds'),
            mean_estimated_router_ttft_seconds=mean('estimated_router_ttft_seconds'),
            mean_baseline_ttft_seconds=mean('baseline_ttft_seconds'),
            mean_probe_overhead_seconds=mean('probe_overhead_seconds'),
            estimated_speedup=mean('baseline_ttft_seconds')/mean('estimated_router_ttft_seconds'),
            mean_normalized_router_cost=mean('normalized_router_cost'),
            normalized_estimated_speedup=1/mean('normalized_router_cost'),
            mean_thinking_tokens=mean('thinking_tokens'), mean_answer_tokens=mean('answer_tokens'),
            output_caps=sum(r['output_cap_reached'] for r in selected),
            training_overlap=sum(r['training_overlap'] for r in selected),
            missing_feature_fallbacks=sum(r['fallback_reason'] == 'missing_required_feature' for r in selected),
            actions=dict(Counter(r['action'] for r in selected)))
    result = dict(overall=metric(decisions), datasets={})
    for dataset in sorted({r['dataset'] for r in decisions}):
        subset = [r for r in decisions if r['dataset'] == dataset]
        item = dict(overall=metric(subset), tasks={t: metric([r for r in subset if r['task'] == t])
                                                 for t in sorted({r['task'] for r in subset})})
        if dataset == 'longbench-v2':
            item['lengths'] = {label: metric([r for r in subset if r['length'] == label]) for label in LENGTHS}
        result['datasets'][dataset] = item
    return result, decisions


def train(rows, inventory, metadata, output, seed=42, policy_count=3):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'data-snapshot.json', dict(metadata=metadata, rows=rows))
    # All rows enter the final fit; OOF folds select settings inside that training set.
    samples = [dict(id=r['sample_id'], task=r['dataset']+'/'+r['subtask']) for r in rows]
    folds = assign_folds(rows, seed)
    fold_values = [folds[r['sample_id']] for r in rows]
    features = [r['features'] for r in rows]; names = [a['id'] for a in inventory]
    scores = [[r['outcomes'][a]['accuracy'] for a in names] for r in rows]
    # Ratios are per prompt, before any fold-local fitting. Scaling one server's
    # entire timing trace cannot change its weight or the selected policies.
    times = [[r['outcomes'][a]['ttft_seconds']/r['outcomes']['nocache']['ttft_seconds'] for a in names] for r in rows]
    overhead = [r['probe_overhead_seconds']/r['outcomes']['nocache']['ttft_seconds'] for r in rows]
    atomic_json(output/'folds.json', dict(seed=seed, folds=folds, training_ids=[r['sample_id'] for r in rows], heldout_ids=[]))
    settings = dict(metadata, seed=seed, feature_names=list(FEATURES), feature_definitions=DEFINITIONS,
                    policy_count=policy_count, grid=GRID, targets=dict(macro_loss=.02, speedup=4),
                    selection_objective='Equal prompt accuracy and mean per-prompt normalized total cost; OOF only',
                    cost_units='dimensionless multiples of each prompt baseline; search mean_total_ttft is normalized cost, not seconds',
                    score_weighting='prompt', sample_weight=1/len(rows))
    atomic_json(output/'settings.json', settings)
    policies, table, _ = search(samples, features, scores, times, overhead, inventory, seed, policy_count,
                                 feature_names=FEATURES, folds=fold_values, score_weighting='prompt')
    atomic_json(output/'search.json', table)
    reports = {}
    for policy in policies:
        raw = policy.pop('raw_tree'); name = policy['id']
        atomic_json(output/f'{name}-fit.json', raw)
        policy['trainer'] = 'coverage-five-prompt-normalized-v2'
        policy['training_cost_units'] = 'mean per-prompt baseline multiples including measured probe overhead'
        tree = dict(policy, schema=SCHEMA, feature_names=list(FEATURES), feature_definitions=DEFINITIONS,
                    prompt_protocol=PROMPT_PROTOCOL,
                    compatible_evaluation_protocols=list(dict.fromkeys(r.get('evaluation_protocol') for r in rows)),
                    actions=inventory, training_ids=[r['sample_id'] for r in rows],
                    training_input_hashes={r['sample_id']: r['input_sha256'] for r in rows}, heldout_ids=[],
                    evaluation='training', training_overlap=True,
                    provenance=dict(snapshot_sha256=file_hash(output/'data-snapshot.json'),
                        training_run_sha256=file_hash(output/'settings.json'),
                        corpus_protocol_sha256=fingerprint(metadata['dataset_hashes']),
                        plan_sha256=file_hash(output/'folds.json'), hardware='See portable dataset outcome GPU UUIDs; mixed source hardware'))
        tree = export(output/f'{name}.json', tree)
        # Verify native fit versus exported predicates at every exact boundary.
        vectors = [dict.fromkeys(FEATURES, .5), {}]
        def boundaries(node, values):
            if 'feature' not in node:
                return
            feature = FEATURES[node['feature']]; threshold = node['threshold']
            for value in (np.nextafter(threshold, -np.inf), threshold, np.nextafter(threshold, np.inf)):
                vectors.append(dict(values) | {feature: float(value)})
            boundaries(node['left'], dict(values) | {feature: threshold})
            boundaries(node['right'], dict(values) | {feature: float(np.nextafter(threshold, np.inf))})
        boundaries(raw, dict.fromkeys(FEATURES, .5))
        dense = names.index('nocache'); sparse = [i for i in range(len(names)) if i != dense]
        for vector in [*features, *vectors]:
            missing = any(vector.get(f) is None for f in FEATURES)
            expected = 'nocache' if missing else names[choose(leaf(raw, [vector[f] for f in FEATURES]),
                policy['training_costs'], policy['setting'], dense, sparse)]
            if decide(tree, vector)['action'] != expected:
                raise ValueError('Exported tree differs from fitted decisions')
        report, decisions = evaluate(tree, rows)
        atomic_json(output/f'{name}-evaluation.json', dict(report, metadata=metadata))
        with (output/f'{name}-decisions.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(decisions[0])); writer.writeheader(); writer.writerows(decisions)
        reports[name] = dict(report, oof=policy['oof'])
    summary = dict(metadata, samples=len(rows), primary='router1', policies=reports)
    atomic_json(output/'summary.json', summary)
    (output/'summary.txt').write_text(json.dumps(summary, indent=2)+'\n')
    atomic_json(output/'complete.json', dict(complete=True, files={str(p.relative_to(output)): file_hash(p)
                        for p in output.rglob('*') if p.is_file()}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('ruler', 'longbench-v2', 'both'), required=True)
    parser.add_argument('--ruler-data', type=Path); parser.add_argument('--longbench-data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--actions', nargs='+', default=['common'], help='All 12 measured actions are required')
    parser.add_argument('--seed', type=int, default=42); parser.add_argument('--policies', type=int, default=3)
    parser.add_argument('--tree', type=Path, help='Evaluate an existing tree only; no refit')
    args = parser.parse_args()
    rows, inventory, metadata = select_data(args.dataset, args.ruler_data, args.longbench_data, args.actions)
    if args.tree:
        tree = load_tree(args.tree)
        if tree['schema'] != SCHEMA or not set(actions(tree['actions'])) <= set(actions(inventory)):
            raise ValueError('Evaluation data does not cover the tree feature/action schema')
        args.output.mkdir(parents=True, exist_ok=False)
        result, decisions = evaluate(tree, rows)
        metadata.update(evaluation='offline-measured-outcomes',
                        training_overlap=any(r['training_overlap'] for r in decisions),
                        interpretation='Existing-tree evaluation without refit; overlap with original training IDs is reported per dataset')
        atomic_json(args.output/'evaluation.json', dict(result, metadata=metadata, refit=False, tree_sha256=file_hash(args.tree)))
        atomic_json(args.output/'decisions.json', decisions)
        with (args.output/'decisions.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(decisions[0])); writer.writeheader(); writer.writerows(decisions)
        (args.output/'evaluation.txt').write_text(json.dumps(dict(result, metadata=metadata), indent=2)+'\n')
    else:
        if args.policies != 3:
            parser.error('This workflow exports exactly three policies')
        result = train(rows, inventory, metadata, args.output, args.seed, args.policies)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
