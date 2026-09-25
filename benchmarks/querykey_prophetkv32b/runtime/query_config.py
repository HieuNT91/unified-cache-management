"""Private, explicit routing for the query-row ablation."""
BUDGETS=(5,10,20,30,40,50,60)
LAYERS=(45,48,50,56,58)
TASKS=('niah_multikey_2','niah_multikey_3','cwe')
CASES=tuple(f'{query}-{layers}-{budget}' for budget in BUDGETS
            for query in ('focus','full_question') for layers in ('all64','selected5'))
def configuration(case):
    query,layer,budget=case.split('-')
    if query not in ('focus','full_question') or layer not in ('all64','selected5') or int(budget) not in BUDGETS:
        raise ValueError('Unknown ablation configuration: '+case)
    return dict(query_scope=query,layer_scope=layer,budget=int(budget),
                method='prophetkv' if layer=='all64' else 'selective_prophetkv',
                scoring_layers=list(range(64)) if layer=='all64' else list(LAYERS))
def positions(entry,case):
    return entry['focus']['absolute_positions'] if configuration(case)['query_scope']=='focus' else entry['query']['positions']
def schedule(samples,retained):
    keys={(r['sample_id'],r['case']) for r in retained}
    return [m|dict(case=c) for c in CASES for m in samples if (m['id'],c) not in keys]
