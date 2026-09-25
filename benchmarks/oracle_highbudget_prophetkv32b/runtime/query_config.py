"""Two explicitly ordered experiments, with original full-question queries."""
LAYERS=(45,48,50,56,58)
ORACLE_BUDGETS=(10,30,50,70,90)
HIGH_BUDGETS=(90,80,95)
ORACLE_CASES=('target_only-all64-0',)+tuple(f'target_union-{scope}-{b}' for b in ORACLE_BUDGETS for scope in ('all64','selected5'))
HIGH_CASES=tuple(f'prophetkv-{scope}-{b}' for b in HIGH_BUDGETS for scope in ('all64','selected5'))
CASES=ORACLE_CASES+HIGH_CASES
STAGES=('oracle','highbudget')
TASKS={'oracle':('niah_multikey_1','niah_multikey_2','niah_multikey_3'),
       'highbudget':('niah_multikey_2','niah_multikey_3','cwe')}
COUNTS={'oracle':10,'highbudget':50}
TARGETS={'oracle':330,'highbudget':900}
def configuration(case):
    if case not in CASES:raise ValueError('Unknown experiment configuration: '+case)
    mode,layer,budget=case.split('-')
    return dict(query_scope='full_question',selection_mode=mode,layer_scope=layer,budget=int(budget),
        method='prophetkv' if layer=='all64' else 'selective_prophetkv',
        scoring_layers=[] if mode=='target_only' else list(range(64)) if layer=='all64' else list(LAYERS))
def positions(entry,case):
    configuration(case)
    return entry['query']['positions']
def oracle_positions(entry,case):
    return entry['oracle']['eligible_positions'] if configuration(case)['selection_mode']!='prophetkv' else []
