"""Current fixed-head original MixtureDataset with identity dependency adapters.
Tests declared weight support versus available children. No production token store.
"""
import ast, asyncio, hashlib, json, pathlib, runpy, types, warnings

R = pathlib.Path(__file__).resolve().parents[1]
P = R / 'sources/integer_exposure_2026_10_07/mixture.py'
# Reuse documented dependency adapters, then replace the old class with current source.
adapters = runpy.run_path(str(R / 'scripts/probe_mixture_identity_cpu.py'))
ns = adapters['ns']
nodes = [n for n in ast.parse(P.read_text()).body if
         (isinstance(n, ast.ClassDef) and n.name == 'MixtureDataset') or
         (isinstance(n, ast.FunctionDef) and n.name == '_compute_block_assignment')]
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(
    module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes,
    type_ignores=[])), str(P), 'exec'), ns)
Mix, Identity = ns['MixtureDataset'], adapters['Identity']

async def main():
    checks, cases = [], {}
    def ck(name, condition):
        assert condition, name
        checks.append(name)
    def build(weights, order=('A', 'B')):
        return Mix({n: Identity(n, 100) for n in order}, weights, 10, key=7)
    def quota(m, phase=0):
        return dict(zip(m.dataset_index, map(int, m._counts_per_block_per_stage[phase])))
    def attempt(weights):
        try:
            m = build(weights)
            return {'quota': quota(m), 'error': None}
        except Exception as e:
            return {'quota': None, 'error': type(e).__name__, 'message': str(e)}

    good = build({'A': .4, 'B': .4})
    typo = build({'A': .4, 'B': .4, 'TYPO': .2})
    ck('Known-only weights renormalize equally', quota(good) == {'A': 5, 'B': 5})
    ck('Missing positive child accepted and remainder shifts to A', quota(typo) == {'A': 6, 'B': 4})
    stream = await typo.get_batch(list(range(10)))
    ck('Actual identity read follows shifted quota', sum(x.startswith('A:') for x in stream) == 6)
    rev = build({'A': .4, 'B': .4, 'TYPO': .2}, ('B', 'A'))
    ck('Missing support remainder follows tied child order', quota(rev) == {'B': 6, 'A': 4})
    zero = build({'A': .4, 'B': .4, 'TYPO': 0})
    ck('Unknown zero weight leaves known quotas unchanged', quota(zero) == quota(good))
    missing = attempt({'TYPO': 1})
    ck('All missing positive support fails downstream argmax', missing['error'] == 'ValueError' and 'argmax' in missing['message'])
    neg = attempt({'A': .4, 'B': .4, 'TYPO': -.1})
    ck('Unknown negative weight rejected by normalizer', neg['error'] == 'ValueError' and 'negative' in neg['message'])
    staged = build([(0, {'A': 1}), (20, {'A': .4, 'B': .4, 'TYPO': .2})])
    ck('Stage one excludes inactive B without excluding later support', quota(staged, 0) == {'A': 10, 'B': 0})
    ck('Later missing support shifts later quota', quota(staged, 1) == {'A': 6, 'B': 4})
    ss = await staged.get_batch(list(range(20, 30)))
    ck('Later shifted stage retains actual cumulative A index', set(x for x in ss if x.startswith('A:')) == {f'A:{i}' for i in range(20, 26)})
    ck('Single and batched read agree on malformed support', ss == [await staged.getitem_async(i) for i in range(20, 30)])
    cases.update(known_only=quota(good), missing_positive=quota(typo), reversed_children=quota(rev),
                 unknown_zero=quota(zero), all_missing=missing, unknown_negative=neg,
                 malformed_block=stream, later_stage_block=ss,
                 staged_quotas=[quota(staged, i) for i in range(2)])

    configp = R/'sources/boundaries_2026_10_05/datasets.py'
    config_class = next(n for n in ast.parse(configp.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'LmDataConfig')
    init = next(n for n in config_class.body if isinstance(n, ast.FunctionDef) and n.name == '__post_init__')
    guards = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[init], type_ignores=[])), str(configp), 'exec'), guards)
    def guard(weights, components=('A','B')):
        obj = types.SimpleNamespace(components=dict.fromkeys(components), train_weights=weights,
            max_train_batches=None, num_validation_sequences=None, experiment_budget=None, target_budget=None)
        try:
            guards['__post_init__'](obj)
            return {'accepted': True, 'error': None}
        except Exception as e:
            return {'accepted': False, 'error': type(e).__name__, 'message': str(e)}
    for name, weights in [('dict', {'A':.4,'B':.4,'TYPO':.2}),
                          ('stage', [(0,{'A':1}),(20,{'A':.4,'B':.4,'TYPO':.2})]),
                          ('zero', {'A':1,'TYPO':0})]:
        cases['config_guard_'+name] = guard(weights)
        ck('Original config rejects unknown '+name+' key', not cases['config_guard_'+name]['accepted'])
    ck('Original config accepts known-only weights', guard({'A':.5,'B':.5})['accepted'])
    declared = guard({'A':.4,'B':.4,'TYPO':.2}, ('A','B','TYPO'))
    ck('Declared component guard cannot see later actual missing child', declared['accepted'])
    cases['declared_typo_component_guard'] = declared

    # Inspect archived supports independently of child construction adapters.
    specp = R / 'sources/phase_budget_2026_10_07/harrier.json'
    spec = json.loads(specp.read_text()); keys = set(spec['available_tokens'])
    support = []
    for label, phases in [('spec', [x['weights'] for x in spec['phases']]),
        ('historical_wandb', [x[1] for x in json.loads((R / 'sources/wandb/hero-fa4sm100-nomask-step146k_meta.json').read_text())['config']['data']['value']['train_weights']]),
        ('oct7_wandb', [x[1] for x in json.loads((R / 'sources/live_2026_10_07/meta.json').read_text())['config']['data']['value']['train_weights']])]:
        for i, weights in enumerate(phases):
            unknown = sorted(k for k, v in weights.items() if v > 0 and k not in keys)
            support.append({'source': label, 'phase': i, 'unknown_positive_inventory_keys': unknown})
            ck(label + ' phase ' + str(i) + ' positive keys present in spec inventory', not unknown)
    files = [P, configp, specp, R/'scripts/probe_mixture_identity_cpu.py',
             R/'sources/wandb/hero-fa4sm100-nomask-step146k_meta.json', R/'sources/live_2026_10_07/meta.json']
    result = {'scope': __doc__, 'checks_passed': len(checks), 'checks': checks,
        'runtime': {'python': adapters['platform'].python_version(), 'numpy': ns['np'].__version__,
                    'jax': ns['jax'].__version__, 'backend': ns['jax'].default_backend()},
        'cases': cases, 'archived_support_comparison': support,
        'source_sha256': {str(p.relative_to(R)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'actual_Hero_children': None, 'actual_Hero_token_stream': None, 'actual_training_benefit': None}
    (R/'analysis/mixture_support_cpu.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('Mixture support original-class CPU checks:', len(checks), 'passed')

if __name__ == '__main__':
    asyncio.run(main())
