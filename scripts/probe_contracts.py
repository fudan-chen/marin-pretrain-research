# -*- coding: utf-8 -*-
"""Execute selected original pure helpers with explicit NumPy/environment/I/O substitutes.

No module imports from the archived training code, JAX computation, checkpoint reads,
unpickling, or hardware benchmarks. AST function bodies are preserved unchanged.
"""
import ast
import hashlib
import json
import logging
import pathlib
import types
import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
D = R / 'sources/contracts_2026_10_05'
checks = []

def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)

def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise AssertionError('Missing requested helper')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return [{'name': n.name, 'first_line': n.lineno, 'last_line': n.end_lineno} for n in nodes]

# Original reduction helper: substitute array operations only.
ns = {'jnp': np}
functions = {'api.py': extract(D/'api.py', ['_apply_reduction'], ns)}
losses = np.array([2., 6., 100.])
weights = np.array([1., .5, 0.])
weighted = ns['_apply_reduction'](losses, None, weights)
single = float(ns['_apply_reduction'](losses, 'mean', weights))
double = float(np.sum(weighted*weights)/np.sum(weights))
check('Fractional unreduced weights applied in original helper', np.array_equal(weighted, [2.,3.,0.]))
check('Single and double weighting diverge', single == 10/3 and double == 7/3)
binary = np.array([1., 1., 0.])
check('Binary masks conceal the repeated-weight distinction', np.array_equal(binary**2, binary))
check('Unweighted mean returns ordinary NLL mean', ns['_apply_reduction'](losses, 'mean', None) == 36)
check('Weighted sum preserves numerator', ns['_apply_reduction'](losses, 'sum', weights) == 5)
with np.errstate(divide='ignore', invalid='ignore'):
    check('Zero-weight mean guarded to zero', ns['_apply_reduction'](losses, 'mean', np.zeros(3)) == 0)
try:
    ns['_apply_reduction'](losses, 'bad', weights)
except ValueError:
    check('Invalid reduction rejects', True)
else:
    raise AssertionError('Invalid reduction accepted')

# Original environment precedence helpers: isolated fake environment, real os untouched.
environment = {}
env = {'os': types.SimpleNamespace(environ=environment), '_FAST_BWD_ENV_VAR': 'LEVANTER_CE_XLA_FAST_BWD',
       '_FAST_BWD_LIBRARY_DEFAULT': False, '_TRUTHY': ('1','true','yes','on'),
       '_FALSY': ('0','false','no','off')}
functions['xla.py'] = extract(D/'xla.py', ['_fast_backward_env_override','_resolve_fast_backward'], env)
resolver_rows=[]
for raw in [None, '1', '0', ' TRUE ', 'off', 'fasle']:
    for requested in [None, True, False]:
        environment.clear()
        if raw is not None:
            environment['LEVANTER_CE_XLA_FAST_BWD']=raw
        try:
            result=env['_resolve_fast_backward'](requested)
            expected=(False if requested is None else requested) if raw is None else raw.strip().lower() in env['_TRUTHY']
            check('Original backward precedence '+repr((raw,requested)), raw!='fasle' and result==expected)
            resolver_rows.append({'environment':raw,'call_site':requested,'resolved':result,'error':None})
        except ValueError:
            check('Invalid environment rejects '+repr(requested),raw=='fasle')
            resolver_rows.append({'environment':raw,'call_site':requested,'resolved':None,'error':'ValueError'})

# Original restore policy and sorting, with candidate scan, loader, and barrier replaced.
io={'entries':[], 'fail':set(), 'calls':[], 'barriers':[]}
def load(**kw):
    io['calls'].append(kw['candidate'])
    if kw['candidate'] in io['fail']:
        raise FileNotFoundError(kw['candidate'])
    return {'restored':kw['candidate']}
restore_ns={'_scan_checkpoint_root':lambda root:io['entries'], 'load_checkpoint':load,
            'load_grug_checkpoint':load,'barrier_sync_named':lambda *a,**k:io['barriers'].append((a,k)),
            'RESTORE_COMPLETE_BARRIER':'grug_checkpoint_restore_complete','RESTORE_BARRIER_TIMEOUT':2400,
            'logger':logging.getLogger('contract-probe')}
logging.getLogger('contract-probe').addHandler(logging.NullHandler())
functions['checkpointing.py']=extract(D/'checkpointing.py', ['_checkpoint_candidates','restore_grug_state_from_checkpoint'],restore_ns)
restore_cases=[]
specs=[('disabled',False,[],set(),False), ('optional_empty',None,[],{'root'},False),
       ('required_empty',True,[],{'root'},True),
       ('optional_existing_unreadable',None,[(20,'b','root/20')],{'root','root/20'},True),
       ('fallback_older',True,[(10,'a','root/10'),(20,'b','root/20')],{'root/20'},False),
       ('metadata_step_beats_time',True,[(10,'z','root/10'),(20,'a','root/20')],set(),False)]
for name, setting, entries, fails, should_error in specs:
    io.update(entries=entries,fail=fails,calls=[],barriers=[])
    try:
        result=restore_ns['restore_grug_state_from_checkpoint']({'initial':True},checkpoint_search_paths=['root'],
                 load_checkpoint_setting=setting, mesh=None,allow_partial=False)
        error=None
    except FileNotFoundError:
        result=None;error='FileNotFoundError'
    check('Original restore policy '+name,(error is not None)==should_error)
    if name=='fallback_older':
        check('Unreadable newest candidate falls back to older loadable checkpoint',result=={'restored':'root/10'} and io['calls']==['root/20','root/10'])
    if name=='metadata_step_beats_time':
        check('Candidate ordering prioritizes step above timestamp',result=={'restored':'root/20'})
    if name=='disabled':
        check('Explicit disabled resume performs no I/O substitute calls',io['calls']==[])
    check('Restore barrier is only called after successful substituted load '+name,len(io['barriers'])==(1 if result and 'restored' in result else 0))
    restore_cases.append({'case':name,'setting':setting,'candidate_metadata':entries,'result':result,'error':error,'load_calls':io['calls'],'barrier_calls':len(io['barriers'])})

# Arithmetic illustration: independently bounded per-layer int32 count, overflow across layers.
per_layer=4096*4096*8
counts=np.full(48,per_layer,dtype=np.int32)
wide=int(np.sum(counts,dtype=np.int64));narrow=int(np.sum(counts,dtype=np.int32))
check('Per-layer assignment count fits int32',per_layer < 2**31)
check('Total assignment count exceeds int32',wide==6442450944 and narrow==-2147483648)
train=(R/'sources/scale_2026_10_05/train_hero_ep.py').read_text()
model=(D/'model.py').read_text();api=(D/'api.py').read_text();eval_code=(R/'sources/scale_2026_10_05/eval.py').read_text()
check('Fused API applies weights after adding final-logit z penalty',api.index('loss = loss + logsumexp_weight * (lse**2)') < api.index('reduced_loss = _apply_reduction(loss, reduction, weight)'))
check('Evaluator callback returns model-weighted positions and original weights', 'return per_pos_loss, per_pos_weight, per_pos_token_id' in train and 'weighted_loss = losses * weights' in eval_code)
check('Router z loss explicitly has logging-only metric name','train/router/z_loss_logging_only' in model and 'loss = cross_entropy_loss' in model)
check('Training passes final-logit z-loss and evaluation disables it','logsumexp_weight=z_loss' in train and 'logsumexp_weight=None' in train)
result={'pinned_revision':'84869ae8c91ffe64e9f761c5bd714542eb1876e0','status':'local_source_helper_probe_only',
        'checks_passed':len(checks),'checks':checks,'extracted_original_functions':functions,
        'substitutions':['NumPy instead of jax.numpy in reduction helper','isolated fake os.environ',
                        'fake candidate scan, checkpoint loader and restore barrier'],
        'mask_example':{'nll':losses.tolist(),'weights':weights.tolist(),'weighted_unreduced':weighted.tolist(),
                        'single_weight_mean':single,'double_weight_mean':double,'binary_current_scope_not_verified':True},
        'backward_precedence':resolver_rows,'restore_cases':restore_cases,
        'counter_example':{'per_layer':per_layer,'layers':48,'int64_total':wide,'int32_total':narrow,
                           'actual_drop_count':None},
        'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*.py'))},
        'not_verified':['actual historical execution SHA/environment','JAX/autodiff/GPU kernel behavior',
                        'fractional weights in Hero validation inputs','checkpoint contents and distributed barrier behavior',
                        'actual production counter overflow or numerical root cause']}
(R/'analysis/implementation_contract_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('Original helper and arithmetic checks:',len(checks))
