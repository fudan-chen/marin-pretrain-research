"""Original watch dispatcher/config with recording tree-stat substitutes; no actual norms or JAX.
Original diagnostic builder with identity JIT, synthetic gradient and beta helpers.
"""
import ast,dataclasses,hashlib,json,pathlib,types
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/watch_2026_10_06/watch.py';T=R/'sources/scale_2026_10_05/train_hero_ep.py';checks=[];calls=[];grad_calls=[]
class Model:
 def __init__(self,name):self.name=name
state=[(('count',),3),(('mu',),Model('old_mu')),(('nu',),Model('old_nu'))]
def stats(prefix,value,split,**kw):calls.append({'prefix':prefix,'value':getattr(value,'name',value),'flags':kw});return {prefix+'/recorded':'stub_not_norm'}
ns={'Target':str,'dataclass':dataclasses.dataclass,'dataclasses':dataclasses,'cast':lambda t,x:x,'VALID_WATCH_TARGETS':{'grads','params','opt_state','updates'},'summary_statistics_for_tree':stats,'jax':types.SimpleNamespace(tree=types.SimpleNamespace(leaves_with_path=lambda s,**kw:s),jit=lambda f:f),'key_path_to_str':lambda p:'.'.join(p),'replace':dataclasses.replace,'apply_qb_betas':lambda p,b:Model(p.name+'+pending')}
body=[x for x in ast.parse(P.read_text()).body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in {'_validate_watch_targets','compute_watch_stats','WatchConfig'}];future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0);exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+body,type_ignores=[])),str(P),'exec'),ns)
def ck(n,v):assert v,n;checks.append(n)
def compute(targets,**kw):return ns['compute_watch_stats'](watch_targets=targets,include_norms=True,include_per_parameter_norms=True,include_histogram=False,split_scan_layers=True,**kw)
c=ns['WatchConfig']();ck('Default targets include gradients and parameters only',c.watch_targets==['grads','params']);ck('Default interval ten enables watch',c.interval==10 and c.is_enabled);ck('Nonpositive interval disables watch',not dataclasses.replace(c,interval=0).is_enabled);ck('Empty targets disable watch',not dataclasses.replace(c,watch_targets=[]).is_enabled);ck('Per-parameter flag alone cannot enable watch',not dataclasses.replace(c,include_norms=False,include_histograms=False,include_per_parameter_norms=True).is_enabled);ck('Histogram-only config is enabled',dataclasses.replace(c,include_norms=False,include_histograms=True).is_enabled)
try:compute(['updates']);bad=False
except ValueError:bad=True
ck('Requested missing updates raise instead of silent omission',bad)
try:compute(['wrong']);bad=False
except ValueError:bad=True
ck('Unknown watch target is rejected',bad)
calls.clear();compute(['grads','params','updates'],grads=Model('gradient'),params=Model('before'),updates=Model('delta'));ck('Actual dispatcher preserves distinct gradient parameter update prefixes',[x['prefix'] for x in calls]==['grad','params','updates']);ck('Stats options forwarded without proving real norms',all(x['flags']=={'include_histogram':False,'include_norms':True,'include_per_parameter_norms':True} for x in calls))
calls.clear();compute(['opt_state'],opt_state=state,model_tree_type=Model);ck('Model-tree filtered state excludes scalar count from dispatch',[x['prefix'] for x in calls]==['opt_state/mu','opt_state/nu']);calls.clear();compute(['opt_state'],opt_state=state);ck('Unfiltered state forwards scalar count as well',len(calls)==3)
ns['_loss_and_grads']=lambda p,b,mp,z:(grad_calls.append((p.name,b,z)) or ((0,{}),Model('synthetic_gradient')))
body=[x for x in ast.parse(T.read_text()).body if isinstance(x,ast.FunctionDef) and x.name in {'_compute_diagnostic_watch_stats','_make_diagnostic_watch_step'}];exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+body,type_ignores=[])),str(T),'exec'),ns)
try:ns['_make_diagnostic_watch_step'](None,z_loss_weight=0,watch_config=dataclasses.replace(c,watch_targets=['updates']));bad=False
except ValueError:bad=True
ck('Original diagnostic builder rejects updates target',bad)
try:ns['_make_diagnostic_watch_step'](None,z_loss_weight=0,watch_config=dataclasses.replace(c,watch_targets=['opt_state']));bad=False
except ValueError:bad=True
ck('Original diagnostic builder rejects optimizer state target',bad)
calls.clear();step=ns['_make_diagnostic_watch_step'](None,z_loss_weight=0,watch_config=dataclasses.replace(c,watch_targets='params, grads'));step(Model('parameter'), 'same_batch', 'pending');ck('Diagnostic computes synthetic gradient after applying pending beta',grad_calls[-1]==('parameter+pending','same_batch',None));ck('Comma-separated targets normalized by diagnostic builder',[x['prefix'] for x in calls]==['params','grad'])
train=next(x for x in ast.parse(T.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='_make_train_step');call=next(x for x in ast.walk(train) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='compute_watch_stats');fields={x.arg:ast.unparse(x.value) for x in call.keywords};ck('Inline static call passes preupdate state and applied-beta parameter',fields['opt_state']=='opt_state_in' and fields['params']=='qb_params' and fields['updates']=='updates');ck('Inline train watch call has no interval guard in train-step source','interval' not in ast.unparse(train))
# Execute original flat-array tree-stat function with recorded NumPy global_norm.
import contextlib,numpy as np
norm_calls=[]
def global_norm(tree):
 norm_calls.append('tree' if isinstance(tree,dict) else 'leaf')
 values=tree.values() if isinstance(tree,dict) else [tree]
 return np.sqrt(sum(float(np.sum(np.asarray(x,dtype=np.float64)**2)) for x in values))
class Dummy:pass
flatns={'jax':types.SimpleNamespace(tree=types.SimpleNamespace(leaves=lambda x,**kw:list(x.values())),named_scope=lambda x:contextlib.nullcontext()),'jax_utils':types.SimpleNamespace(leaf_key_paths=lambda x,**kw:{k:k for k in x}),'optax':types.SimpleNamespace(global_norm=global_norm),'haliax':types.SimpleNamespace(nn=types.SimpleNamespace(Stacked=Dummy,ArrayStacked=Dummy)),'NamedArray':Dummy,'is_named_array':lambda x:False,'is_jax_array_like':lambda x:isinstance(x,np.ndarray),'cast':lambda t,x:x}
flatns['jax'].Array=np.ndarray
def strict_zip(*xs,strict=False):
 if strict and len({len(x) for x in xs})>1:raise ValueError('Length mismatch')
 return zip(*xs)
flatns['zip']=strict_zip
fp=R/'sources/watch_2026_10_06/tree_stats.py';f=next(x for x in ast.parse(fp.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='summary_statistics_for_tree');exec(compile(ast.fix_missing_locations(ast.Module(body=[future,f],type_ignores=[])),str(fp),'exec'),flatns);flat={'a':np.ones(2),'b':np.ones(3)}
result=flatns[f.name]('params',flat,False,include_norms=False,include_histogram=False)
ck('Original flat tree stats still computes total norm with include_norms false',norm_calls==['tree'] and 'params/norm/total' in result)
norm_calls.clear();result=flatns[f.name]('params',flat,False,include_norms=True,include_per_parameter_norms=False)
ck('Disabled per-parameter outputs still construct Python norm calls before filtering',norm_calls==['leaf','leaf','tree'] and list(result)==['params/norm/total'])
ck('Flat total value agrees with independent element count',float(result['params/norm/total'])==np.sqrt(5))
ck('Tree-stat total field has unconditional source assignment',"to_log[f'{prefix}/norm/total']" in ast.unparse(f))
out={'checks_passed':len(checks),'checks':checks,'scope':__doc__,'inline_static_fields':fields,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,T,fp]},'flat_tree_norm_scope':'original flat-array stat function, recorded NumPy norm substitute; no scan split, histogram or distributed norm', 'actual_tree_norm_result':None,'actual_JAX_execution':None,'actual_watch_overhead':None,'historical_execution_sha':None};(R/'analysis/watch_probe.json').write_text(json.dumps(out,indent=2)+'\n');print('Watch dispatch/config checks:',len(checks))
