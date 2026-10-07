"""Original initializer/train closure: EMA-only copy, donation-off, and real initial-state IO.
Three-expert synthetic model/loss and injected beta inherited explicitly from V111.
Not full Transformer, device/master offload, actual GPU runtime, or Hero failure.
"""
import ast,dataclasses,functools,hashlib,json,pathlib,tempfile,types,importlib.metadata
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,PartitionSpec as P,AxisType
from candidate_ema_buffer_copy import copy_initial_ema_buffers,STATUS as CANDIDATE_STATUS
R=pathlib.Path(__file__).resolve().parents[1];H=R/'scripts/probe_pending_resume_step.py';source=H.read_text()
# Initialize original-source extraction/adapters without executing or rewriting V111 observations.
ctx={'__file__':str(H),'__name__':'ema_alias_helpers'}
exec(compile(source[:source.index('# Actual original initializer')],str(H),'exec'),ctx)
ns=ctx['ns'];State=ctx['State'];tx=ctx['tx'];mp=ctx['mp'];original_step=ctx['step'];copy=ctx['copy'];host=ctx['host'];equal=ctx['equal'];T=ctx['T'];prior=ctx['prior']
ctx['actual_load']=ns['load_checkpoint'];ctx['__name__']='ema_alias_helpers'
nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ['save','load']]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(H),'exec'),ctx)
ns['Transformer']=types.SimpleNamespace(init=lambda config,key:ctx['fresh_model']());ns['P']=P
mesh=Mesh(np.array(jax.devices()),('data',),axis_types=(AxisType.Explicit,))
def init():
 with jax.set_mesh(mesh):return ns['initial_state'](types.SimpleNamespace(num_layers=1,num_experts=3),optimizer=tx,mp=mp,key=jax.random.PRNGKey(0),ema_beta=.9)
# Donation-off diagnostic changes only the original closure's JIT donation keyword.
factory=next(n for n in ast.parse(T.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_make_train_step')
factory.name='_make_train_step_without_donation'
inner=next(n for n in factory.body if isinstance(n,ast.FunctionDef) and n.name=='train_step')
assert len(inner.decorator_list)==1
kw=next(k for k in inner.decorator_list[0].keywords if k.arg=='donate_argnums');kw.value=ast.Tuple(elts=[],ctx=ast.Load())
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),factory],type_ignores=[])),str(T),'exec'),ns)
no_donation=ns['_make_train_step_without_donation'](tx,mp,z_loss_weight=0.,ema_beta=.9,watch_config=None,offload_opt_state=False,master_param_mode=ns['MasterParamMode'].DEVICE)
batch={'target':jnp.array(2.),'beta':jnp.array([[0.,3.,0.]],jnp.float32)}
checks=[]
def check(n,v):assert v,n;checks.append({'name':n,'passed':True})
def allocation(s):
 groups={};logical=0
 for path,leaf in jax.tree_util.tree_flatten_with_path(s)[0]:
  name=jax.tree_util.keystr(path);size=int(leaf.size*leaf.dtype.itemsize);logical+=size
  ptr=int(leaf.unsafe_buffer_pointer())
  groups.setdefault(ptr,[]).append({'path':name,'bytes':size})
 return {'logical_leaf_bytes':logical,'unique_observed_buffer_bytes':sum(max(x['bytes'] for x in g) for g in groups.values()),'alias_groups':[[x['path'] for x in g] for g in groups.values() if len(g)>1],'leaf_count':sum(map(len,groups.values()))}
def pointers(s):return [int(x.unsafe_buffer_pointer()) for x in jax.tree.leaves(s)]
def result(s,fn):
 before=allocation(s);expected=host(s)
 try:
  out,metrics,_=fn(s,batch);jax.block_until_ready(out);error=None
 except Exception as exc:out=None;metrics=None;error=type(exc).__name__+': '+str(exc)
 deleted=[jax.tree_util.keystr(path) for path,leaf in jax.tree_util.tree_flatten_with_path(s)[0] if leaf.is_deleted()]
 digests={jax.tree_util.keystr(path):{'shape':list(leaf.shape),'dtype':str(leaf.dtype),'sha256':hashlib.sha256(np.asarray(leaf).tobytes()).hexdigest()} for path,leaf in jax.tree_util.tree_flatten_with_path(out)[0]} if out is not None else None
 return {'full_output_leaf_digests':digests,'deleted_input_leaf_paths_after_attempt':deleted,'input_allocation':before,'input_matches_reference':equal(expected,reference),'error':error,'loss':float(metrics['train/loss']) if metrics is not None else None,'output':ctx['describe'](out) if out is not None else None},out
reference=host(init());rows={};outputs={}
s=init();rows['original_shared_donated'],outputs['original_shared_donated']=result(s,original_step)
s=init();param_before=pointers(s.params);opt_before=pointers(s.opt_state)
fixed=copy_initial_ema_buffers(s)
only_ema_changed=pointers(fixed.params)==param_before and pointers(fixed.opt_state)==opt_before
value_preserved=equal(host(fixed),reference)
rows['ema_only_copy_donated'],outputs['ema_only_copy_donated']=result(fixed,original_step)
s=copy(init());rows['all_leaves_copy_donated'],outputs['all_leaves_copy_donated']=result(s,original_step)
s=init();rows['original_shared_no_donation'],outputs['original_shared_no_donation']=result(s,no_donation)
s=init();before_roundtrip=allocation(s)
with tempfile.TemporaryDirectory(prefix='marin-ema-alias-roundtrip-') as tmp:
 paths=ctx['save'](pathlib.Path(tmp)/'step0',s)
 restored=ns['restore_grug_state_from_checkpoint'](init(),checkpoint_search_paths=[tmp],load_checkpoint_setting=True,mesh=None,allow_partial=False,_load_fn=ctx['load'])
 roundtrip_values=equal(host(restored),reference);after_roundtrip=allocation(restored)
 rows['original_initial_state_IO_donated'],outputs['original_initial_state_IO_donated']=result(restored,original_step)
check('Real JAX CPU and eleven concrete original state leaves',jax.default_backend()=='cpu' and all(r['input_allocation']['leaf_count']==11 for r in rows.values()))
check('All five variants retain identical state values before step',all(r['input_matches_reference'] for r in rows.values()))
check('Original shared EMA uses two aliased parameter buffer pairs',len(rows['original_shared_donated']['input_allocation']['alias_groups'])==2 and all(any('ema_params' in p for p in g) and any('params' in p and 'ema_params' not in p for p in g) for g in rows['original_shared_donated']['input_allocation']['alias_groups']))
check('Original donated closure rejects shared input buffer',rows['original_shared_donated']['error'] is not None and 'donate the same buffer twice' in rows['original_shared_donated']['error'])
check('EMA-only copying changes no parameter or optimizer pointers and no values',only_ema_changed and value_preserved)
check('EMA-only copy removes alias and original donated closure advances',rows['ema_only_copy_donated']['error'] is None and rows['ema_only_copy_donated']['input_allocation']['alias_groups']==[] and rows['ema_only_copy_donated']['output']['step']==1)
check('All-leaf copying and EMA-only copying give exactly equal full outputs',equal(outputs['ema_only_copy_donated'],outputs['all_leaves_copy_donated']))
check('Turning off donation accepts original alias with same full numeric output',rows['original_shared_no_donation']['error'] is None and rows['original_shared_no_donation']['input_allocation']['alias_groups']==rows['original_shared_donated']['input_allocation']['alias_groups'] and equal(outputs['original_shared_no_donation'],outputs['ema_only_copy_donated']))
check('Actual original local IO preserves values but removes initial EMA alias',roundtrip_values and before_roundtrip['alias_groups'] and not after_roundtrip['alias_groups'] and len(paths)==11)
check('Restored initial state succeeds in same donated closure with equal full output',rows['original_initial_state_IO_donated']['error'] is None and equal(outputs['original_initial_state_IO_donated'],outputs['ema_only_copy_donated']))
check('In fixture EMA copy changes unique buffers by 24 bytes but logical state bytes unchanged',rows['ema_only_copy_donated']['input_allocation']['unique_observed_buffer_bytes']-rows['original_shared_donated']['input_allocation']['unique_observed_buffer_bytes']==24 and rows['ema_only_copy_donated']['input_allocation']['logical_leaf_bytes']==rows['original_shared_donated']['input_allocation']['logical_leaf_bytes'])
check('All eleven successful output leaf payload digests match independently recorded reference',all(r['full_output_leaf_digests']==rows['ema_only_copy_donated']['full_output_leaf_digests'] and len(r['full_output_leaf_digests'])==11 for r in rows.values() if r['error'] is None))
check('Donation-off successful input buffers remain valid',rows['original_shared_no_donation']['deleted_input_leaf_paths_after_attempt']==[])
no_ema=dataclasses.replace(init(),ema_params=None)
check('Local candidate preserves EMA-disabled state identity',copy_initial_ema_buffers(no_ema) is no_ema and CANDIDATE_STATUS=='candidate_not_integrated')
prev=json.loads((R/'analysis/pending_resume_step_cpu.json').read_text())
check('Archived Hero EMA disabled; all GPU and full-model deployment claims remain out of scope',prev['ema_alias_control']['archived_Hero_ema_beta'] is None)
files=set(prev['source_sha256']);files.update(['analysis/pending_resume_step_cpu.json','scripts/candidate_ema_buffer_copy.py']);files.add(str(pathlib.Path(__file__).relative_to(R)))
output={'scope':__doc__,'candidate_status':CANDIDATE_STATUS,'checks_passed':len(checks),'checks':checks,'runtime':{n:importlib.metadata.version(n) for n in ['jax','jaxlib','numpy','optax','equinox','tensorstore','jmp']},'cases':rows,'EMA_only_copy_preserves_parameter_optimizer_buffers':only_ema_changed,'checkpoint_paths':paths,'roundtrip_before':before_roundtrip,'roundtrip_after':after_roundtrip,'same_full_output_successful_variants':True,'diagnostic_donation_off_ast_change':'Original train closure JIT donate_argnums tuple changed from (0,) to (); no train body changes','source_sha256':{p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in sorted(files)},'substitutions':prev['substitutions']+['Donation-off diagnostic only changes original closure decorator donation tuple; no upstream integration','Array pointer groups measured before invocation on this CPU; unique byte sum is not allocator peak/RSS/HBM measurement'],'actual_Hero_EMA_incident':None,'actual_full_Transformer_execution':None,'actual_GPU_execution':None,'actual_distributed_execution':None,'actual_production_peak_memory':None,'actual_donation_performance_cost':None,'actual_upstream_integration':None}
(R/'analysis/ema_alias_matrix_cpu.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print('Original EMA alias matrix controls',len(checks));print(json.dumps(rows,indent=2))
