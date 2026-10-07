"""Original train closure/route/setter and real local IO with injected nonfinite beta.
Synthetic scalar expert model/squared loss; injected estimator is explicitly not original QB estimation.
Original loss predicate executed; no entire production loop, callbacks, distributed publisher or Hero event.
"""
import ast,dataclasses,hashlib,json,pathlib,tempfile,importlib.metadata
import jax,jax.numpy as jnp,numpy as np
from candidate_qb_health_flags import qb_health_flags,STATUS as GUARD_STATUS
compiled_health=jax.jit(qb_health_flags)
R=pathlib.Path(__file__).resolve().parents[1];H=R/'scripts/probe_pending_resume_step.py';src=H.read_text();ctx={'__file__':str(H),'__name__':'pending_finite_helpers'}
exec(compile(src[:src.index('# Actual original initializer')],str(H),'exec'),ctx)
ns=ctx['ns'];ctx['actual_load']=ns['load_checkpoint'];step=ctx['step'];apply=ctx['apply'];copy=ctx['copy'];host=ctx['host'];forward=ctx['forward'];T=ctx['T']
exec(compile(ast.Module(body=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name in ['save','load']],type_ignores=[]),str(H),'exec'),ctx)
guard=next(n for n in ast.walk(ast.parse(T.read_text())) if isinstance(n,ast.If) and ast.unparse(n.test)=="not jnp.isfinite(metrics['train/loss'])")
predicate=compile(ast.Expression(guard.test),str(T),'eval')
def passes(loss):return not bool(eval(predicate,{'jnp':jnp,'metrics':{'train/loss':loss}}))
def clean(v):
 if isinstance(v,list):return [clean(x) for x in v]
 if isinstance(v,float) and not np.isfinite(v):return 'NaN' if np.isnan(v) else ('+Inf' if v>0 else '-Inf')
 return v
checks=[]
def check(n,v):assert v,n;checks.append({'name':n,'passed':True})
def health(s):
 view=apply(s.params,s.pending_qb_betas);bias=view.stacked_blocks.stacked.mlp.router_bias
 leaves=jax.tree_util.tree_flatten_with_path(s)[0]
 bad=[jax.tree_util.keystr(p) for p,x in leaves if not bool(jnp.all(jnp.isfinite(x)))]
 flags=compiled_health(s.pending_qb_betas,s.params.stacked_blocks.stacked.mlp.router_bias,bias)
 return {'candidate_pending_stored_next_flags':flags.tolist(),'candidate_all_flags_pass':bool(jnp.all(flags)),'step':int(s.step),'nonfinite_state_leaf_paths':bad,'pending':clean(s.pending_qb_betas.tolist()),'pending_finite':bool(jnp.all(jnp.isfinite(s.pending_qb_betas))),'applied_bias':clean(bias.tolist()),'applied_bias_finite':bool(jnp.all(jnp.isfinite(bias))),'stored_bias_finite':bool(jnp.all(jnp.isfinite(s.params.stacked_blocks.stacked.mlp.router_bias))),'params_w':s.params.w.tolist(),'current_prediction':float(forward(s.params)[0]),'next_forward_ids':forward(view)[1].tolist(),'next_forward_prediction':float(forward(view)[0])}
def digest(s):return {jax.tree_util.keystr(p):hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for p,x in jax.tree_util.tree_flatten_with_path(s)[0]}
betas={'finite':[[0.,3.,0.]],'one_nan':[[0.,np.nan,0.]],'one_inf':[[0.,np.inf,0.]],'all_large_finite':[[3e38,3e38,3e38]]}
rows={};all_states={}
for name,beta in betas.items():
 s=ctx['fresh']();stages=[];states=[]
 for i,b in enumerate([beta,[[3.,0.,0.]],[[0.,3.,0.]]]):
  s,metrics,_=step(s,{'target':jnp.array(2.),'beta':jnp.array(b,jnp.float32)});jax.block_until_ready(s)
  row=health(s);row.update(loss=float(metrics['train/loss']),original_loss_guard_passes=passes(metrics['train/loss']));stages.append(row);states.append(copy(s))
 rows[name]={'stages':stages};all_states[name]=states
check('Original loss-only predicate accepts all twelve observed steps',all(r['original_loss_guard_passes'] and np.isfinite(r['loss']) for c in rows.values() for r in c['stages']))
check('First training loss and updated parameters identical across injected beta controls',len({c['stages'][0]['loss'] for c in rows.values()})==1 and all(c['stages'][0]['params_w']==rows['finite']['stages'][0]['params_w'] for c in rows.values()))
check('Injected NaN and Inf first affect output pending not stored bias',all(not rows[n]['stages'][0]['pending_finite'] and rows[n]['stages'][0]['stored_bias_finite'] and rows[n]['stages'][0]['nonfinite_state_leaf_paths']==['.pending_qb_betas'] for n in ['one_nan','one_inf']))
check('Following original step can have nonfinite bias while loss remains finite',all(not rows[n]['stages'][1]['stored_bias_finite'] and rows[n]['stages'][1]['original_loss_guard_passes'] for n in ['one_nan','one_inf']))
check('Finite replacement beta later restores state finiteness without undoing parameter trajectory',all(rows[n]['stages'][2]['nonfinite_state_leaf_paths']==[] and rows[n]['stages'][2]['params_w']!=rows['finite']['stages'][2]['params_w'] for n in ['one_nan','one_inf']))
# Finite large equal beta has exact real-arithmetic centered bias zero; compare actual source/JIT behavior.
large=rows['all_large_finite']['stages'][0]
check('Finite extreme beta can overflow original centering on this CPU',large['pending_finite'] and not large['applied_bias_finite'])
check('Applied-bias preflight adds information beyond pending finite check',all(not(c['stages'][0]['pending_finite'] and c['stages'][0]['applied_bias_finite']) for n,c in rows.items() if n!='finite') and rows['finite']['stages'][0]['pending_finite'] and rows['finite']['stages'][0]['applied_bias_finite'])
# Actual host writer / tree reader / original Grug restore with first-step nonfinite pending.
for name in ['one_nan','one_inf']:
 s=all_states[name][0];before=digest(s)
 with tempfile.TemporaryDirectory(prefix='marin-pending-finite-') as tmp:
  paths=ctx['save'](pathlib.Path(tmp)/'step1',s)
  restored=ns['restore_grug_state_from_checkpoint'](ctx['fresh'](),checkpoint_search_paths=[tmp],load_checkpoint_setting=True,mesh=None,allow_partial=False,_load_fn=ctx['load'])
 after=digest(restored);out,metrics,_=step(restored,{'target':jnp.array(2.),'beta':jnp.array([[3.,0.,0.]],jnp.float32)});jax.block_until_ready(out)
 rows[name]['roundtrip']={'paths':paths,'before_leaf_digests':before,'after_leaf_digests':after,'all_bytes_preserved':before==after,'restored_next_state':health(out),'restored_next_loss':float(metrics['train/loss']),'original_loss_guard_passes':passes(metrics['train/loss'])}
check('Real OCDBT roundtrip retains all eleven leaf payloads including nonfinite pending',all(rows[n]['roundtrip']['all_bytes_preserved'] and len(rows[n]['roundtrip']['paths'])==11 for n in ['one_nan','one_inf']))
check('Restored poisoned pending enters original next step with finite reported loss',all(rows[n]['roundtrip']['original_loss_guard_passes'] and not rows[n]['roundtrip']['restored_next_state']['stored_bias_finite'] for n in ['one_nan','one_inf']))
check('Original optimizer moments stay finite in observed poisoned-bias step',all(not any('opt_state' in p for p in rows[n]['stages'][1]['nonfinite_state_leaf_paths']) for n in ['one_nan','one_inf']))
check('Actual routing choice can differ before all-state finiteness returns',all(rows[n]['stages'][0]['next_forward_ids']!=rows['finite']['stages'][0]['next_forward_ids'] for n in ['one_nan','one_inf']))
check('JIT diagnostic candidate rejects first and second affected states but accepts baseline',all(rows[n]['stages'][0]['candidate_all_flags_pass']==False and rows[n]['stages'][1]['candidate_all_flags_pass']==False for n in ['one_nan','one_inf','all_large_finite']) and all(s['candidate_all_flags_pass'] for s in rows['finite']['stages']))
check('Candidate finite flags do not establish historical update equivalence',all(rows[n]['stages'][2]['candidate_all_flags_pass'] and rows[n]['stages'][2]['params_w']!=rows['finite']['stages'][2]['params_w'] for n in ['one_nan','one_inf','all_large_finite']) and GUARD_STATUS=='candidate_not_integrated')
prev=json.loads((R/'analysis/pending_resume_step_cpu.json').read_text());files=set(prev['source_sha256']);files.update(['scripts/candidate_qb_health_flags.py','analysis/pending_resume_step_cpu.json',str(pathlib.Path(__file__).relative_to(R))])
o={'scope':__doc__,'candidate_status':GUARD_STATUS,'checks_passed':len(checks),'checks':checks,'runtime':{n:importlib.metadata.version(n) for n in ['jax','jaxlib','numpy','optax','equinox','tensorstore','jmp']},'original_guard_test':ast.unparse(guard.test),'cases':rows,'substitutions':prev['substitutions']+['Original loop finite-loss predicate only; no actual production callback/checkpointer invocation or full loop','Injected beta payload deliberately replaces estimator output; not proof original QB estimator emits these values','Nonfinite JSON values stored as explicit strings; leaf digests record actual binary payloads'],'source_sha256':{p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in sorted(files)},'actual_Hero_nonfinite_beta_event':None,'actual_original_QB_estimator_nonfinite_output':None,'actual_production_poisoned_checkpoint':None,'actual_GPU_execution':None,'actual_distributed_numerical_guard':None,'actual_upstream_guard_integration':None,'actual_guard_overhead':None}
(R/'analysis/pending_finite_gap_cpu.json').write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print('Original pending finite-gap controls',len(checks));print(json.dumps(rows,indent=2,allow_nan=False))
