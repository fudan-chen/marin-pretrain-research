"""Execute the archived main-loop try/except/else/finally AST with synthetic train steps,
callbacks and checkpoint handoff recorder. No GPU, autodiff, real callbacks or checkpoint I/O.
"""
import ast,collections,contextlib,hashlib,json,pathlib,time,types
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/scale_2026_10_05/train_hero_ep.py';tree=ast.parse(P.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_run_grug_local');loop=next(x for x in ast.walk(fn) if isinstance(x,ast.Try) and any(isinstance(y,ast.While) and 'current_step < stop_step' in ast.unparse(y.test) for y in x.body));code=compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),str(P),'exec');checks=[];cases={}
def ck(n,v):assert v,n;checks.append(n)
class Iterator:
 this_load_time=0
 def __iter__(self):return self
 def __next__(self):return 'synthetic_batch'

def run(name,stop=1,kind='clean',callback_fail=False,save_fail=False,diag_fail=False):
 events=[];calls=[];saves=[];logs=[];traincalls=[]
 state=types.SimpleNamespace(step=0,params=np.array([1.],np.float32),pending_qb_betas=None)
 def train(s,b):
  traincalls.append(s.step);loss=float('nan') if kind=='bad_loss' or (kind=='delayed_bad_loss' and not np.isfinite(s.params).all()) else (9.14 if kind=='finite_collapse' and s.step>0 else 2.)
  p=np.array([np.nan],np.float32) if kind in ('bad_update','delayed_bad_loss') else s.params/7 if kind=='finite_collapse' else s.params+.1
  return types.SimpleNamespace(step=s.step+1,params=p,pending_qb_betas=None),{'train/loss':loss},None
 def callbacks_run(s,**kw):
  calls.append({'step':s.step,'force':kw.get('force',False),'parameter_finite':bool(np.isfinite(s.params).all())})
  if callback_fail:raise RuntimeError('synthetic callback failure')
 def on_step(**kw):
  s=kw['tree'];saves.append({'step':kw['step'],'force':kw.get('force',False),'parameter_finite':bool(np.isfinite(s.params).all())})
  if save_fail:raise RuntimeError('synthetic checkpoint handoff failure')
 def diagnostic(*a):raise RuntimeError('synthetic diagnostic failure')
 callbacks=types.SimpleNamespace(ProgressEvent=types.SimpleNamespace(TRAIN_STEP_STARTED='started',TRAIN_STEP_FINISHED='finished',TRAINING_FINISHED='training_finished',CHECKPOINT_STARTED='save_started',CHECKPOINT_FINISHED='save_finished'),progress_event_scope=lambda *a:contextlib.nullcontext())
 env={'state':state,'current_step':0,'stop_step':stop,'last_loss':0.,'last_step_duration':0.,'time':time,'config':types.SimpleNamespace(trainer=types.SimpleNamespace(gc_interval=None)),'gc_start_step':-1,'iterator':Iterator(),'watch_config':types.SimpleNamespace(is_enabled=diag_fail,interval=1),'diagnostic_watch_step':diagnostic if diag_fail else None,'jax':types.SimpleNamespace(profiler=types.SimpleNamespace(TraceAnnotation=lambda *a:contextlib.nullcontext()),block_until_ready=lambda x:x),'jnp':np,'state_callbacks':types.SimpleNamespace(emit_event=events.append,run=callbacks_run),'callbacks':callbacks,'train_step':train,'levanter':types.SimpleNamespace(tracker=types.SimpleNamespace(log=lambda data,step:logs.append((step,data)))),'MOE_DROPPED_ASSIGNMENTS_METRIC':'synthetic_drop_absent','checkpointer':types.SimpleNamespace(on_step=on_step,wait_until_finished=lambda:events.append('wait_save')),'logger':types.SimpleNamespace(exception=lambda msg:events.append('fatal_logged'))}
 error=None
 try:exec(code,env)
 except BaseException as e:error=str(e)
 result={'error':error,'train_call_steps':traincalls,'state_step_after':env['state'].step,'state_parameter_finite_after':bool(np.isfinite(env['state'].params).all()),'events':events,'callback_calls':calls,'checkpoint_handoffs':saves};cases[name]=result;return result
clean=run('clean');ck('Clean one-step path performs periodic and forced final callbacks',[x['force'] for x in clean['callback_calls']]==[False,True]);ck('Clean one-step path hands off periodic and forced final saves',[x['force'] for x in clean['checkpoint_handoffs']]==[False,True]);ck('Clean final path waits for recorded save completion','wait_save' in clean['events'])
badloss=run('bad_loss',kind='bad_loss');ck('Loss rejection occurs after next-state step is assigned',badloss['state_step_after']==1 and badloss['error'] is not None);ck('Nonfinite loss avoids callback and checkpoint handoffs',badloss['callback_calls']==badloss['checkpoint_handoffs']==[]);ck('Train finished event precedes loss rejection',badloss['events'][:2]==['started','finished']);ck('Fatal loss path skips forced save but emits training finished',badloss['events'][-2:]==['fatal_logged','training_finished'] and 'wait_save' not in badloss['events'])
bad=run('finite_loss_bad_update',kind='bad_update');ck('Finite loss does not reject synthetic NaN parameter update',bad['error'] is None and not bad['state_parameter_finite_after']);ck('Unchecked finite-loss next state reaches recorded checkpoint handoff',len(bad['checkpoint_handoffs'])==2 and all(not x['parameter_finite'] for x in bad['checkpoint_handoffs']))
late=run('next_step_detects_bad_state',stop=2,kind='delayed_bad_loss');ck('Delayed loss failure first appears on synthetic second step',late['state_step_after']==2 and late['train_call_steps']==[0,1] and late['error'] is not None);ck('First bad state already reached save recorder before later rejection',len(late['checkpoint_handoffs'])==1 and late['checkpoint_handoffs'][0]['step']==1 and not late['checkpoint_handoffs'][0]['parameter_finite'])
collapse=run('finite_collapse_with_finite_high_loss',stop=2,kind='finite_collapse');ck('Finite high loss and finite shrinking parameter do not trip finite guard',collapse['error'] is None and collapse['state_parameter_finite_after'] and collapse['state_step_after']==2);ck('Finite collapse still reaches periodic and final handoffs',len(collapse['checkpoint_handoffs'])==3 and all(x['parameter_finite'] for x in collapse['checkpoint_handoffs']))
cb=run('callback_raises',callback_fail=True);ck('Callback failure skips checkpoint and forced-final callbacks',len(cb['callback_calls'])==1 and cb['checkpoint_handoffs']==[] and cb['error'] is not None)
save=run('save_handoff_raises',save_fail=True);ck('Checkpoint handoff failure is not followed by forced retry',len(save['checkpoint_handoffs'])==1 and save['error'] is not None and 'wait_save' not in save['events'])
diag=run('diagnostic_raises',diag_fail=True);ck('Diagnostic failure occurs before synthetic optimizer call',diag['train_call_steps']==[] and diag['state_step_after']==0 and diag['checkpoint_handoffs']==[])
ck('Reviewed loop explicitly checks only train/loss for finiteness',sum(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='isfinite' for x in ast.walk(loop))==1)
out={'checks_passed':len(checks),'checks':checks,'scope':__doc__,'source_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'source_loop_lines':[loop.lineno,loop.end_lineno],'cases':cases,'actual_GPU_behavior':None,'actual_callback_guard_coverage':None,'actual_checkpoint_commit':None,'actual_checkpoint_restore':None,'actual_historical_failure':None};(R/'analysis/failure_loop_probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Main loop synthetic control-flow checks:',len(checks))
