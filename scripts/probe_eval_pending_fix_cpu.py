"""Execute pinned author eval wrapper, original runner/core and original regression on CPU.
Tiny Eqx model and recording hooks replace full Transformer/evaluator; no complete training loop.
"""
import ast,abc,dataclasses,functools,inspect,json,pathlib,hashlib,types,typing,enum,importlib.metadata
import jax,jax.numpy as jnp,numpy as np,equinox as eqx
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/eval_fix_2026_10_08'
env=dict(abc=abc,ABC=abc.ABC,inspect=inspect,dataclass=dataclasses.dataclass,dataclasses=dataclasses,field=dataclasses.field,StrEnum=enum.StrEnum,Generic=typing.Generic,S=typing.TypeVar('S'),jax=jax,jnp=jnp,eqx=eqx)
def execute(nodes,path):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(path),'exec'),env)
def extract(path,names):
 nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert len(nodes)==len(names);execute(nodes,path)
extract(D/'callback_core.py',['ProgressEvent','_ignore_progress_event','StepInfo','Callback','LambdaCallback'])
extract(D/'state_adapter.py',['CallbackStateView','_Hook','StateCallbackRunner'])
extract(D/'train.py',['_apply_qb_betas','_callback_step_with_pending_qb'])
real_apply=env['_apply_qb_betas'];apply_calls=[]
def recording_apply(model,beta):
 apply_calls.append(np.asarray(beta).tolist());return real_apply(model,beta)
env['_apply_qb_betas']=recording_apply
nested=next(n for n in ast.walk(ast.parse((D/'train.py').read_text())) if isinstance(n,ast.FunctionDef) and n.name=='with_pending_qb')
factory=ast.parse('''def factory(pending_qb_betas_for_callbacks):
 def placeholder(): pass
 def set_pending(value):
  nonlocal pending_qb_betas_for_callbacks
  pending_qb_betas_for_callbacks=value
 return with_pending_qb,set_pending
''').body[0];factory.body[0]=nested;execute([factory],D/'train.py')
runner_call=next(n for n in ast.walk(ast.parse((D/'train.py').read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript) and isinstance(n.func.value,ast.Name) and n.func.value.id=='StateCallbackRunner')
getters={k.arg:eval(compile(ast.Expression(k.value),str(D/'train.py'),'eval'),env) for k in runner_call.keywords}
Runner=env['StateCallbackRunner'];Event=env['ProgressEvent']
class MLP(eqx.Module):router_bias:jax.Array
class Block(eqx.Module):mlp:MLP
class Stack(eqx.Module):stacked:Block
class Tiny(eqx.Module):
 stacked_blocks:Stack
 weight:jax.Array

def model(bias,w):return Tiny(Stack(Block(MLP(jnp.array([bias],jnp.float32)))),jnp.array([w],jnp.float32))
def bias(m):return np.asarray(m.stacked_blocks.stacked.mlp.router_bias).tolist()
def snapshot(state):return [np.array(x,copy=True) for x in jax.tree.leaves((state.params,state.ema_params,state.pending_qb_betas,state.opt_state))]
def unchanged(state,before):return all(np.array_equal(a,b) for a,b in zip(snapshot(state),before))
def state(step=2,ema=True):return types.SimpleNamespace(step=jnp.array(step),params=model([.3,.2,-.5],2.),ema_params=model([.4,-.6,.2],7.) if ema else None,pending_qb_betas=jnp.array([[1.,3.,2.]]),opt_state=(jnp.array(11),))
def runner():return Runner(**getters)
checks=[];cases={}
def check(n,v):assert v,n;checks.append({'name':n,'passed':True})
# Execute author's actual regression unchanged except supplying the monkeypatch API.
train=types.SimpleNamespace(_apply_qb_betas=real_apply,_callback_step_with_pending_qb=env['_callback_step_with_pending_qb'])
# The wrapper resolves _apply_qb_betas in env, so patch both bindings as module setattr would.
def setattr_adapter(obj,name,value):
 setattr(obj,name,value)
 if obj is train:env[name]=value
saved=env['_apply_qb_betas'];env.update(train=train,SimpleNamespace=types.SimpleNamespace)
extract(D/'test_moe_hero_ep.py',['test_eval_callback_models_apply_pending_query_bias'])
env['test_eval_callback_models_apply_pending_query_bias'](types.SimpleNamespace(setattr=setattr_adapter));env['_apply_qb_betas']=saved
check('Original author string regression executes with pinned original runner/core',True)
s=state();before=snapshot(s);ordinary=[];evals=[]
r=runner();wrap,set_pending=env['factory'](s.pending_qb_betas)
def record_eval(info,*args,**kwargs):
 evals.append({'current':bias(info.model),'ema':bias(info.eval_model),'step':info.step,'next_step':info.next_step,'force':kwargs.get('force',False),'optimizer_identity':info.opt_state is s.opt_state,'weight_current':np.asarray(info.model.weight).tolist(),'weight_ema':np.asarray(info.eval_model.weight).tolist()})
 info.emit_event(Event.EVALUATION_STARTED)
r.add_hook(lambda info:ordinary.append(bias(info.model)));r.add_hook(wrap(record_eval),every=3)
r.run(s,loss=1.,step_duration=.1)
check('Non-due evaluation and ordinary hook perform no QB application',len(apply_calls)==0 and ordinary==[bias(s.params)] and not evals)
# Forced checkpoint-only path without optimization.
events=[]
class EventListener(env['Callback']):
 def on_step(self,info,force=False):pass
 def on_event(self,event):events.append(event)
r.add_hook(EventListener());r.run(s,loss=1.,step_duration=.1,force=True)
check('Forced checkpoint-only callback applies pending to current and EMA',len(apply_calls)==2 and evals[-1]['current']==[[1.,-1.,0.]] and evals[-1]['ema']==[[1.,-1.,0.]])
check('Evaluation preserves distinct model weights optimizer identity and callback clocks',evals[-1]['weight_current']==[2.] and evals[-1]['weight_ema']==[7.] and evals[-1]['optimizer_identity'] and evals[-1]['step']==1 and evals[-1]['next_step']==2 and evals[-1]['force'])
check('Raw input state and ordinary hook remain stored views',unchanged(s,before) and ordinary[-1]==bias(s.params))
check('Replaced StepInfo preserves event handler',events==[Event.EVALUATION_STARTED])
set_pending(jnp.array([[3.,0.,0.]]));r.run(s,loss=1.,step_duration=.1,force=True)
check('Original nested wrapper reads refreshed closure pending not first bound value',evals[-1]['current']==[[-2.,1.,1.]] and evals[-1]['ema']==[[-2.,1.,1.]])
cases['ema_on']=evals.copy()
# Production runner getter uses current params when EMA is off.
s0=state(3,ema=False);r0=runner();wrap0,_=env['factory'](s0.pending_qb_betas);seen=[]
r0.add_hook(wrap0(lambda info,force=False:seen.append((bias(info.model),bias(info.eval_model)))));r0.run(s0,loss=1.,step_duration=0.)
check('EMA-off production getter fallback works with real setter',s0.ema_params is None and seen==[([[1.,-1.,0.]],[[1.,-1.,0.]])])
# Wrapper is applied separately per evaluation hook, not once per step.
r2=runner();w2,_=env['factory'](s.pending_qb_betas);twice=[];n=len(apply_calls)
r2.add_hook(w2(lambda info,force=False:twice.append(bias(info.model))));r2.add_hook(w2(lambda info,force=False:twice.append(bias(info.model))));r2.run(s,loss=1.,step_duration=0.,force=True)
check('Two due evaluation hooks trigger four setter calls with equal views',len(apply_calls)-n==4 and twice==[[[1.,-1.,0.]],[[1.,-1.,0.]]])
cases['multiple_eval_hooks']={'hook_count':2,'setter_calls':len(apply_calls)-n,'views':twice}
# Decorator widens signature; runner forwards force even during ordinary callbacks.
rplain=runner();wp,_=env['factory'](s.pending_qb_betas)
rplain.add_hook(wp(lambda info:None))
try:rplain.run(s,loss=1.,step_duration=0.)
except TypeError as exc:plain_error=str(exc)
check('Wrapped plain hook without force parameter fails even in ordinary run',"unexpected keyword argument 'force'" in plain_error)
cases['plain_hook_contract']={'error':plain_error,'existing_production_hooks_compatible':'cb_tagged_evaluate accepts force; dropless hook accepts kwargs, checked in pinned source'}
# Original runner aborts later hooks on an evaluation exception; original state remains unchanged.
r3=runner();w3,_=env['factory'](s.pending_qb_betas);after=[]
def fail(info,force=False):raise RuntimeError('synthetic evaluator failure')
r3.add_hook(w3(fail));r3.add_hook(lambda info:after.append(True))
try:r3.run(s,loss=1.,step_duration=0.,force=True)
except RuntimeError as exc:error=str(exc)
check('Evaluation error propagates and prevents later hooks in same runner call',error=='synthetic evaluator failure' and not after and unchanged(s,before))
# Zero beta means replacement by zero, not preserve-old-bias.
zero=env['_callback_step_with_pending_qb'](env['StepInfo'](env['CallbackStateView'](s.step,s.params,s.ema_params,s.opt_state),1.,.1),jnp.zeros((1,3)))
check('Zero pending replaces stored bias with zero without mutating input',bias(zero.model)==[[0.,0.,0.]] and unchanged(s,before))
# Execute original tagged callback with recording evaluator and context adapters.
import contextlib
scored=[];tracked=[]
env.update(progress_event_scope=lambda *args:contextlib.nullcontext(),jax_config=types.SimpleNamespace(enable_pgle=lambda x:contextlib.nullcontext()),levanter=types.SimpleNamespace(tracker=types.SimpleNamespace(log=lambda d,step:tracked.append((step,d)))),_join_prefix=lambda a,b:a+'/'+b)
def evaluate(evaluator,m,prefix):
 scored.append(bias(m));return {prefix+'/loss':1.}
env['eval_model']=evaluate
extract(D/'eval.py',['cb_tagged_evaluate'])
re=runner();we,_=env['factory'](s.pending_qb_betas);re.add_hook(we(env['cb_tagged_evaluate'](None,eval_current=True,eval_ema=True)))
n=len(apply_calls);re.run(s,loss=1.,step_duration=0.,force=True);re.run(s,loss=1.,step_duration=0.,force=True)
check('Duplicate forced evaluation suppresses scores after four real QB setter calls',len(scored)==2 and len(apply_calls)-n==4)
cases['duplicate_forced_eval']={'runner_calls':2,'setter_calls':len(apply_calls)-n,'model_scores':len(scored)}
sz=state(0);rz=runner();wz,_=env['factory'](sz.pending_qb_betas);rz.add_hook(wz(env['cb_tagged_evaluate'](None,eval_current=True,eval_ema=True)));n=len(apply_calls);score_count=len(scored)
rz.run(sz,loss=1.,step_duration=0.,force=True)
check('Step-zero forced callback applies QB before negative completed-step score guard',len(apply_calls)-n==2 and len(scored)==score_count)
cases['zero_completed_steps']={'state_step':0,'completed_step':-1,'setter_calls':len(apply_calls)-n,'model_scores':len(scored)-score_count}
rc=runner();wc,_=env['factory'](s.pending_qb_betas);rc.add_hook(wc(env['cb_tagged_evaluate'](None,eval_current=True,eval_ema=False)));n=len(apply_calls);score_count=len(scored)
rc.run(s,loss=1.,step_duration=0.,force=True)
check('Current-only scoring still constructs current and EMA corrected views',len(apply_calls)-n==2 and len(scored)-score_count==1)
cases['current_only_eval']={'setter_calls':len(apply_calls)-n,'model_scores':len(scored)-score_count}
# Inspect acquisition issue status; open does not establish deployment or ancestry.
issue=json.loads((D/'issue_9352.json').read_text());cases['issue_status']={'state':issue['state'],'updated_at':issue['updated_at'],'body_sha256':hashlib.sha256(issue['body'].encode()).hexdigest()}
files=[pathlib.Path(__file__)]+sorted(D.glob('*'))
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'cases':cases,'runtime':{n:importlib.metadata.version(n) for n in ['jax','jaxlib','equinox','numpy']},'backend':jax.default_backend(),'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'substitutions':['Tiny Eqx router hierarchy instead of complete Transformer','Recording evaluation and ordinary hooks; no real Paloma or checkpoint publisher','AST original callback/core functions/classes with type-only imports omitted','Synthetic enclosing closure and setter preserve original nested with_pending_qb; production loop assignment sites inspected, not executed','Minimal monkeypatch.setattr adapter for unchanged author regression'],'actual_GPU_execution':None,'actual_Paloma_loss':None,'actual_production_fix_deployed':None,'actual_full_training_loop_execution':None,'actual_eager_QB_memory_cost':None}
(R/'analysis/eval_pending_fix_cpu.json').write_text(json.dumps(out,indent=2)+'\n');print('Pending evaluation fix controls:',len(checks))
