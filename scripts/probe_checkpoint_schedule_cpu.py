"""Original Checkpointer scheduling methods + original uploaded AsyncManager + original publication callback.
Manual object construction, virtual writer/metadata/retention events; no filesystem writes, deletes or model.
Sources are a fixed Marin snapshot and uploaded JAX dependency, not a proven historical deployed pair.
"""
import ast,concurrent.futures,datetime,enum,hashlib,json,logging,os,pathlib,platform,threading,types
import jax,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];C=R/'sources/checkpoint_commit_2026_10_05/checkpoint.py';AD=R/'scripts/probe_async_manager_cpu.py'
parent=AD.read_text();marker='f,ref=owned_future();';prefix=parent.split(marker)[0]
if len(prefix)==len(parent):raise RuntimeError('adapter definition marker missing')
ad={'__file__':str(AD),'__name__':'schedule_manager_definitions'};exec(compile(prefix,str(AD),'exec'),ad);Manager=ad['Manager'];P=ad['P']
class Retention(enum.StrEnum):TEMPORARY='temporary';PERMANENT='permanent'
checks=[]
def check(n,v):
 if not v:raise RuntimeError(n)
 checks.append({'name':n,'passed':True})
root=ast.parse(C.read_text());cls=next(n for n in root.body if getattr(n,'name',None)=='Checkpointer');names={'on_step','save_checkpoint','request_checkpoint','_consume_checkpoint_request','_get_current_step_save_interval'}
methods=[n for n in cls.body if getattr(n,'name',None) in names]
probeclass=ast.ClassDef(name='Shell',bases=[],keywords=[],body=methods,decorator_list=[])
namespace={'jax':jax,'jnp':np,'broadcast_one_to_all':lambda x:x,'CheckpointRetention':Retention,'os':os,'logger':logging.getLogger(__name__)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),probeclass],type_ignores=[])),str(C),'exec'),namespace);Shell=namespace['Shell']
global_save=next(n for n in root.body if isinstance(n,ast.FunctionDef) and n.name=='save_checkpoint');callback_node=next(n for n in global_save.body if isinstance(n,ast.FunctionDef) and n.name=='my_callback')
def setup(mode,*,time_policy=False):
 c=Shell();c._manager=Manager();c.base_path='/virtual/permanent';c.temporary_base_path='/virtual/temporary';c.keep_last_temporary_checkpoints=1;c.debug=None;c.write_config=None;c._last_save_step=0;c._checkpoint_request_lock=threading.Lock();c._requested_retention=None;c.now=datetime.datetime(2026,10,8)+datetime.timedelta(seconds=101);c._last_save_time=datetime.datetime(2026,10,8);c._dt_now_injection=lambda:c.now;c.save_interval=datetime.timedelta(seconds=100) if time_policy else None;c.step_policies=[types.SimpleNamespace(until=None,every=None)];c.events=[];c.published=[5];c.submissions=[];c.mode=mode
 def record(path):c.events.append(['record_temp',path])
 def prune(keep):
  c.events.append(['prune_attempt',keep])
  if c.mode=='retention_failure':raise RuntimeError('retention failed')
 c._record_temporary_checkpoint=record;c._prune_temporary_checkpoints=prune
 def writer(tree,step,checkpoint_path,manager,commit_callback,is_temporary,debug,write_config):
  manager.wait_until_finished()
  if c.mode=='sync_failure':raise RuntimeError('staging failed')
  c.submissions.append(step);c.events.append(['writer_submit',step])
  def metadata(path,step,temporary,extra):
   c.events.append(['metadata_attempt',step])
   if c.mode=='metadata_failure':raise RuntimeError('metadata failed')
   c.published.append(step);c.events.append(['metadata_published',step])
  e={'progress_logger':None,'checkpoint_path':checkpoint_path,'step':step,'is_temporary':is_temporary,'metadata':None,'_save_metadata':metadata,'commit_callback':commit_callback,'logger':logging.getLogger(__name__)}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[callback_node],type_ignores=[])),str(C),'exec'),e)
  f=concurrent.futures.Future()
  if c.mode=='commit_failure':f.set_exception(RuntimeError('commit failed'))
  else:f.set_result(None)
  manager._add_futures([f]);manager._start_async_commit(e['my_callback'])
 c.writer=writer;return c
# Original method globals share the writer symbol. Each controlled invocation selects its shell's backend.
def step(c,n,force=False):
 namespace['save_checkpoint']=c.writer
 try:c.on_step(tree={'artificial':1},step=n,force=force);return {'returned':True}
 except Exception as e:return {'returned':False,'error':str(e)}
def wait(c):
 try:c._manager.wait_until_finished();return {'returned':True}
 except Exception as e:return {'returned':False,'error':str(e)}
def snapshot(c):return {'last_handoff_step':c._last_save_step,'last_handoff_time':c._last_save_time.isoformat(),'request_pending':None if c._requested_retention is None else str(c._requested_retention),'published_steps':list(c.published),'submitted_steps':list(c.submissions),'events':[list(e) for e in c.events]}
cases={}
for mode in ['sync_failure','commit_failure','metadata_failure','retention_failure','success']:
 c=setup(mode);c.request_checkpoint(Retention.PERMANENT);initial=step(c,10);observed=wait(c);before=snapshot(c);same=step(c,10);after=snapshot(c)
 cases[mode]={'on_step':initial,'wait':observed,'before_same_step':before,'same_step':same,'after_same_step':after}
 # For post-handoff failures, explicit force override after consuming failure can submit again.
 if mode in ['commit_failure','metadata_failure']:
  c.mode='success';forced=step(c,10,True);forced_wait=wait(c);cases[mode]['forced_control']={'on_step':forced,'wait':forced_wait,'state':snapshot(c)}
 if mode=='sync_failure':
  c.mode='success';later=step(c,11);cases[mode]['later_without_new_request']={'on_step':later,'state':snapshot(c)}
clock=setup('commit_failure',time_policy=True);step(clock,1);failure=wait(clock);clock.now+=datetime.timedelta(seconds=1);later=step(clock,2);cases['time_after_failure']={'wait':failure,'next_step':later,'state':snapshot(clock)}
check('Original source method selection complete',len(methods)==5)
check('Synchronous staging failure leaves handoff watermark unchanged but consumes request',cases['sync_failure']['before_same_step']['last_handoff_step']==0 and cases['sync_failure']['before_same_step']['request_pending'] is None and cases['sync_failure']['before_same_step']['submitted_steps']==[])
check('Consumed request not automatically retried without time step or fresh-request trigger',cases['sync_failure']['later_without_new_request']['state']['submitted_steps']==[])
check('Commit and metadata failures keep updated handoff step but no new publication',all(cases[m]['before_same_step']['last_handoff_step']==10 and cases[m]['before_same_step']['published_steps']==[5] and cases[m]['wait']['returned'] is False for m in ['commit_failure','metadata_failure']))
check('Ordinary same-step call does not resubmit failed post-handoff save',all(cases[m]['after_same_step']['submitted_steps']==[10] for m in ['commit_failure','metadata_failure']))
check('Explicit force control after consumed failure resubmits and publishes same virtual path',all(cases[m]['forced_control']['state']['submitted_steps']==[10,10] and cases[m]['forced_control']['state']['published_steps']==[5,10] for m in ['commit_failure','metadata_failure']))
check('Commit and metadata failures never invoke retention in selected callback chain',all(not any(e[0]=='prune_attempt' for e in cases[m]['before_same_step']['events']) for m in ['commit_failure','metadata_failure']))
check('Retention failure follows actual virtual metadata publication and preserves marker despite wait error',cases['retention_failure']['before_same_step']['published_steps']==[5,10] and cases['retention_failure']['wait']['error']=='retention failed' and [e[0] for e in cases['retention_failure']['before_same_step']['events']]==['writer_submit','metadata_attempt','metadata_published','prune_attempt'])
check('Successful permanent save publishes then requests temporary pruning only after metadata',cases['success']['wait']=={'returned':True} and cases['success']['before_same_step']['events'][-2:]==[['metadata_published',10],['prune_attempt',0]])
check('Time-triggered failed handoff suppresses immediate time retry at following update',cases['time_after_failure']['state']['submitted_steps']==[1] and cases['time_after_failure']['state']['last_handoff_step']==1)
check('Single-process CPU only',jax.process_count()==1 and jax.default_backend()=='cpu')
files=[C,P,AD]
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'python':platform.python_version(),'jax':jax.__version__,'numpy':np.__version__,'backend':jax.default_backend()},'cases':cases,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'probe_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'substitutions':['Manual Shell field initialization; five unmodified Checkpointer methods','Original nested global save_checkpoint.my_callback with in-memory metadata and retention backends','Original uploaded AsyncManager class via definitions-only previous probe adapter','Single-process broadcast identity adapter, NumPy boolean array instead of device broadcast','Virtual writer boundary replaces serializer/filesystem; no actual file publication or deletion','Fixed Marin source snapshot plus uploaded JAX dependency, not proven historical deployed pair'],'actual_Hero_failed_save_deduplication':None,'actual_production_checkpoint_deleted':None,'actual_full_checkpoint_restore':None,'actual_training_loss_effect':None}
(R/'analysis/checkpoint_schedule_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Checkpoint source scheduler/callback controls:',len(checks),'passed')
if any(c['wait']['returned'] is False for k,c in cases.items() if k!='time_after_failure'):print('Injected failure boundaries preserved; no actual checkpoint IO')
