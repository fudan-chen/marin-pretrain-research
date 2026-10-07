"""Complete original background/thread utility modules with real stdlib queues/threads.
Loader host producer and CPU background subclass are original; mesh and identity IO adapted.
Pending waits are bounded observations. Manual sentinel injection is cleanup only.
"""
import ast, asyncio, contextlib, hashlib, importlib.util, json, pathlib, platform, runpy, sys, threading, time, traceback, types
import tblib
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/loader_background_2026_10_07'
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
# The background module imports this exact original stdlib-only utility module.
for n in ['levanter','levanter.utils']:
 if n not in sys.modules:sys.modules[n]=types.ModuleType(n)
tu=load('levanter.utils.thread_utils',D/'thread_utils.py')
bg=load('marin_background_probe',D/'background_iterable.py');BI=bg.BackgroundIterator
lp=R/'sources/eval_identity_2026_10_06/loader.py'
node=next(n for n in ast.parse(lp.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='_JaxCpuBackgroundIterator')
ns={'BackgroundIterator':BI,'Ex':bg.Ex,'local_cpu_mesh':contextlib.nullcontext}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])),str(lp),'exec'),ns)
CPU=ns['_JaxCpuBackgroundIterator'];a=runpy.run_path(str(R/'scripts/probe_loader_stall_cpu.py'))
checks=[];cases={}
def ck(n,c):assert c,n;checks.append(n)
def next_result(it):
 try:return {'kind':'value','value':next(it)}
 except BaseException as e:return {'kind':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
def pending_next(it):
 entered=threading.Event();done=threading.Event();out={};orig=it.q.get
 def observed(*args,**kwargs):entered.set();return orig(*args,**kwargs)
 it.q.get=observed
 def consume():out.update(next_result(it));done.set()
 t=threading.Thread(target=consume,daemon=True);t.start();assert entered.wait(1);return t,done,out
class AsyncGate:
 def __init__(self):self.enter=threading.Event();self.cancelled=False;self.loop=None;self.release=None
 async def produce(self):
  self.loop=asyncio.get_running_loop();self.release=asyncio.Event();self.enter.set()
  try:await self.release.wait()
  except asyncio.CancelledError:self.cancelled=True;raise
  yield 'released'
 def unblock(self):self.loop.call_soon_threadsafe(self.release.set)
# Normal synchronous drain plus exact traceback transport.
def broken_stream():
 yield 'good0';yield 'good1';raise OSError('artificial queue fault')
it=BI(broken_stream,4);it.thread.join(1);r=[next_result(it) for _ in range(3)];it.stop();cases['buffered_error']=r
ck('Queued valid items precede transported producer error',[x['kind'] for x in r]==['value','value','OSError'] and [x['value'] for x in r[:2]]==['good0','good1'])
ck('Original traceback wrapper retains producer frame','broken_stream' in r[-1]['traceback'])
def fails_to_construct():raise ValueError('artificial constructor fault')
it=BI(fails_to_construct,1);r=next_result(it);it.stop();cases['construction_error']=r;ck('Producer construction error reaches consumer',r['kind']=='ValueError')
async def finite_async():yield 0;yield 1
it=BI(finite_async,4);r=[next_result(it) for _ in range(3)];it.thread.join(1);cases['finite_async_first_end']=r
ck('Original async producer yields ordered values then end',[x['kind'] for x in r]==['value','value','StopIteration'])
# A second call after normal end must not block under Python's iterator contract.
t,done,out=pending_next(it);time.sleep(.15);obs={'pending_after_first_end':not done.is_set(),'producer_alive':it.thread.is_alive(),'queue_size':it.qsize()}
it.q.put(bg._SENTINEL);t.join(1);it.stop();obs['cleanup_result']=out;obs['consumer_alive_after_cleanup']=t.is_alive();cases['repeated_end']=obs
ck('Repeated next after consumed sentinel remains pending in bounded observation',obs['pending_after_first_end'] and not obs['producer_alive'] and obs['queue_size']==0)
ck('Probe-only sentinel cleanup terminates repeated-end consumer',not obs['consumer_alive_after_cleanup'] and out['kind']=='StopIteration')
# stop(wait=True) joins but does not cancel a blocked original async producer.
gate=AsyncGate();it=BI(gate.produce,1);assert gate.enter.wait(1);stopped=threading.Event()
def stop_and_mark():it.stop(wait=True);stopped.set()
st=threading.Thread(target=stop_and_mark,daemon=True);st.start();time.sleep(.15)
obs={'stop_pending_before_release':not stopped.is_set(),'producer_alive_before_release':it.thread.is_alive(),'read_cancelled_before_release':gate.cancelled}
gate.unblock();st.join(1);it.thread.join(1);obs['all_finished_after_release']=not st.is_alive() and not it.thread.is_alive();cases['join_wait']=obs
ck('Stop join waits for pending async producer rather than cancelling',obs['stop_pending_before_release'] and obs['producer_alive_before_release'] and not obs['read_cancelled_before_release'])
ck('External release permits original stop join to finish',obs['all_finished_after_release'])
# A consumer already inside q.get is not woken by stop flag or producer exit.
gate=AsyncGate();it=BI(gate.produce,1);assert gate.enter.wait(1);t,done,out=pending_next(it)
it.stop(wait=False);time.sleep(.15);obs={'consumer_pending_after_stop':not done.is_set(),'producer_alive_after_stop':it.thread.is_alive(),'read_cancelled_after_stop':gate.cancelled}
gate.unblock();it.thread.join(1);time.sleep(.05);obs['producer_finished_after_release']=not it.thread.is_alive();obs['consumer_pending_after_producer_exit']=not done.is_set()
it.q.put(bg._SENTINEL);t.join(1);it.stop();obs['cleanup_consumer_finished']=not t.is_alive();cases['stop_blocked_consumer']=obs
ck('Stop flag does not wake consumer already blocked in empty queue',obs['consumer_pending_after_stop'])
ck('Producer exit after stop still leaves blocked consumer pending',obs['producer_finished_after_release'] and obs['consumer_pending_after_producer_exit'])
ck('Explicit cleanup leaves no blocked consumer or producer',obs['cleanup_consumer_finished'] and not obs['read_cancelled_after_stop'])
# Backpressure stop differs: _enqueue has a timed queue.put poll.
second=threading.Event()
def full_queue():
 yield 0;second.set();yield 1;yield 2
it=BI(full_queue,1);assert second.wait(1);before=it.qsize();start=time.monotonic();it.stop(wait=True);elapsed=time.monotonic()-start
cases['full_queue_stop']={'queued_before_stop':before,'thread_alive_after_stop':it.thread.is_alive(),'elapsed_seconds':elapsed,'next_after_stop':next_result(it)}
ck('Backpressure enqueue stops through queue timeout poll',before==1 and not it.thread.is_alive() and elapsed<2)
ck('Next invoked after stop returns end',cases['full_queue_stop']['next_after_stop']['kind']=='StopIteration')
# Original unbuffered wrapper keeps its exhausted flag, a useful positive control.
it=BI(finite_async,-1);deadline=time.monotonic()+1
while not it.iterator.loop.is_running() and time.monotonic()<deadline:time.sleep(.001)
r=[next_result(it) for _ in range(4)];it.stop();it.iterator.close();cases['unbuffered_repeated_end']=r
ck('Unbuffered original wrapper repeats StopIteration correctly',[x['kind'] for x in r]==['value','value','StopIteration','StopIteration'])
# Real original CPU background wrapper around the original loader host producer.
for fetch in [1,4]:
 h=a['Harness'](a['a']['Schedule'](4),0,fetch=fetch);store=a['FaultStore']();h.dl.data_store=store
 it=CPU(h._produce_batches,4);r=[]
 while True:
  x=next_result(it);r.append(x)
  if x['kind']!='value':break
 it.stop();cases['loader_fault_fetch_'+str(fetch)]={'events':r,'requests':store.requests,'producer_alive_after_stop':it.thread.is_alive()}
ck('Actual background queue retains shallow-loader valid batch then original error',[x['kind'] for x in cases['loader_fault_fetch_1']['events']]==['value','OSError'] and cases['loader_fault_fetch_1']['events'][0]['value']['identities']==[0,1,2,3])
ck('Actual background queue carries deep-loader fault before first batch',[x['kind'] for x in cases['loader_fault_fetch_4']['events']]==['OSError'] and all(not cases['loader_fault_fetch_'+str(f)]['producer_alive_after_stop'] for f in [1,4]))
# Original unbuffered wrapper catches RuntimeError around future.result too.
def fault_async_factory(error_type):
 async def producer():
  raise error_type('artificial async runtime fault')
  yield None
 return producer
for error_type in [RuntimeError,OSError]:
 it=BI(fault_async_factory(error_type),-1);deadline=time.monotonic()+1
 while not it.iterator.loop.is_running() and time.monotonic()<deadline:time.sleep(.001)
 r=next_result(it);it.stop();it.iterator.close();cases['unbuffered_fault_'+error_type.__name__]=r
ck('Original unbuffered RuntimeError becomes apparent stream end',cases['unbuffered_fault_RuntimeError']['kind']=='StopIteration')
ck('Original unbuffered OSError remains an error',cases['unbuffered_fault_OSError']['kind']=='OSError')
it=BI(fault_async_factory(RuntimeError),1);r=next_result(it);it.stop();cases['buffered_runtime_fault']=r
ck('Original buffered RuntimeError is transported without stream-end conversion',r['kind']=='RuntimeError' and r['message']=='artificial async runtime fault')
original_checks_passed=len(checks)
candidate_module=runpy.run_path(str(R/'scripts/background_queue_consumer_candidate.py'))
Candidate=candidate_module['consumer_candidate'](bg)
it=Candidate(finite_async,4);r=[next_result(it) for _ in range(4)];it.stop();cases['candidate_finite_end']=r
ck('Candidate retains normal stream and repeats end without blocking',[x['kind'] for x in r]==['value','value','StopIteration','StopIteration'])
it=Candidate(broken_stream,4);r=[next_result(it) for _ in range(4)];it.stop();cases['candidate_error']=r
ck('Candidate preserves valid FIFO items and original error before terminal end',[x['kind'] for x in r]==['value','value','OSError','StopIteration'] and [x['value'] for x in r[:2]]==['good0','good1'])
ck('Candidate transported error retains original producer frame','broken_stream' in r[2]['traceback'])
gate=AsyncGate();it=Candidate(gate.produce,1);assert gate.enter.wait(1);t,done,out=pending_next(it)
it.stop(wait=False);consumer_finished=done.wait(1);obs={'consumer_finished_before_release':consumer_finished,'consumer_result_before_release':out.copy(),'producer_alive_before_release':it.thread.is_alive(),'read_cancelled_before_release':gate.cancelled}
gate.unblock();it.thread.join(1);t.join(1);it.stop();obs['all_finished_after_release']=not t.is_alive() and not it.thread.is_alive();cases['candidate_stop_consumer']=obs
ck('Candidate blocked consumer observes stop without injected sentinel',consumer_finished and out['kind']=='StopIteration' and obs['producer_alive_before_release'])
ck('Candidate does not claim read cancellation and leaves no threads after release',not obs['read_cancelled_before_release'] and obs['all_finished_after_release'])
files=[D/'background_iterable.py',D/'thread_utils.py',lp,R/'scripts/probe_loader_resume.py',R/'scripts/probe_loader_stall_cpu.py',pathlib.Path(__file__),R/'scripts/background_queue_consumer_candidate.py']
from importlib.metadata import version
out={'scope':__doc__,'checks_passed':len(checks),'original_checks_passed':original_checks_passed,'candidate_checks_passed':len(checks)-original_checks_passed,'checks':checks,'runtime':{'python':platform.python_version(),'tblib':version('tblib')},'cases':cases,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'substitutions':['null CPU mesh','artificial gated async producer','identity/fault store and host batchification','queue get observer delegates unchanged stdlib get','sentinel injected only after pending observations to terminate probe consumers'],'candidate_integrated_into_Marin':False,'candidate_remote_IO_cancellation':False,'actual_Hero_shutdown_incident':None,'actual_full_DataLoader':None,'actual_remote_IO_cancellation':None,'actual_distributed_shutdown':None}
(R/'analysis/background_queue_cpu.json').write_text(json.dumps(out,indent=2)+'\n');print('Original background queue controls:',len(checks),'passed')
