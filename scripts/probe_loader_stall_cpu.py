"""Original loader host flow, watchdog and next warnings with bounded fault adapters.
Real asyncio tasks and event-gated waits, unchanged ten-second watchdog threshold.
No full background queue, device batchification, real token store or training.
"""
import ast, asyncio, hashlib, json, logging, pathlib, runpy, threading, time, types
R=pathlib.Path(__file__).resolve().parents[1]
a=runpy.run_path(str(R/'scripts/probe_loader_resume.py'))
P=a['LP'];cl=next(n for n in ast.parse(P.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DataLoaderIterator')
methods=[n for n in cl.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ['run_and_report_slowness','__next__']]
ns={'asyncio':asyncio,'logging':logging,'time':time,'logger':logging.getLogger('loader-probe'),'BackgroundIterator':type('BackgroundIterator',(),{})}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+methods,type_ignores=[])),str(P),'exec'),ns)
class Harness(a['Harness']):run_and_report_slowness=ns['run_and_report_slowness']
class Capture(logging.Handler):
 def __init__(self):super().__init__(logging.WARNING);self.messages=[];self.event=asyncio.Event()
 def emit(self,r):self.messages.append(r.getMessage());self.event.set()
class GateStore(a['Store']):
 def __init__(self,mode):super().__init__(length=100 if mode=='length' else None);self.mode=mode;self.entered=asyncio.Event();self.release=asyncio.Event();self.cancelled=False
 async def async_len(self):
  if self.mode=='length':self.entered.set();await self.release.wait()
  return self.length
 async def get_batch(self,indices):
  self.requests.append(list(indices));self.entered.set()
  if self.mode=='read':
   try:await self.release.wait()
   except asyncio.CancelledError:self.cancelled=True;raise
  return [{'identity':i} for i in indices]
class FaultStore(a['Store']):
 async def get_batch(self,indices):
  self.requests.append(list(indices))
  if 4 in indices:raise OSError('artificial unreadable identity four')
  return [{'identity':i} for i in indices]
async def main():
 checks=[];cases={};cap=Capture();root=logging.getLogger();root.addHandler(cap)
 def ck(n,c):assert c,n;checks.append(n)
 try:
  h=Harness(a['Schedule'](4),0,fetch=1);s=GateStore('read');h.dl.data_store=s
  g=h._produce_batches();t=asyncio.create_task(g.__anext__());await s.entered.wait()
  await asyncio.wait_for(cap.event.wait(),12)
  ck('Original ten-second read watchdog warns while fetch remains pending',not t.done() and any('10.0 seconds' in x for x in cap.messages))
  ck('Warning is not cancellation or automatic retry',not s.cancelled and len(s.requests)==1)
  s.release.set();v=await t;await g.aclose();await asyncio.sleep(0)
  ck('Release returns original requested identities',v['identities']==[0,1,2,3])
  cases['read_wait']={'warnings':cap.messages[:],'request':s.requests,'returned':v,'automatic_timeout':False}
  cap.messages.clear();cap.event.clear()
  h=Harness(a['Schedule'](4),0,fetch=1);s=GateStore('length');h.dl.data_store=s
  g=h._produce_batches();t=asyncio.create_task(g.__anext__());await s.entered.wait()
  await asyncio.sleep(10.3)
  ck('Finite length await emits no read-watchdog warning in bounded interval',not t.done() and cap.messages==[])
  ck('Length wait occurs before batch read',s.requests==[])
  cases['length_wait']={'observation_seconds':10.3,'pending_before_release':not t.done(),'warnings_before_release':cap.messages[:],'requests_before_release':s.requests[:]}
  s.release.set();v=await t;await g.aclose();await asyncio.sleep(0)
  ck('Release finite length allows normal original batch',v['identities']==[0,1,2,3])
  cap.messages.clear();cap.event.clear()
  h=Harness(a['Schedule'](4),0,fetch=1);s=GateStore('read');h.dl.data_store=s
  g=h._produce_batches();t=asyncio.create_task(g.__anext__());await s.entered.wait();t.cancel()
  cancelled=False
  try:await t
  except asyncio.CancelledError:cancelled=True
  await g.aclose();await asyncio.sleep(0)
  ck('Explicit caller cancellation reaches gated read',cancelled and s.cancelled)
  cases['explicit_cancellation']={'caller_cancelled':cancelled,'read_cancelled':s.cancelled}
  for fetch in [4,1]:
   h=Harness(a['Schedule'](4),0,fetch=fetch);s=FaultStore();h.dl.data_store=s;g=h._produce_batches();returned=[];error=None
   try:
    returned.append(await g.__anext__())
    returned.append(await g.__anext__())
   except OSError as e:error=str(e)
   finally:await g.aclose()
   cases['fault_fetch_'+str(fetch)]={'returned':returned,'requests':s.requests,'error':error}
  ck('Deep fetch later fault prevents earlier first yield',cases['fault_fetch_4']['returned']==[] and cases['fault_fetch_4']['requests']==[list(range(16))])
  ck('Shallow fetch yields earlier valid batch before same later fault',len(cases['fault_fetch_1']['returned'])==1 and cases['fault_fetch_1']['returned'][0]['identities']==[0,1,2,3])
  ck('Both fetch depths propagate original fault',all(cases['fault_fetch_'+str(f)]['error']=='artificial unreadable identity four' for f in [4,1]))
  cap.messages.clear();cap.event.clear()
  class Blocking:
   def __init__(self,fail=False):self.enter=threading.Event();self.release=threading.Event();self.fail=fail
   def __next__(self):
    self.enter.set()
    if not self.release.wait(3):raise RuntimeError('probe release deadline')
    if self.fail:raise OSError('artificial next failure')
    return 'ready'
  for fail in [False,True]:
   b=Blocking(fail);obj=types.SimpleNamespace(_batches=b,dl=types.SimpleNamespace(fetch_batch_size=1,max_buffered_batches=0))
   t=asyncio.create_task(asyncio.to_thread(ns['__next__'],obj));await asyncio.to_thread(b.enter.wait,1)
   await asyncio.sleep(.65);before=cap.messages[:];pending=not t.done();b.release.set();err=None;v=None
   try:v=await t
   except OSError as e:err=str(e)
   after=cap.messages[:]
   cases['sync_next_'+str(fail)]={'pending_before_release':pending,'warnings_before_release':before,'warnings_after_release':after,'result':v,'error':err}
   cap.messages.clear();cap.event.clear()
  ck('Synchronous next warning is absent while call blocks',all(cases['sync_next_'+str(f)]['pending_before_release'] and cases['sync_next_'+str(f)]['warnings_before_release']==[] for f in [False,True]))
  ck('Synchronous successful slow next logs after returning',any('Data loader stalled' in x for x in cases['sync_next_False']['warnings_after_release']) and cases['sync_next_False']['result']=='ready')
  ck('Synchronous failed next bypasses post-return stall warning',cases['sync_next_True']['warnings_after_release']==[] and cases['sync_next_True']['error']=='artificial next failure')
 finally:root.removeHandler(cap)
 files=[P,a['SP'],R/'scripts/probe_loader_resume.py',pathlib.Path(__file__)]
 out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime_python':a['platform'].python_version(),'cases':cases,'original_methods':{n.name:[n.lineno,n.end_lineno] for n in methods},'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'actual_Hero_stall':None,'actual_training_resume':None,'actual_full_background_queue':None}
 (R/'analysis/loader_stall_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
 print('Original loader stall and fault controls:',len(checks),'passed')
if __name__=='__main__':asyncio.run(main())
