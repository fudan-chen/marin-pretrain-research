"""Execute three original async loader methods with an identity store and host layout.
No complete DataLoader, JAX batchification, background thread, real tokens or restore.
"""
import ast,asyncio,contextlib,dataclasses,hashlib,json,logging,pathlib,platform,sys,types,typing
if sys.version_info < (3,10):raise SystemExit("Original loader uses zip(strict=...); Python 3.10+ required")
ROOT=pathlib.Path(__file__).resolve().parents[1]
LP=ROOT/'sources/eval_identity_2026_10_06/loader.py'
SP=ROOT/'sources/deepening_2026_10_04/schedule_production.py'
ns={'__name__':__name__};exec(compile('from __future__ import annotations\n'+SP.read_text(),str(SP),'exec'),ns)
Schedule=ns['BatchSchedule'];Step=ns['ScheduleStep']
tree=ast.parse(LP.read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DataLoaderIterator')
names=['_produce_batches','_dataset_get_available_batch_number','_do_retrieve_batch_of_batches']
methods=[n for n in cl.body if isinstance(n,ast.AsyncFunctionDef) and n.name in names]
Ex=typing.TypeVar('Ex')
@dataclasses.dataclass
class Batch(typing.Generic[Ex]):
 index:int
 global_data_offset:int
 global_size:int
 data_by_local_index:dict
scope={'dataclasses':dataclasses,'local_cpu_mesh':contextlib.nullcontext,'logger':logging.getLogger(__name__),'_Batch':Batch}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+methods,type_ignores=[])),str(LP),'exec'),scope)
class Store:
 def __init__(self,length=None):self.length=length;self.requests=[]
 def is_finite(self):return self.length is not None
 async def async_len(self):return self.length
 async def get_batch(self,indices):
  self.requests.append(list(indices));return [{'identity':i} for i in indices]
class Harness:
 _produce_batches=scope['_produce_batches']
 _dataset_get_available_batch_number=scope['_dataset_get_available_batch_number']
 _do_retrieve_batch_of_batches=scope['_do_retrieve_batch_of_batches']
 def __init__(self,schedule,start,fetch=4,length=None,pad=False,duplicate=False):
  store=Store(length);self._start_from_batch=start
  def layout(i):
   r=range(schedule.batch_size_at_step(i));return {'host0':r,'host1':r} if duplicate else {'host0':r}
  self.dl=types.SimpleNamespace(scheduler=schedule,data_store=store,fetch_batch_size=fetch,_pad_final_batch=pad,_allow_non_divisible_batch_size=False,local_data_indices_by_device_for_step=layout)
 async def run_and_report_slowness(self,coro,description):return await coro
 def _batchify_local_data(self,b):return {'step':b.index,'offset':b.global_data_offset,'size':b.global_size,'identities':[b.data_by_local_index[i]['identity'] for i in sorted(b.data_by_local_index)]}
async def first(h):
 it=h._produce_batches();x=await it.__anext__();await it.aclose();return x
async def collect(h):return [x async for x in h._produce_batches()]
async def main():
 old=Schedule([Step(0,4),Step(3,8)]);changed=Schedule(8);future=Schedule([Step(0,4),Step(3,8),Step(4,6)])
 checks=[]
 def check(label,value):
  assert value,label;checks.append(label)
 h=Harness(old,0);first_batch=await first(h)
 check('Prefetch reads through index nineteen before first batch is yielded',max(h.dl.data_store.requests[0])==19 and first_batch['identities']==list(range(4)))
 resumed=Harness(old,1);next_batch=await first(resumed)
 check('Resume follows completed step, not prior fetched high-water',next_batch['identities']==list(range(4,8)))
 a=await first(Harness(old,4));b=await first(Harness(changed,4));c=await first(Harness(future,4))
 check('Rewritten historical schedule changes retrieved identities',a['identities']==list(range(20,28)) and b['identities']==list(range(32,40)))
 check('Future-only schedule keeps next offset but changes batch grouping',c['offset']==20 and c['identities']==list(range(20,26)))
 single=await first(Harness(old,4,fetch=1))
 check('Fetch depth does not change first returned batch in identity-store control',single==a)
 dup=Harness(old,4,fetch=1,duplicate=True);d=await first(dup)
 check('Overlapping artificial device indices are deduplicated per batch',d==a and len(dup.dl.data_store.requests[0])==8)
 padded=Harness(old,0,length=22,pad=True);p=await collect(padded)
 check('Finite partial last batch retains two real identities',p[-1]['step']==4 and p[-1]['size']==2 and p[-1]['identities']==[20,21])
 dropped=Harness(old,0,length=22,pad=False);dr=await collect(dropped)
 check('Finite no-pad branch omits partial final batch',len(dr)==4 and dr[-1]['identities']==list(range(12,20)))
 exact=await collect(Harness(old,0,length=20,pad=True))
 check('Exact endpoint produces no empty extra batch',len(exact)==4)
 check('Resume at completed padded endpoint yields no further batch',await collect(Harness(old,5,length=22,pad=True))==[])
 beyond=None
 try:await collect(Harness(old,6,length=22,pad=True))
 except AssertionError as e:beyond=type(e).__name__
 check('Resume beyond artificial finite endpoint hits original assertion',beyond=='AssertionError')
 result={'scope':__doc__,'runtime_python':platform.python_version(),'checks_passed':len(checks),'checks':checks,'prefetch':{'first_returned':first_batch,'first_store_request':h.dl.data_store.requests[0],'resume_completed_step_one':next_batch},'history_change':{'old':a,'rewritten':b,'future_only':c},'finite':{'padded_batches':p,'no_pad_batches':dr,'beyond_endpoint_error':beyond},'substitutions':['identity async store','null CPU mesh context','all-local artificial device index map','host result adapter replacing JAX batchification','await adapter replacing watchdog'],'original_methods':{n.name:[n.lineno,n.end_lineno] for n in methods},'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [LP,SP]},'actual_Hero_next_tokens':None,'actual_checkpoint_restore':None,'actual_JAX_batchification':None,'actual_background_prefetch':None}
 (ROOT/'analysis/loader_resume_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('Original async loader controls:',len(checks),'passed')
if __name__=='__main__':asyncio.run(main())
