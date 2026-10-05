"""Original asyncio byte budget and NumPy-free shape planning using synthetic sharding.
No TensorStore, hardware sharding, or actual checkpoint memory measurement.
"""
import ast,asyncio,collections,contextlib,dataclasses,hashlib,json,math,pathlib,threading,types,zlib
R=pathlib.Path(__file__).resolve().parents[1];p=R/'sources/checkpoint_memory_2026_10_05/byte_budget.py';q=R/'sources/checkpoint_commit_2026_10_05/tensorstore_serialization.py';checks=[];cases=[]
def check(name,condition):
 assert condition,name
 checks.append(name)
ns={};exec(compile('from __future__ import annotations\n'+p.read_text(),str(p),'exec'),ns);Budget=ns['HostByteBudget']
async def budget_checks():
 gate=Budget(10);await gate.acquire(6);waiting=asyncio.create_task(gate.acquire(6));await asyncio.sleep(0)
 check('sum above target waits while another snapshot is active',not waiting.done())
 gate.release(6);await asyncio.wait_for(waiting,1)
 check('release admits waiting snapshot and peak tracks concurrent bytes',gate.peak_bytes==6)
 gate.release(6);await asyncio.sleep(0)
 oversized=Budget(10);await oversized.acquire(14);waiting=asyncio.create_task(oversized.acquire(1));await asyncio.sleep(0)
 check('oversized snapshot proceeds alone above target',oversized.peak_bytes==14 and not waiting.done())
 th=threading.Thread(target=lambda:oversized.release(14));th.start();th.join();await asyncio.wait_for(waiting,1)
 check('completion-thread release wakes event-loop waiter',oversized._in_flight==1)
 oversized.release(1);await asyncio.sleep(0)
 gate=Budget(10)
 try:
  async with gate.reserve(7):raise RuntimeError('synthetic staging error')
 except RuntimeError:pass
 await asyncio.sleep(0)
 check('reserve exception releases capacity',gate._in_flight==0)
 cases.append({'case':'original_async_budget','target_bytes':10,'ordinary_peak_bytes':6,'oversized_peak_bytes':14,'thread_release_verified':True,'actual_RSS':None})
asyncio.run(budget_checks())
try:Budget(0)
except ValueError:check('nonpositive budget rejected',True)
else:raise AssertionError('Expected invalid budget')
# Extract unmodified planning functions, dataclasses and validation class.
tree=ast.parse(q.read_text());names={'_estimate_array_nbytes','_hashable_index','_uniform_replica_count','_shard_shape','_is_safe_to_slice','_capped_chunk_shape','plan_array_write','_shard_write_region','_process_staged_bytes'};classes={'TensorStoreWriteConfig','_WritePlan','_ShardWrite'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.ClassDef) and n.name in classes]
class SyntheticArray:pass
class IndivisibleError(Exception):pass
process=[0]
pl={'dataclass':dataclasses.dataclass,'math':math,'collections':collections,'zlib':zlib,'jax':types.SimpleNamespace(Array=SyntheticArray,process_index=lambda:process[0]),'IndivisibleError':IndivisibleError,'_HOST_MEMORY_KIND':'pinned_host','_DEFAULT_STAGED_CHUNKS':32}
mod=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]));exec(compile(mod,str(q),'exec'),pl)
Config=pl['TensorStoreWriteConfig'];defaults=Config();check('declared staging target equals 16 GiB',defaults.max_staged_host_bytes==16*1024**3)
class Sharding:
 def __init__(self,shape,replicas,memory='device'):self.shape=shape;self.replicas=replicas;self.memory_kind=memory
 def shard_shape(self,shape):return self.shape
 def devices_indices_map(self,shape):return {i:tuple(slice(0,n) for n in self.shape) for i in range(self.replicas)}
class Array(SyntheticArray):
 def __init__(self,shape,replicas,memory='device'):
  self.shape=shape;self.dtype=types.SimpleNamespace(itemsize=4);self.sharding=Sharding(shape,replicas,memory)
  self.addressable_shards=[types.SimpleNamespace(replica_id=i,index=tuple(slice(0,n) for n in shape),data=types.SimpleNamespace(size=math.prod(shape),dtype=self.dtype)) for i in range(replicas)]
config=Config(min_replica_slice_bytes=1,max_chunk_bytes=1024,max_staged_host_bytes=10)
a=Array((8,4),4);plan=pl['plan_array_write']('params/matrix',a,config)
check('four replicas divide first divisible axis',plan.split_axis==0 and plan.write_replicas==4 and plan.block==2)
regions=[pl['_shard_write_region'](shard,plan) for shard in a.addressable_shards]
check('replica slices cover global axis exactly once',[(r.index[0].start,r.index[0].stop) for r in regions]==[(0,2),(2,4),(4,6),(6,8)])
check('per-process staged sum counts distinct writer slices',pl['_process_staged_bytes'](a,plan)==128)
cases.append({'case':'synthetic_replica_split','shape':[8,4],'replicas':4,'min_slice_bytes':1,'plan':dataclasses.asdict(plan),'written_axis_intervals':[[r.index[0].start,r.index[0].stop] for r in regions]})
small=pl['plan_array_write']('params/matrix',a,defaults)
check('default minimum slice rejects tiny array split',small.write_replicas==1 and small.split_axis is None)
check('CRC path selects one writer rather than all replicas',small.writer_replica==zlib.crc32(b'params/matrix')%4 and sum(pl['_shard_write_region'](s,small) is not None for s in a.addressable_shards)==1)
pinned=pl['plan_array_write']('bias',Array((8,),4,'pinned_host'),config)
check('unsafe pinned host vector avoids device slicing',pinned.write_replicas==1)
even=pl['_capped_chunk_shape']((64,64),4,1024);odd=pl['_capped_chunk_shape']((33,33),4,1024)
check('even axes can satisfy chunk target',math.prod(even)*4<=1024)
check('all-odd shape remains over chunk target',odd==(33,33) and math.prod(odd)*4==4356)
cases.append({'case':'original_chunk_shape','target_bytes':1024,'even_input_shape':[64,64],'even_chunk_shape':list(even),'even_chunk_bytes':math.prod(even)*4,'odd_input_shape':[33,33],'odd_chunk_shape':list(odd),'odd_chunk_bytes':math.prod(odd)*4,'actual_Hero_oversized_chunk':None})
host=types.SimpleNamespace(shape=(8,4),dtype=types.SimpleNamespace(itemsize=4),size=32);plan=pl['plan_array_write']('host',host,config)
check('unsharded host array counts only process zero',pl['_process_staged_bytes'](host,plan)==128)
process[0]=1;check('nonzero process omits unsharded host write bytes',pl['_process_staged_bytes'](host,plan)==0)
x={'scope':'original asyncio budget on CPU event loop; original shape/replica planning functions with synthetic jax.Array/sharding','checks':checks,'checks_passed':len(checks),'source_sha256':{str(path.relative_to(R)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (p,q)},'functions':{n.name:[n.lineno,n.end_lineno] for n in nodes},'declared_defaults':dataclasses.asdict(defaults),'cases':cases,'substitutes':['synthetic jax.Array and sharding index map','dtype-size records instead of device data','process index stub'],'actual_RSS':None,'actual_model_write_plan':None,'actual_GPU_slice':None,'actual_TensorStore_IO':None,'historical_execution_sha':None}
(R/'analysis/checkpoint_memory_probe.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'checks_passed':len(checks),'cases':cases},indent=2))
