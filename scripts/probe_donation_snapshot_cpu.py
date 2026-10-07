"""Original checkpoint staging helper with real single-CPU JAX donation and delayed consumption."""
import ast,asyncio,functools,hashlib,json,pathlib,threading,gc
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import SingleDeviceSharding
R=pathlib.Path(__file__).resolve().parents[1]
P=R/'sources/donation_snapshot_2026_10_07/lib/levanter/src/levanter/tensorstore_serialization.py'
tree=ast.parse(P.read_text());names={'_transfer_shard_to_pageable_host','_slice_shard_on_device'}
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'_HOST_MEMORY_KIND','_GPU_PLATFORM','_PAGEABLE_HOST_MEMORY_KIND'}}
ns={'asyncio':asyncio,'jax':jax,'np':np,'SingleDeviceSharding':SingleDeviceSharding,**constants};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(P),'exec'),ns)
helper=ns['_transfer_shard_to_pageable_host'];checks=[];records=[]
def check(name,value):
 assert value,name
 checks.append({'name':name,'passed':True})
@functools.partial(jax.jit,donate_argnums=(0,))
def update(x):return x+jnp.float32(100)
retained_input=jnp.arange(8,dtype=jnp.float32);retained_view=np.asarray(retained_input)
retained_snapshot=asyncio.run(helper(retained_input.addressable_shards[0]));retained_result=update(retained_input);retained_result.block_until_ready()
retained_control={'input_deleted':retained_input.is_deleted(),'view_first':float(retained_view[0]),'snapshot_first':float(retained_snapshot[0]),'updated_first':float(np.asarray(retained_result)[0])}
check('retained external CPU host view prevents donation in this runtime',not retained_control['input_deleted'] and retained_control['view_first']==retained_control['snapshot_first']==0 and retained_control['updated_first']==100)
del retained_view,retained_input,retained_result,retained_snapshot;gc.collect()
for size in [8,4096]:
 x=jnp.arange(size,dtype=jnp.float32);expected=np.arange(size,dtype=np.float32);host=np.asarray(x)
 snapshot=asyncio.run(helper(x.addressable_shards[0]));independent=not np.shares_memory(host,snapshot);check('detached private host snapshot '+str(size),independent and np.array_equal(snapshot,expected))
 del host;gc.collect()
 started=threading.Event();release=threading.Event();output={}
 def writer():
  started.set()
  if not release.wait(5):output['error']='release timeout';return
  output['saved']=snapshot.copy()
 thread=threading.Thread(target=writer);thread.start();assert started.wait(5)
 try:
  y=update(x);y.block_until_ready();deleted=x.is_deleted();check('original JAX input deleted after actual donation '+str(size),deleted)
  z=update(y);z.block_until_ready();check('two actual donated updates produce expected new state '+str(size),np.array_equal(np.asarray(z),expected+200))
 finally:release.set();thread.join(5)
 check('delayed consumer still receives pre-donation snapshot '+str(size),not thread.is_alive() and 'error' not in output and np.array_equal(output['saved'],expected))
 records.append({'size':size,'input_deleted':deleted,'saved_first':float(output['saved'][0]),'new_first':float(np.asarray(z)[0]),'independent_host_memory':independent})
# Existing pinned-host branch: original helper with a minimal shard adapter around a real CPU-host array.
class MemoryKind:
 memory_kind=constants['_HOST_MEMORY_KIND']
class HostOperand:
 sharding=MemoryKind()
 def __init__(self,array):self.array=array
 def __array__(self,dtype=None,copy=None):return np.array(self.array,dtype=dtype,copy=True if copy is None else copy)
class Shard:
 def __init__(self,data):self.data=data
original_host=np.arange(8,dtype=np.float32);snap=asyncio.run(helper(Shard(HostOperand(original_host))));original_host[:]=-999
check('host-memory branch ownership survives source mutation with explicit adapter',np.array_equal(snap,np.arange(8,dtype=np.float32)) and not np.shares_memory(snap,original_host))
paths=[P,R/'sources/donation_snapshot_2026_10_07/lib/levanter/tests/test_tensorstore_serialization.py',R/'sources/donation_snapshot_2026_10_07/lib/levanter/src/levanter/checkpoint.py',R/'sources/main_incident_2026_10_07/train.py',pathlib.Path(__file__)]
data={'checks_passed':len(checks),'checks':checks,'cases':records,'retained_host_view_control':retained_control,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'jax_version':jax.__version__,'jaxlib_version':__import__('jaxlib').__version__,'numpy_version':np.__version__,'devices':[str(x) for x in jax.devices()],'boundary':'Original helper AST and constants; actual CPU JAX arrays, jitted donate_argnums updates, event-gated delayed NumPy copy consumer. No full Levanter import, TensorStore IO, GPU DMA, pinned allocator, multirank or actual Hero TrainState. Host-memory branch uses explicit minimal __array__ adapter.','actual_GPU_donation':None,'actual_Hero_checkpoint_integrity':None,'actual_training_resume':None}
(R/'analysis/donation_snapshot_cpu.json').write_text(json.dumps(data,indent=2)+'\n');print('Donation snapshot controls:',len(checks),'passed')
