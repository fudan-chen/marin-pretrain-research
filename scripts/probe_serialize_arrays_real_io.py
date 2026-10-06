"""Original host-array serialization with real TensorStore and single-process JAX manager.
Not full tree serialization, production metadata publication, or distributed execution.
"""
import ast,asyncio,contextlib,dataclasses,hashlib,importlib.metadata,inspect,json,logging,os,pathlib,sys,tempfile,threading,types,urllib.parse
from typing import Any,Sequence,Callable,Optional
import jax,numpy as np,tensorstore as ts
from jax.experimental.array_serialization import serialization as array_ser,tensorstore_impl as ts_impl
R=pathlib.Path(__file__).resolve().parents[1]
P=R/'sources/checkpoint_commit_2026_10_05/tensorstore_serialization.py'
B=R/'sources/checkpoint_memory_2026_10_05/byte_budget.py'
ns=dict(globals(),logger=logging.getLogger('serialize_probe'),dataclass=dataclasses.dataclass,ARRAY_DRIVER='zarr3',KVSTORE_DRIVER='ocdbt',_DEFAULT_STAGED_CHUNKS=32,_STAGED_BYTE_OVERHEAD=4)
exec(compile(B.read_text(),str(B),'exec'),ns)
names=['_serialize_arrays','_estimate_array_nbytes','_tensorstore_write_context','_create_ocdbt_spec','build_kvstore_spec','TensorStoreWriteConfig']
nodes=[n for n in ast.parse(P.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
exec(compile(ast.fix_missing_locations(module),str(P),'exec'),ns)
ns['_trim_host_memory_after_commits']=lambda futures:None # Explicitly omit process heap trim.
def run():
 checks=[];results={};config=ns['TensorStoreWriteConfig'](max_staged_host_bytes=96,cache_pool_bytes=0)
 def check(name,condition):assert condition,name;checks.append(name)
 def spec(root,key):return ns['_create_ocdbt_spec'](root,key,entry=types.SimpleNamespace(shape=(6,4),dtype='float32',chunk_shape=(2,4)))
 def invoke(arrays,specs,manager,events,done=None):
  def local(status):
   events.append(status)
   if done is not None:done.set()
  return ns['_serialize_arrays'](arrays,specs,[None]*len(arrays),manager,config,lambda:events.append('commit_callback'),local,lambda size:events.append({'staged':size}))
 with tempfile.TemporaryDirectory(prefix='marin-real-serialize-') as tmp:
  root=str(pathlib.Path(tmp)/'ok');a=np.arange(1,25,dtype=np.float32).reshape(6,4);expected=a.copy();b=a+100;events=[];m=array_ser.GlobalAsyncCheckpointManager()
  peak=invoke([a,b],[spec(root,'a'),spec(root,'b')],m,events);a[:]=-999;b[:]=-999;m.wait_until_finished()
  restored=[np.asarray(ts.open(ns['_create_ocdbt_spec'](root,key),open=True,read=True).result().read().result()) for key in ['a','b']]
  check('Original host snapshots preserve pre-mutation values',np.array_equal(restored[0],expected) and np.array_equal(restored[1],expected+100))
  check('Single-process actual manager commits before wait returns',events.count('commit_callback')==1 and events.count('local_completed')==1)
  check('Two host arrays respect one-array budget',peak==96)
  results['success']={'events':events,'peak_bytes':peak,'restored_values':[x.tolist() for x in restored]}
  # Existing float32 metadata conflicts with a valid int32 spec: open failure is asynchronous.
  badroot=str(pathlib.Path(tmp)/'fail');ts.open(spec(badroot,'a'),create=True).result();bad=spec(badroot,'a');bad['metadata']['data_type']='int32'
  bad_events=[];badm=array_ser.GlobalAsyncCheckpointManager();done=threading.Event();badpeak=invoke([expected,expected+100],[bad,spec(badroot,'b')],badm,bad_events,done)
  try:badm.wait_until_finished();error=None
  except Exception as e:error=type(e).__name__+': '+str(e)
  check('Actual manager wait propagates real TensorStore metadata conflict',error is not None and 'data_type' in error)
  check('Failed local commit never invokes success callback',done.wait(5) and bad_events.count('local_failed')==1 and 'commit_callback' not in bad_events)
  good_after_failure=np.asarray(ts.open(ns['_create_ocdbt_spec'](badroot,'b'),open=True,read=True).result().read().result())
  check('Failure releases budget allowing second real write to complete',badpeak==96 and np.array_equal(good_after_failure,expected+100))
  # Manager's consumed error behavior is not assumed to repeat; use a fresh failed manager.
  blocked_events=[];blockedm=array_ser.GlobalAsyncCheckpointManager();invoke([expected],[bad],blockedm,blocked_events)
  next_events=[]
  try:invoke([expected],[spec(str(pathlib.Path(tmp)/'next'),'a')],blockedm,next_events);next_error=None
  except Exception as e:next_error=type(e).__name__+': '+str(e)
  check('Previous pending failure blocks next original serialization before staging',next_error is not None and next_events==[] and not (pathlib.Path(tmp)/'next').exists())
  blockedm.wait_until_finished()
  check('Installed manager consumes reported exception once',blockedm._exception is None)
  results['failure']={'events':bad_events,'peak_bytes':badpeak,'wait_error':error,'second_array_restored_values':good_after_failure.tolist(),'next_save_error':next_error,'next_save_events':next_events}
 check('Extracted six original declarations',len(nodes)==6)
 manager_file=pathlib.Path(inspect.getfile(array_ser.GlobalAsyncCheckpointManager))
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'python':sys.version.split()[0],'jax':jax.__version__,'tensorstore':importlib.metadata.version('tensorstore'),'process_count':jax.process_count(),'manager_source_sha256':hashlib.sha256(manager_file.read_bytes()).hexdigest(),'manager_check_for_errors_source':inspect.getsource(array_ser.GlobalAsyncCheckpointManager.check_for_errors)},'observations':results,'explicit_substitutions':{'heap_trim':'no-op','write_plans':'None entries; host branch does not consume plans','entry':'SimpleNamespace'},'actual_GlobalAsyncCheckpointManager':True,'actual_HostByteBudget':True,'actual_full_tree_serializer':None,'actual_multirank_commit':None,'actual_production_metadata_publication':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,B]}}
 (R/'analysis/serialize_arrays_real_io.json').write_text(json.dumps(o,indent=2)+'\n');print('Original serializer real IO checks:',len(checks))
if __name__=='__main__':run()
