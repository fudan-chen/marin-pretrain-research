"""Original checkpoint metadata/discovery/control helpers, local fixtures and fake async I/O.
No JAX, TensorStore, cloud access, actual arrays, or historical checkpoint restoration.
"""
import ast,contextlib,datetime,json,logging,os,pathlib,tempfile,types,urllib.parse,uuid,dataclasses,hashlib
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/checkpoint_commit_2026_10_05';checks=[];functions={};cases=[]
logger=logging.getLogger('checkpoint-probe');logger.addHandler(logging.NullHandler());logger.propagate=False

def check(name,condition):
 assert condition,name
 checks.append(name)
def extract(p,names,ns):
 tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert set(n.name for n in nodes)==set(names)
 mod=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]));exec(compile(mod,str(p),'exec'),ns)
 functions[p.name]=[{'name':n.name,'lines':[n.lineno,n.end_lineno]} for n in nodes]
class LocalFS:
 def makedirs(self,p,exist_ok=False):pathlib.Path(p).mkdir(parents=True,exist_ok=exist_ok)
 def exists(self,p):return pathlib.Path(p).exists()
 def open(self,p,mode='r'):return open(p,mode)
 def ls(self,p,detail=True):
  if not pathlib.Path(p).exists():raise FileNotFoundError(p)
  return [{'name':str(x),'type':'directory' if x.is_dir() else 'file'} for x in pathlib.Path(p).iterdir()]
 def mv(self,src,dst,recursive=True):os.replace(src,dst)
 def rm(self,p):pathlib.Path(p).unlink()
local=LocalFS();objects={};process=[0];moves=[]
class StoragePath:
 def __init__(self,p):self.path=str(p)
 def __str__(self):return self.path
 @property
 def is_remote(self):return self.path.startswith('s3://')
 def write_text(self,s):
  if self.is_remote:objects[self.path]=s
  else:pathlib.Path(self.path).write_text(s)
class CopyDeleteDenied:
 def __init__(self):self.objects={};self.calls=[]
 def mv(self,src,dst,recursive=True):
  self.calls.append('copy_then_delete');self.objects[dst]=self.objects[src];raise PermissionError('synthetic protected-prefix deletion denied')
 def rm(self,p):self.calls.append('cleanup_delete');raise PermissionError('synthetic protected-prefix deletion denied')
ns={'contextlib':contextlib,'uuid':uuid,'retry_with_backoff':lambda op,**kw:op(),'is_transient_s3_error':lambda e:False,'ExponentialBackoff':lambda **k:k,'factory':types.SimpleNamespace(url_to_fs=lambda p:(local,p))}
extract(D/'atomic.py',['unique_temp_path','_mv_with_retry','atomic_rename'],ns)
atomic=ns['atomic_rename'];bad=CopyDeleteDenied()
try:
 with atomic('s3://protected/metadata.json',filesystem=bad) as temp:bad.objects[temp]='payload'
except PermissionError:pass
else:raise AssertionError('Expected denied deletion')
check('copy/delete failure can leave destination visible',bad.objects.get('s3://protected/metadata.json')=='payload' and len(bad.objects)==2)
check('failed cleanup does not swallow original error',bad.calls==['copy_then_delete','cleanup_delete'])
cases.append({'case':'artificial_object_store_copy_delete_denied','destination_visible':True,'temporary_object_remains':True,'caller_success':False})
ns.update(fsspec=types.SimpleNamespace(get_fs_token_paths=lambda p:(local,None,[str(p)])),os=os,datetime=datetime,json=json,logger=logger,StoragePath=StoragePath,prefix_join=lambda root,name:root.rstrip('/')+'/'+name,jax=types.SimpleNamespace(process_index=lambda:process[0]),atomic_rename=atomic,urllib=urllib,_get_fs_and_plain_path=lambda p,fs=None:(fs or local,str(p)))
@dataclasses.dataclass
class Candidate:
 path:str
 step:int
 timestamp:datetime.datetime
 metadata:dict
ns['CheckpointCandidate']=Candidate
names=['_save_metadata','_load_metadata','_checkpoint_candidate_sort_key','_is_path_under_any','_discover_checkpoint_paths_single','_discover_checkpoint_candidates_single','discover_checkpoint_candidates','discover_latest_checkpoint','latest_checkpoint_path','save_checkpoint']
extract(D/'checkpoint.py',names,ns)
# Async serialization is deliberately substituted by a callback capture.
pending=[];events=[]
def serialize(path,tree,manager,**kw):
 pathlib.Path(path,'manifest.json').write_text('{}');pending.append(kw);events.append('staged');return 0
ns.update(CheckpointDebugConfig=lambda:types.SimpleNamespace(enabled=False,flush_logs=False),equinox=types.SimpleNamespace(filter=lambda tree,predicate:tree),is_jax_array_like=lambda v:False,tree_serialize_leaves_tensorstore=serialize,flush_debug_output=lambda *a:None)
with tempfile.TemporaryDirectory(prefix='marin-checkpoint-contract-') as tmp:
 root=pathlib.Path(tmp);p=root/'step-10'
 result=ns['save_checkpoint']({'fake_numeric_leaf':1},10,str(p),manager=object(),commit_callback=lambda:events.append('retention_callback'))
 check('async helper returns path before completion metadata',result==str(p) and (p/'manifest.json').exists() and not (p/'metadata.json').exists())
 check('manifest-only candidate is excluded from discovery',ns['discover_checkpoint_candidates'](tmp)==[])
 pending.pop()['commit_callback']()
 check('metadata precedes user retention callback',events==['staged','retention_callback'] and json.loads((p/'metadata.json').read_text())['step']==10)
 check('metadata-only fixture is discoverable without arrays',ns['latest_checkpoint_path'](tmp)==str(p) and not (p/'manifest.ocdbt').exists())
 cases.append({'case':'fake_async_save','path_returned_before_metadata':True,'manifest_only_discovered':False,'metadata_only_discovered':True,'real_arrays_written':False})
 # Metadata write failure must stop the caller's retention callback.
 q=root/'step-30';retained=[]
 ns['save_checkpoint']({},30,str(q),manager=object(),commit_callback=lambda:retained.append('called'))
 original_metadata=ns['_save_metadata']
 def denied_marker(*args,**kwargs):raise PermissionError('synthetic marker publication denied')
 ns['_save_metadata']=denied_marker
 try:pending.pop()['commit_callback']()
 except PermissionError:pass
 else:raise AssertionError('Expected metadata denial')
 finally:ns['_save_metadata']=original_metadata
 check('failed completion-marker publication prevents retention callback',retained==[] and not (q/'metadata.json').exists() and (p/'metadata.json').exists())
 # A later user callback failure does not retract a successfully published marker.
 q=root/'step-31'
 def bad_retention():raise RuntimeError('synthetic post-metadata callback failure')
 ns['save_checkpoint']({},31,str(q),manager=object(),commit_callback=bad_retention)
 try:pending.pop()['commit_callback']()
 except RuntimeError:pass
 else:raise AssertionError('Expected user callback failure')
 check('post-marker callback error leaves discoverable metadata',(q/'metadata.json').exists())
 cases.append({'case':'fake_commit_callback_failures','metadata_failure_blocks_retention':True,'post_metadata_callback_failure_keeps_marker':True,'real_arrays_written':False})
 # Remove only this fixture marker so the numeric discovery tests retain their intended set.
 (q/'metadata.json').unlink()
 # Readable metadata selects by numeric step, then timestamp, not lexicographic directory.
 for name,step,time in [('arbitrary-old-name',9,'2026-10-05T23:00:00'),('step-2',20,'2026-10-05T00:00:00'),('step-100',None,None)]:
  q=root/name;q.mkdir()
  if step is not None:(q/'metadata.json').write_text(json.dumps({'step':step,'timestamp':time}))
 (root/'malformed').mkdir();(root/'malformed/metadata.json').write_text('{')
 check('numeric metadata step wins over path and newer timestamp',ns['latest_checkpoint_path'](tmp)==str(root/'step-2'))
 check('max_step excludes newer numeric step',ns['latest_checkpoint_path'](tmp,max_step=10)==str(p))
 check('exclude prefix does not accidentally exclude sibling',ns['_is_path_under_any']('/root/a2',['/root/a']) is False and ns['_is_path_under_any']('/root/a/child',['/root/a']) is True)
 # Failure before callback leaves an existing completion marker in a reused path.
 def fail(*args,**kwargs):raise RuntimeError('synthetic precommit failure')
 ns['tree_serialize_leaves_tensorstore']=fail
 try:ns['save_checkpoint']({'fake_numeric_leaf':2},10,str(p),manager=object())
 except RuntimeError:pass
 else:raise AssertionError('Expected precommit failure')
 check('reused path does not retract pre-existing metadata on failure',(p/'metadata.json').exists() and str(p) in [c.path for c in ns['discover_checkpoint_candidates'](tmp)])
 cases.append({'case':'reuse_path_precommit_failure','old_metadata_remains_discoverable':True,'actual_array_corruption':None})
 # Local metadata uses original atomic helper; real fixture file readback.
 ns['_save_metadata'](str(p),11,False,{'owner':'fixture'})
 check('local metadata roundtrip retains reserved and extra fields',json.loads((p/'metadata.json').read_text())['owner']=='fixture' and not list(p.glob('metadata.json.tmp.*')))
ns['_save_metadata']('s3://protected/step-12',12,False,{'owner':'fixture'})
check('remote metadata bypasses copy/delete rename',json.loads(objects['s3://protected/step-12/metadata.json'])['step']==12)
process[0]=1;before=dict(objects);ns['_save_metadata']('s3://protected/step-13',13,False)
check('nonzero process does not publish metadata',objects==before)
process[0]=0
try:ns['save_checkpoint']({},1,'unused',metadata={'step':2})
except ValueError:pass
else:raise AssertionError('Reserved metadata must fail')
check('reserved metadata rejected before I/O',True)
# Static phase ordering, not a running TensorStore or distributed manager.
ser=ast.parse((D/'tensorstore_serialization.py').read_text());fn=next(n for n in ser.body if isinstance(n,ast.FunctionDef) and n.name=='tree_serialize_leaves_tensorstore')
call_lines={}
for n in ast.walk(fn):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('write_manifest','_serialize_arrays'):call_lines[n.func.id]=n.lineno
check('layout manifest is published before array serializer call',call_lines['write_manifest']<call_lines['_serialize_arrays'])
x={'scope':'original metadata/discovery/save-wrapper/atomic helpers with local fixtures and fake async serializer; TensorStore ordering static only','checks':checks,'checks_passed':len(checks),'functions':functions,'static_call_lines':call_lines,'cases':cases,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*.py'))},'substitutes':['local filesystem adapter','minimal StoragePath with in-memory s3 paths','jax process index stub','equinox filter identity','fake async serializer/callback capture','single-call retry substitute, no retry timing tested'],'actual_tensorstore_write':None,'actual_cloud_permissions':None,'actual_checkpoint_restore':None,'actual_distributed_commit':None,'historical_execution_sha':None}
(R/'analysis/checkpoint_commit_probe.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'checks_passed':len(checks),'cases':cases},indent=2))
