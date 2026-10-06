"""Real local original leaf restore, manifest schema, and checkpoint discovery.
Synthetic markers; local StoragePath adapter; not full production publisher or state restore.
"""
import ast,dataclasses,datetime,enum,hashlib,importlib.metadata,json,logging,math,pathlib,tempfile,time
from typing import Any,Mapping,Sequence
import fsspec,jax,numpy as np,tensorstore as ts
from jax.sharding import SingleDeviceSharding
from pydantic import BaseModel,ConfigDict
import probe_serialize_arrays_real_io as writer
R=writer.R;P=writer.P;M=R/'sources/checkpoint_commit_2026_10_05/checkpoint_manifest.py';C=R/'sources/checkpoint_commit_2026_10_05/checkpoint.py'
class LocalStoragePath:
 def __init__(self,p):self.p=pathlib.Path(p)
 def exists(self):return self.p.exists()
 def read_text(self):return self.p.read_text()
 def write_text(self,s):self.p.parent.mkdir(parents=True,exist_ok=True);self.p.write_text(s)
ns=dict(writer.ns,**globals(),StoragePath=LocalStoragePath,prefix_join=lambda a,b:str(pathlib.Path(a)/b),dataclass=dataclasses.dataclass,StrEnum=enum.StrEnum,MANIFEST_FILENAME='manifest.json',CHECKPOINT_FORMAT_VERSION=1,logger=logging.getLogger('restore_probe'),_CONCURRENCY_RESOURCES=('data_copy_concurrency','s3_request_concurrency','gcs_request_concurrency','http_request_concurrency','file_io_concurrency'))
def extract(p,names):
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name in names]
 assert len(nodes)==len(names),(p,len(nodes),len(names))
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),str(p),'exec'),ns)
extract(M,['CheckpointArray','CheckpointManifest','build_manifest','manifest_path','write_manifest','read_manifest'])
ns['CheckpointManifest'].model_rebuild(_types_namespace=ns)
extract(P,['_format_gib','ReplicaRestoreMode','TensorStoreReadConfig','_LeafReadPlan','_RestoreReadResult','_tensorstore_read_context','_hashable_index','_leaf_read_plan','_read_store_shard','_stage_leaf','_finish_leaf','_read_batches','_deserialize_leaves','_restore_ocdbt'])
extract(C,['CheckpointCandidate','_load_metadata','_get_fs_and_plain_path','_discover_checkpoint_paths_single','_discover_checkpoint_candidates_single','_checkpoint_candidate_sort_key','_is_path_under_any','discover_checkpoint_candidates','discover_latest_checkpoint'])
def main():
 checks=[];rows={};expected=np.arange(1,25,dtype=np.float32).reshape(6,4);device=SingleDeviceSharding(jax.devices('cpu')[0]);cfg=ns['TensorStoreReadConfig'](max_in_flight_bytes=96,replica_mode=ns['ReplicaRestoreMode'].EVERY_REPLICA)
 def check(n,c):assert c,n;checks.append(n)
 def spec(root,key):return ns['_create_ocdbt_spec'](str(root),key,entry=ns['CheckpointArray'](path=key,shape=(6,4),dtype='float32',chunk_shape=(2,4)))
 def manifest(root):ns['write_manifest'](str(root),ns['build_manifest']([ns['CheckpointArray'](path=k,shape=(6,4),dtype='float32',chunk_shape=(2,4)) for k in ['a','b']],array_driver='zarr3',kvstore_driver='ocdbt'))
 def marker(root,step):(root/'metadata.json').write_text(json.dumps({'step':step,'timestamp':'2026-10-07T00:00:00+00:00','is_temporary':False}))
 def restore(root):
  leaves,indices=ns['_restore_ocdbt'](str(root),['a','b'],[0,1],[device,device],None,cfg,False);return {'values':[np.asarray(x).tolist() for x in leaves],'indices':indices,'jax_arrays':all(isinstance(x,jax.Array) for x in leaves)}
 with tempfile.TemporaryDirectory(prefix='marin-restore-candidate-') as tmp:
  root=pathlib.Path(tmp);good=root/'step10';good.mkdir();manifest(good);events=[];manager=writer.array_ser.GlobalAsyncCheckpointManager()
  ns['_serialize_arrays']([expected,expected+100],[spec(good,'a'),spec(good,'b')],[None,None],manager,ns['TensorStoreWriteConfig'](max_staged_host_bytes=96),lambda:events.append('committed'),None,None);manager.wait_until_finished();marker(good,10)
  rows['complete']=restore(good);check('Original real reader restores both original writer arrays',rows['complete']['values']==[expected.tolist(),(expected+100).tolist()] and rows['complete']['jax_arrays'] and rows['complete']['indices']==[0,1])
  partial=root/'step20';partial.mkdir();manifest(partial);ts.open(spec(partial,'a'),create=True).result().write(expected).commit.result();b=ts.open(spec(partial,'b'),create=True).result();b[:2,:].write((expected+100)[:2]).commit.result();marker(partial,20)
  rows['partial']=restore(partial);check('Strict leaf presence still permits legal missing-chunk fill',rows['partial']['values'][1]==(expected+100)[:2].tolist()+[[0.0]*4]*4)
  check('Filled original reader output is finite but differs from intended values',np.isfinite(rows['partial']['values']).all() and rows['partial']['values']!=rows['complete']['values'])
  absent=root/'step30';absent.mkdir();manifest(absent);ts.open(spec(absent,'a'),create=True).result().write(expected).commit.result();marker(absent,30)
  orphan=root/'step40';orphan.mkdir();manifest(orphan);ts.open(spec(orphan,'a'),create=True).result().write(expected).commit.result()
  candidates=ns['discover_checkpoint_candidates'](str(root));rows['discovered_steps']=[x.step for x in candidates];check('Actual discovery ignores manifest without metadata marker',rows['discovered_steps']==[10,20,30])
  latest=ns['discover_latest_checkpoint'](str(root));check('Discovery selects highest marker step without array validation',latest==str(absent))
  try:restore(absent);error=None
  except Exception as e:error=type(e).__name__+': '+str(e)
  rows['missing_array_error']=error;check('Manifest-listed but absent metadata fails original real reader',error is not None and 'NOT_FOUND' in error)
  (absent/'metadata.json').unlink();next_latest=ns['discover_latest_checkpoint'](str(root));rows['selected_after_removing_step30_marker']=pathlib.Path(next_latest).name;check('Next discovery selects marker-valid partially written fixture',next_latest==str(partial))
  try:ns['_restore_ocdbt'](str(good),['a','b','required_missing'],[0,1,2],[device]*3,None,cfg,False);missing_leaf=None
  except FileNotFoundError as e:missing_leaf=str(e)
  check('Strict restore rejects leaf absent from actual manifest',missing_leaf is not None);rows['missing_manifest_leaf_error']=missing_leaf
  mismatch=root/'shape_probe';mismatch.mkdir();manifest(mismatch);ts.open(spec(mismatch,'a'),create=True).result().write(expected).commit.result()
  short_spec=ns['_create_ocdbt_spec'](str(mismatch),'b',entry=ns['CheckpointArray'](path='b',shape=(2,4),dtype='float32',chunk_shape=(2,4)))
  ts.open(short_spec,create=True).result().write((expected+100)[:2]).commit.result();rows['manifest_shape_mismatch']=restore(mismatch)
  check('Original leaf reader uses stored shape without matching manifest shape',len(rows['manifest_shape_mismatch']['values'][1])==2 and ns['read_manifest'](str(mismatch)).arrays[1].shape==(6,4))
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'tensorstore':importlib.metadata.version('tensorstore'),'pydantic':importlib.metadata.version('pydantic'),'fsspec':fsspec.__version__},'observations':rows,'explicit_substitutions':{'StoragePath':'local pathlib adapter; no remote semantics','metadata_markers':'manually injected controls, not original publication callback','heap_trim':'no-op inherited','write_plans':'None, host branch only'},'original_CheckpointArray_schema':True,'original_leaf_read_pipeline':True,'original_checkpoint_discovery':True,'restore_mode':'EVERY_REPLICA, one CPU device','actual_full_state_restore':None,'actual_production_stale_marker_event':None,'actual_multirank_commit':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,M,C,writer.B]}}
 (R/'analysis/restore_candidate_real_io.json').write_text(json.dumps(o,indent=2)+'\n');print('Original restore/discovery real IO checks:',len(checks))
if __name__=='__main__':main()
