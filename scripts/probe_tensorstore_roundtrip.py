"""Real local TensorStore Zarr3/OCDBT IO using original spec helpers.
Small synthetic arrays; not the complete Marin serializer or production checkpoint.
"""
import ast,hashlib,importlib.metadata,json,os,pathlib,subprocess,sys,tempfile,types,urllib.parse
from typing import Any
import numpy as np,tensorstore as ts
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/checkpoint_commit_2026_10_05/tensorstore_serialization.py'
ns={'os':os,'urllib':urllib,'Any':Any,'ARRAY_DRIVER':'zarr3','KVSTORE_DRIVER':'ocdbt'};tree=ast.parse(P.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['build_kvstore_spec','_create_ocdbt_spec']];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(P),'exec'),ns);spec=ns['_create_ocdbt_spec']
def read(root):
 a=ts.open(spec(root,'params/token_embed'),open=True,read=True,recheck_cached=True).result();return {'values':np.asarray(a.read().result()).tolist(),'shape':list(a.shape),'dtype':str(a.dtype),'pid':os.getpid()}
def child(root):
 result=subprocess.run([sys.executable,__file__,'--read-root',root],capture_output=True,text=True,check=True,timeout=30)
 return json.loads(result.stdout)
def main():
 checks=[]
 def check(n,v):assert v,n;checks.append(n)
 expected=np.arange(1,25,dtype=np.float32).reshape(6,4);entry=types.SimpleNamespace(shape=(6,4),dtype='float32',chunk_shape=(2,4));rows={}
 with tempfile.TemporaryDirectory(prefix='marin-ts-io-') as tmp:
  root=str(pathlib.Path(tmp)/'complete');a=ts.open(spec(root,'params/token_embed',entry=entry),create=True,open=True).result();write=a.write(expected);write.copy.result();write.commit.result()
  rows['complete_child']=child(root);check('Fresh child process restores exact committed array',rows['complete_child']['pid']!=os.getpid() and np.array_equal(rows['complete_child']['values'],expected))
  check('Shape and dtype survive fresh process',rows['complete_child']['shape']==[6,4] and rows['complete_child']['dtype']=='dtype("float32")')
  try:ts.open(spec(root,'pending_qb_betas'),open=True,read=True).result();missing_error=None
  except ValueError as e:missing_error=str(e)
  check('Missing array metadata rejects open',missing_error is not None)
  try:ts.open(spec(root,'params/token_embed'),open=True,dtype=ts.int32).result();dtype_error=None
  except ValueError as e:dtype_error=str(e)
  check('Incompatible dtype constraint rejects existing array',dtype_error is not None)
  # A created array with unwritten chunks is legal Zarr data, and returns fill values.
  partial=str(pathlib.Path(tmp)/'partial');pa=ts.open(spec(partial,'params/token_embed',entry=entry),create=True).result();pa[:2,:].write(expected[:2,:]).commit.result();rows['partial_child']=child(partial)
  got=np.array(rows['partial_child']['values']);check('Unwritten chunks read as zero fill in this default Zarr3 control',np.array_equal(got[:2],expected[:2]) and np.array_equal(got[2:],np.zeros((4,4))))
  keys=a.kvstore.list().result();rows['complete_logical_keys']=[k.decode() for k in keys]
  chunk=next(k for k in keys if k.decode().startswith('c/'))
  a.kvstore.write(chunk,None).result();rows['deleted_chunk_key']=chunk.decode();rows['deleted_chunk_child']=child(root)
  restored=np.array(rows['deleted_chunk_child']['values']);deleted_row=int(chunk.decode().split('/')[1])*2;expected_deleted=expected.copy();expected_deleted[deleted_row:deleted_row+2]=0
  check('Deleted logical chunk restores fill rather than missing-array error',np.array_equal(restored,expected_deleted))
  check('Default Zarr fill result remains finite despite incorrect array',np.isfinite(restored).all())
  rows['missing_array_error']=missing_error;rows['dtype_constraint_error']=dtype_error
  rows['physical_files_after_deletion']=[str(p.relative_to(root)) for p in pathlib.Path(root).rglob('*') if p.is_file()]
 check('Original source spec helpers selected without replacement',len(nodes)==2)
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'python':sys.version.split()[0],'tensorstore':importlib.metadata.version('tensorstore'),'numpy':np.__version__},'fixture':{'expected_values':expected.tolist(),'shape':[6,4],'chunk_shape':[2,4],'driver':'zarr3','kvstore':'ocdbt','storage':'local file','original_spec_helpers_executed':True,'entry_metadata_object':'SimpleNamespace, not original CheckpointArray class'},'observations':rows,'actual_full_Marin_serializer':None,'actual_production_missing_chunk_event':None,'actual_GPU_checkpoint_restore':None,'source_sha256':{str(P.relative_to(R)):hashlib.sha256(P.read_bytes()).hexdigest()}}
 (R/'analysis/tensorstore_roundtrip.json').write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');print('Real TensorStore IO checks:',len(checks),'TensorStore',o['runtime']['tensorstore'])
if __name__=='__main__':
 if len(sys.argv)==3 and sys.argv[1]=='--read-root':print(json.dumps(read(sys.argv[2])))
 else:main()
