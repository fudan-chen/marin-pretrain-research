"""Original PRP bodies, dataset shuffle/slice classes and train/val helper on CPU.
Replaces AsyncDataset base and CPU mesh context; finite identity store, no real tokens.
"""
import ast,asyncio,contextlib,dataclasses,functools,hashlib,json,logging,pathlib,platform,typing
import jax,jax.numpy as jnp,jax.random as jrandom,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];PP=ROOT/'sources/prp_2026_10_07/_prp.py';DP=ROOT/'sources/deepening_2026_10_04/dataset_production.py';TP=ROOT/'sources/deepening_2026_10_04/datasets_production.py';MP=ROOT/'sources/live_2026_10_07/meta.json'
T=typing.TypeVar('T');U=typing.TypeVar('U');T_co=typing.TypeVar('T_co')
class Base(typing.Generic[T]):
 def slice_dataset(self,start_index=None,end_index=None):return ns['SlicedAsyncDataset'](self,start_index,end_index)
ns={'jax':jax,'jnp':jnp,'jrandom':jrandom,'np':np,'typing':typing,'local_cpu_mesh':contextlib.nullcontext,'logger':logging.getLogger(__name__),'dataclass':dataclasses.dataclass,'lru_cache':functools.lru_cache,'AsyncDataset':Base,'T':T,'U':U,'T_co':T_co}
def load(p,selector):
 nodes=[n for n in ast.parse(p.read_text()).body if selector(n)]
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),ns)
load(PP,lambda n:isinstance(n,(ast.ClassDef,ast.FunctionDef)) or isinstance(n,ast.Assign))
names={'SlicedAsyncDataset','PermutationDataset','_BlockShuffleState','_WindowLayout','BlockShufflingDataset','_key_on_local_cpu','_fold_in_on_local_cpu'}
load(DP,lambda n:getattr(n,'name',None) in names)
load(TP,lambda n:getattr(n,'name',None)=='_split_into_trainval_sets')
class Identity(Base[int]):
 def __init__(self,length):self.length=length
 def is_finite(self):return True
 async def async_len(self):return self.length
 def as_sync_dataset(self):return range(self.length)
 async def get_batch(self,indices):return list(indices)
 async def getitem_async(self,i):return i
 def shuffle(self,key,perm_type='feistel'):return ns['PermutationDataset'](self,key,perm_type=perm_type)
 def slice_dataset(self,start_index=None,end_index=None):return ns['SlicedAsyncDataset'](self,start_index,end_index)
async def all_ids(ds):return await ds.get_batch(list(range(await ds.async_len())))
async def main():
 checks=[]
 def check(label,x):assert x,label;checks.append(label)
 check('Genuine CPU runtime',jax.default_backend()=='cpu')
 grid=[]
 for kind in ['feistel','linear']:
  for length in range(1,66):
   p=ns['Permutation'].make(kind,length,jrandom.PRNGKey(7));a=np.asarray(p(np.arange(length)))
   grid.append({'kind':kind,'length':length,'bijective':sorted(a.tolist())==list(range(length))})
 check('Both original PRPs bijective on all 130 small domain controls',all(x['bijective'] for x in grid))
 Block=ns['BlockShufflingDataset'];b=Block(Identity(22),4,window_blocks=3,key=7);ids=await all_ids(b)
 check('Original block shuffle preserves artificial inventory',sorted(ids)==list(range(22)))
 check('Tiny final block stays at end',set(ids[-2:])=={20,21} and set(ids[:-2])==set(range(20)))
 b2=Block(Identity(22),4,window_blocks=3,key=7)
 check('Same snapshot and key reproduces exact mapping',ids==await all_ids(b2))
 check('Duplicate out of order retrieval follows mapping',await b.get_batch([21,0,21,4])==[ids[i] for i in [21,0,21,4]])
 changed=await all_ids(Block(Identity(23),4,window_blocks=3,key=7))
 check('Changed inventory length changes partial tail mapping',ids!=changed[:22])
 changed_window=await all_ids(Block(Identity(22),4,window_blocks=2,key=7))
 check('Window setting changes permutation under same key',ids!=changed_window)
 errs={}
 for i in [-1,22]:
  try:await b.getitem_async(i)
  except (ValueError,IndexError) as e:errs[str(i)]=type(e).__name__
 check('Block shuffle rejects negative and endpoint index',errs=={'-1':'ValueError','22':'IndexError'})
 split=ns['_split_into_trainval_sets'];train,val=split(Identity(22),4);tr=await all_ids(train);va=await all_ids(val)
 check('Same snapshot fixed-key split is disjoint and exhaustive',set(tr).isdisjoint(va) and sorted(tr+va)==list(range(22)))
 train2,val2=split(Identity(22),4)
 check('Independent constructions agree under same snapshot',tr==await all_ids(train2) and va==await all_ids(val2))
 nt,nv=split(Identity(23),4);ntr=await all_ids(nt);nva=await all_ids(nv)
 cross=sorted(set(tr)&set(nva))
 check('Changing snapshot can mix old train with new validation identities',bool(cross))
 pos_t,pos_v=split(Identity(22),4,shuffle=False)
 check('Positional split uses exact last four identities',await all_ids(pos_t)==list(range(18)) and await all_ids(pos_v)==[18,19,20,21])
 decl=json.loads(MP.read_text())['config']['data']['value']
 check('Latest declared recipe does not split validation from training inventory',decl['num_validation_sequences'] is None)
 out={'declared_num_validation_sequences':decl['num_validation_sequences'],'declared_shuffle':decl['shuffle'],'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'python':platform.python_version(),'jax':jax.__version__,'numpy':np.__version__,'backend':jax.default_backend()},'permutation_grid':grid,'block22':ids,'block23':changed,'window2':changed_window,'split22':{'train':tr,'validation':va},'split23':{'train':ntr,'validation':nva},'old_train_new_validation_overlap':cross,'invalid_indices':errs,'substitutions':['AsyncDataset base','null local_cpu_mesh','finite identity store'],'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [PP,DP,TP,MP]},'actual_Hero_inner_shuffle':None,'actual_Hero_split_leakage':None,'actual_token_store':None,'actual_GPU_TPU':None}
 (ROOT/'analysis/inner_shuffle_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Original inner shuffle CPU/source checks:',len(checks),'passed; cross-snapshot overlap:',cross)
if __name__=='__main__':asyncio.run(main())
