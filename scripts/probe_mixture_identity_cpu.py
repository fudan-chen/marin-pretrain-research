"""Original MixtureDataset class and block permutation on real JAX CPU.
Replaced dependencies: AsyncDataset base, stop-strategy container, CPU mesh context,
future_from_value and finite identity child stores. No real token/shuffle store or Hero run.
"""
import ast,asyncio,collections,contextlib,functools,hashlib,json,logging,pathlib,platform,typing,warnings
import jax
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];P=ROOT/'sources/deepening_2026_10_04/mixture_production.py'
T=typing.TypeVar('T')
class Base(typing.Generic[T]):pass
class Stops:
 FIRST_STOP_STRATEGY='first_exhausted';ALL_STOP_STRATEGY='all_exhausted';RESTART_STRATEGY='restart'
 def __contains__(self,x):return x in [self.FIRST_STOP_STRATEGY,self.ALL_STOP_STRATEGY,self.RESTART_STRATEGY]
async def value_future(x):return x
tree=ast.parse(P.read_text());nodes=[x for x in tree.body if (isinstance(x,ast.ClassDef) and x.name=='MixtureDataset') or (isinstance(x,ast.FunctionDef) and x.name=='_compute_block_assignment')]
ns={'AsyncDataset':Base,'T':T,'StopStrategy':Stops(),'local_cpu_mesh':contextlib.nullcontext,'future_from_value':value_future,'jax':jax,'np':np,'asyncio':asyncio,'functools':functools,'warnings':warnings,'logger':logging.getLogger(__name__)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(P),'exec'),ns)
Mix=ns['MixtureDataset']
class Identity:
 def __init__(self,name,length=100,reverse=False):self.name=name;self.length=length;self.reverse=reverse;self.requests=[]
 def is_finite(self):return True
 async def async_len(self):return self.length
 async def get_batch(self,indices):
  self.requests.append(list(indices));return [f'{self.name}:{self.length-1-i if self.reverse else i}' for i in indices]
 async def getitem_async(self,i):return (await self.get_batch([i]))[0]
def build(order=('A','B'),weights=None,seed=7,length=100,reverse=False,stages=None):
 return Mix({n:Identity(n,length,reverse) for n in order},stages if stages else (weights or {n:1/len(order) for n in order}),8,key=seed)
def counts(m):return dict(zip(m.dataset_index,map(int,m._counts_per_block_per_stage[0])))
def bag(x):return dict(sorted(collections.Counter(x).items()))
async def main():
 checks=[]
 def check(n,v):assert v,n;checks.append(n)
 check('Real CPU permutation backend',jax.default_backend()=='cpu')
 m=build();whole=await m.get_batch(list(range(24)));split=[]
 for indices in [range(0,3),range(3,15),range(15,24)]:split+=await m.get_batch(list(indices))
 check('Request partition preserves identities',whole==split)
 repeated=await m.get_batch([11,0,11,7]);check('Unsorted duplicate requests preserve caller order',repeated==[whole[i] for i in [11,0,11,7]])
 other=await build(seed=8).get_batch(list(range(24)))
 check('Changed key changes sequence order',whole!=other)
 check('Complete blocks retain identity multiset across keys',bag(whole)==bag(other))
 partial=await m.get_batch([0,1,2]);partial_other=await build(seed=8).get_batch([0,1,2])
 check('Partial block identity multiset can differ across keys',bag(partial)!=bag(partial_other))
 reordered=build(order=('B','A'));ro=await reordered.get_batch(list(range(24)))
 check('Named equal quotas survive dictionary reorder',counts(m)==counts(reordered))
 check('Dictionary reorder changes sequence despite identical key and named quotas',ro!=whole and bag(ro)==bag(whole))
 tied=build(order=('A','B','C'));tied_rev=build(order=('C','B','A'))
 check('Integer remainder tie follows dataset insertion order',counts(tied)=={'A':4,'B':2,'C':2} and counts(tied_rev)=={'C':4,'B':2,'A':2})
 short=build(length=3);short_stream=await short.get_batch(list(range(24)))
 check('Finite restart wraps logical indices to child inventory',set(short_stream)=={f'{n}:{i}' for n in ['A','B'] for i in range(3)} and len(short_stream)==24)
 changed_length=await build(length=4).get_batch(list(range(24)))
 check('Inventory length changes content at same mix indices',short_stream!=changed_length)
 reversed_store=await build(reverse=True).get_batch(list(range(24)))
 check('Child mapping version changes identities at same index and key',whole!=reversed_store)
 staged=build(stages=[(0,{'A':.5,'B':.5}),(16,{'A':1})]);ss=await staged.get_batch(list(range(24)))
 check('Stage transition retains cumulative child offset',ss[:16]==whole[:16] and set(ss[16:])=={f'A:{i}' for i in range(8,16)})
 err=None
 try:await build(length=0).get_batch([0])
 except ValueError as e:err=str(e)
 check('Empty finite inventory is rejected',err is not None and 'empty finite dataset' in err)
 out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'python':platform.python_version(),'jax':jax.__version__,'numpy':np.__version__,'backend':jax.default_backend()},'whole_seed7':whole,'whole_seed8':other,'partial_seed7':partial,'partial_seed8':partial_other,'reordered_same_quota':ro,'tie_counts':{'ABC':counts(tied),'CBA':counts(tied_rev)},'finite_length3_stream':short_stream,'finite_length4_stream':changed_length,'staged_stream':ss,'empty_inventory_error':err,'source_sha256':{str(P.relative_to(ROOT)):hashlib.sha256(P.read_bytes()).hexdigest()},'actual_Hero_mapping':None,'actual_inner_shuffle':None,'actual_token_store':None,'actual_checkpoint_restore':None}
 (ROOT/'analysis/mixture_identity_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Mixture identity CPU/source controls:',len(checks),'passed')
if __name__=='__main__':asyncio.run(main())
