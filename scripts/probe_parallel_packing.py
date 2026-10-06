"""Original packing methods with synthetic NumPy stores and flat-dictionary-only tree adapters.
No TensorStore I/O, JAX device arrays, model loss, or actual Hero caches.
"""
import ast,asyncio,builtins,contextlib,hashlib,json,pathlib,types
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/packing_2026_10_07/packing.py'
class Tree:
 @staticmethod
 def map(fn,*trees):
  if isinstance(trees[0],dict):return {k:Tree.map(fn,*[t[k] for t in trees]) for k in sorted(trees[0])}
  return fn(*trees)
 @staticmethod
 def leaves(t):return sum([Tree.leaves(t[k]) for k in sorted(t)],[]) if isinstance(t,dict) else [t]
 @staticmethod
 def flatten(t):
  keys=tuple(sorted(t));return [t[k] for k in keys],keys
 @staticmethod
 def unflatten(keys,values):return dict(zip(keys,values))
def broadcast(x,t):return Tree.map(lambda _:x,t)
def paths(t):return {k:k for k in sorted(t)}
def strict_zip(*xs,strict=False):
 if strict and len({len(x) for x in xs})!=1:raise ValueError('length mismatch')
 return builtins.zip(*xs)
ns={'np':np,'asyncio':asyncio,'jax':types.SimpleNamespace(tree=Tree),'ts':types.SimpleNamespace(Batch=contextlib.nullcontext),'tree_broadcast_to':broadcast,'leaf_key_paths':paths,'zip':strict_zip}
t=ast.parse(P.read_text());f=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='pack_documents');c=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='GreedyPrepackedDataset');c.bases=[]
mod=ast.Module(body=ast.parse('from __future__ import annotations').body+[f,c],type_ignores=[]);exec(compile(ast.fix_missing_locations(mod),str(P),'exec'),ns)
class Read:
 def __init__(self,value,synchronous):self.value=value;self.synchronous=synchronous
 def read(self):
  if self.synchronous:return types.SimpleNamespace(result=lambda:self.value.copy())
  async def done():return self.value.copy()
  return done()
class Array:
 def __init__(self,value,synchronous=False):self.value=value;self.synchronous=synchronous
 def __getitem__(self,index):return Read(self.value[index],self.synchronous)
class Store:
 def __init__(self,docs):
  self.num_rows=len(docs);lengths=[len(x) for x in docs];offsets=np.concatenate([[0],np.cumsum(lengths)]);offsets[0]=self.num_rows
  self.offsets=Array(offsets,True);self.data=Array(np.concatenate([np.asarray(x) for x in docs]))
def pack(docs,L=4,**kw):
 ds=ns['GreedyPrepackedDataset']({k:Store(v) for k,v in docs.items()},L,**kw);return ds,asyncio.run(ds.get_batch(list(range(asyncio.run(ds.async_len())))))
checks=[]
def ck(n,v):assert v,n;checks.append(n)
def serial(out):return [{'data':{k:v.tolist() for k,v in d.items()},'segments':{k:v.tolist() for k,v in s.items()}} for d,s in out]
match={'input_ids':[[10,11,12],[20]],'loss_weights':[[1.,.5,0.],[1.]]};ds,out=pack(match);ck('Matched token-aligned fields retain their document segments',out[0][1]['input_ids'].tolist()==out[0][1]['loss_weights'].tolist()==[0,0,0,1])
bad={'input_ids':[[10,11,12],[20]],'loss_weights':[[1.,.5],[1.,0.]]};bad_ds,bad_out=pack(bad);ck('Original generic packer accepts same row counts with different per-document lengths',len(bad_out)==1)
ck('Packed output shapes can match despite field-boundary mismatch',bad_out[0][0]['input_ids'].shape==bad_out[0][0]['loss_weights'].shape==(4,))
ck('Original per-field segment ids expose shifted document ownership',bad_out[0][1]['input_ids'].tolist()==[0,0,0,1] and bad_out[0][1]['loss_weights'].tolist()==[0,0,1,1])
# Dataset wrapper takes only input_ids segment ids with raw loss weights.
dp=R/'sources/deepening_2026_10_04/datasets_production.py';dt=ast.parse(dp.read_text());pc=next(x for x in dt.body if isinstance(x,ast.ClassDef) and x.name=='PackedTokenDataset');ck('Archived packed text wrapper uses input_ids segments for both fields',any(isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id=='seg_ids_raw' for y in x.targets) and ast.unparse(x.value)=="seg_ids['input_ids']" for x in ast.walk(pc)))
_,padded=pack({'input_ids':[[10,11]],'loss_weights':[[1.,0.]]});ck('Original padding uses zeros and negative segment ids',padded[0][0]['input_ids'].tolist()==[10,11,0,0] and padded[0][1]['input_ids'].tolist()==[0,0,-1,-1])
long={'input_ids':[[10,11,12,13,14]],'loss_weights':[[0.,0.,1.,1.,1.]]}
_,left=pack(long,slice_strategy='left');_,right=pack(long,slice_strategy='right');ck('Left keeps prefix while right keeps suffix in all aligned fields',left[0][0]['input_ids'].tolist()==[10,11,12,13] and right[0][0]['input_ids'].tolist()==[11,12,13,14] and right[0][0]['loss_weights'].tolist()==[0.,1.,1.,1.])
try:pack(long,slice_strategy='raise');reject=False
except ValueError:reject=True
ck('Raise rejects over-length document during pack planning',reject)
_,drop=pack({'input_ids':[[10,11,12,13,14],[20]],'loss_weights':[[1.,1.,1.,1.,1.],[1.]]},slice_strategy='drop');ck('Drop omits long document and retains original document identity',drop[0][0]['input_ids'].tolist()==[20,0,0,0] and drop[0][1]['input_ids'].tolist()==[1,-1,-1,-1])
_,cap=pack({'input_ids':[[10],[20],[30]],'loss_weights':[[1.],[1.],[1.]]},max_segments_per_example=2);ck('Segment cap produces two packs before token capacity is filled',len(cap)==2 and cap[0][1]['input_ids'].tolist()==[0,1,-1,-1])
try:pack({'input_ids':[[10],[20]],'loss_weights':[[1.]]});reject=False
except ValueError:reject=True
ck('Original packer rejects different numbers of documents',reject)
# Original default causal mask applied to synthetic weights; loss values are artificial.
ep=R/'sources/boundaries_2026_10_05/examples.py';et=ast.parse(ep.read_text());ec=next(x for x in et.body if isinstance(x,ast.ClassDef) and x.name=='GrugLmExample');mf=next(x for x in ec.body if isinstance(x,ast.FunctionDef) and x.name=='causal_loss_mask');mf.decorator_list=[];mn={'jnp':np};exec(compile(ast.fix_missing_locations(ast.Module(body=ast.parse('from __future__ import annotations').body+[mf],type_ignores=[])),str(ep),'exec'),mn)
mask=mn['causal_loss_mask'](4);good_w=out[0][0]['loss_weights']*mask;bad_w=bad_out[0][0]['loss_weights']*mask
ck('Original final-position mask does not repair field-boundary mismatch',good_w.tolist()==[1.,.5,0.,0.] and bad_w.tolist()==[1.,.5,1.,0.])
artificial_loss=np.array([1.,2.,9.,8.]);numerics={'matched_T':float(good_w.sum()),'mismatch_T':float(bad_w.sum()),'matched_N':float((good_w*artificial_loss).sum()),'mismatch_N':float((bad_w*artificial_loss).sum())}
ck('Synthetic identical token losses yield different weighted N T',numerics=={'matched_T':1.5,'mismatch_T':2.5,'matched_N':2.,'mismatch_N':11.})
mp=R/'sources/live_2026_10_06/meta.json';components=json.loads(mp.read_text())['config']['data']['value']['components'];ck('Latest archived 223 components declare text format and no explicit packing',len(components)==223 and all(x['pack'] is None and x['format']=={'text_key':'text'} for x in components.values()))
result={'checks_passed':len(checks),'checks':checks,'synthetic_weighted_numerics':numerics,'synthetic_matched':serial(out),'synthetic_equal_shape_mismatch':serial(bad_out),'synthetic_left':serial(left),'synthetic_right':serial(right),'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,dp,ep,mp]},'substitutions':['flat dictionary-only tree map/flatten/broadcast adapters','NumPy stores with synchronous offset reads and async data copies','null TensorStore Batch context','Python3.9 strict zip adapter'],'declared_components_checked':223,'actual_Hero_parallel_field_mismatch':None,'actual_TensorStore_reads':None,'actual_GPU_loss':None}
(R/'analysis/parallel_packing_probe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('Original synthetic packing checks:',len(checks))
