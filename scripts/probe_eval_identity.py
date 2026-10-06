"""Execute archived domain-index and loader planning bodies on synthetic finite
CPU datasets. No JAX permutation, background worker, model loss or sharded arrays.
"""
import ast,asyncio,collections,contextlib,dataclasses,hashlib,json,pathlib,types,warnings
import numpy as np
import audit_batch_clock as clock
R=clock.R;E=R/'sources/scale_2026_10_05/eval.py';L=R/'sources/eval_identity_2026_10_06/loader.py';env={'np':np,'jnp':np,'asyncio':asyncio,'warnings':warnings,'defaultdict':collections.defaultdict,'dataclasses':dataclasses,'logger':types.SimpleNamespace(debug=lambda *a:None),'local_cpu_mesh':contextlib.nullcontext,'blocking_wait':asyncio.run};checks=[]
def ck(n,v):assert v,n;checks.append(n)
def install(path,name,selected):
 t=ast.parse(path.read_text());cl=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name==name);body=[x for x in cl.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in selected];module=ast.Module(body=ast.parse('from __future__ import annotations').body+[ast.ClassDef(name=name,bases=[],keywords=[],body=body,decorator_list=[])],type_ignores=[]);exec(compile(ast.fix_missing_locations(module),str(path),'exec'),env);return env[name]
Domain=install(E,'DomainTaggedDataset',['__init__','_get_offsets','_compute_tag_arrays','async_len','getitem_async','get_batch','is_finite'])
class Dataset:
 def __init__(self,name,items):self.name=name;self.items=list(items)
 def is_finite(self):return True
 async def async_len(self):return len(self.items)
 async def getitem_async(self,i):return (self.name,self.items[i])
 async def get_batch(self,indices):return [(self.name,self.items[i]) for i in indices]
 def take(self,n):return Dataset(self.name,self.items[:n])
a=Dataset('A',range(3));b=Dataset('B',range(4));d=Domain([(a,['paloma/A']),(b,['paloma/B'])])
ck('Domain finite length equals concatenated child lengths',asyncio.run(d.async_len())==7)
ck('Domain offsets are stable cumulative boundaries',asyncio.run(d._get_offsets()).tolist()==[0,3,7])
ck('Boundary selects first item of next dataset',asyncio.run(d.getitem_async(3))[0]==('B',0))
indices=[6,0,3,3,2];actual=asyncio.run(d.get_batch(indices));expected=[asyncio.run(d.getitem_async(i)) for i in indices]
ck('Original batch mapping restores unsorted duplicate indices',all(x[0]==y[0] and np.array_equal(x[1],y[1]) for x,y in zip(actual,expected)))
ck('Tag vectors stay attached to the original item',actual[0][1].tolist()==[0,1] and actual[1][1].tolist()==[1,0])
capped=Domain([(a,['paloma/A']),(b,['paloma/B'])],2)
ck('Positive cap takes a prefix from each domain',asyncio.run(capped.async_len())==4 and [asyncio.run(capped.getitem_async(i))[0] for i in range(4)]==[('A',0),('A',1),('B',0),('B',1)])
Loader=install(L,'DataLoader',['__iter__','iter_from_step','__len__','has_len']);Iterator=install(L,'DataLoaderIterator',['_produce_batches','_dataset_get_available_batch_number'])
@dataclasses.dataclass
class Batch:index:int;global_data_offset:int;global_size:int;data_by_local_index:dict
env['_Batch']=Batch
# Python3.9 runtime subscripting in the source; generic typing only, not data transformation.
Batch.__class_getitem__=classmethod(lambda cls,item:cls)
def make_loader(n,batch=4,pad=True,fetch=3):
 dl=Loader();dl.scheduler=clock.schedule([{'start':0,'value':batch}]);dl.data_store=Dataset('D',range(n));dl.fetch_batch_size=fetch;dl._pad_final_batch=pad;return dl
async def plan(dl,start=None):
 it=Iterator();it.dl=dl;it._start_from_batch=start
 async def retrieve(specs):return specs
 it._do_retrieve_batch_of_batches=retrieve;it._batchify_local_data=lambda spec:list(range(spec.global_data_offset,spec.global_data_offset+spec.global_size))
 return [batch async for batch in it._produce_batches()]
first=asyncio.run(plan(make_loader(10)));second=asyncio.run(plan(make_loader(10)))
ck('Original planning restarts the default iterator from zero',first==second==[[0,1,2,3],[4,5,6,7],[8,9]])
ck('Final partial batch retains all logical items exactly once',sum(first,[])==list(range(10)))
ck('Exact-multiple stream yields two batches rather than len estimate',asyncio.run(plan(make_loader(8)))==[list(range(4)),list(range(4,8))])
ck('Original len overcounts exact-multiple progress estimate',len(make_loader(8))==3)
ck('Partial-length progress estimate matches three logical batches',len(make_loader(10))==3)
ck('Disabling partial padding drops the partial logical batch',asyncio.run(plan(make_loader(10,pad=False)))==[list(range(4)),list(range(4,8))])
ck('Changing fetch grouping does not change logical stream',sum(asyncio.run(plan(make_loader(10,fetch=1))),[])==sum(first,[]))
ck('Explicit start from batch one omits only the first logical batch',asyncio.run(plan(make_loader(10),1))==[[4,5,6,7],[8,9]])
# Execute __iter__/iter_from_step with constructor recorder instead of real workers.
env['DataLoaderIterator']=lambda dl,start_from_batch: {'start':start_from_batch}
ck('Original __iter__ creates a new default-start iterator',iter_result:=make_loader(10).__iter__()=={'start':None})
LOG=R/'sources/eval_identity_2026_10_06/logging.py'
Timing=install(LOG,'LoadingTimeTrackerIterator',['__init__'])
env['DataLoaderIterator']=lambda dl,start_from_batch: iter([0,1,2])
one=Timing(make_loader(3));next(one.items);two=Timing(make_loader(3))
ck('Original timing wrapper immediately creates a new loader iterator',list(two.items)==[0,1,2] and list(one.items)==[1,2])
# Config and call-site AST: no claim these were the historical executing source.
T=R/'sources/scale_2026_10_05/train_hero_ep.py';tt=ast.parse(T.read_text());builder=next(x for x in tt.body if isinstance(x,ast.FunctionDef) and x.name=='build_tagged_evaluator');call=next(x for x in ast.walk(builder) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='TaggedEvaluator');ctor=next(x for x in ast.parse(E.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='TaggedEvaluator');init=next(x for x in ctor.body if isinstance(x,ast.FunctionDef) and x.name=='__init__');defaults=dict(zip([x.arg for x in init.args.args][-len(init.args.defaults):],init.args.defaults))
ck('Hero builder omits shuffle and archived evaluator defaults false',not any(k.arg=='shuffle' for k in call.keywords) and isinstance(defaults['shuffle'],ast.Constant) and defaults['shuffle'].value is False)
META=R/'sources/live_2026_10_06/meta.json';cfg=json.loads(META.read_text())['config'];ec=cfg['eval']['value'];dc=cfg['data']['value']
ck('Latest declared eval has no cap and batch 704',ec['max_eval_batches'] is None and ec['eval_batch_size']==704)
ck('Latest eval uses explicit validation caches without train validation split',dc['num_validation_sequences'] is None and dc['components']['paloma/ptb-llama3']['split']=='validation')
ck('PTB and Twitter declarations keep fixed dated cache paths',all(dc['components']['paloma/'+n]['cache_dir'].endswith('/2026.06.28') for n in ['ptb-llama3','twitterAAE_HELM_fixed-llama3']))
padding=next(x for x in ast.parse(L.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='_make_padding_example');ck('Padding helper explicitly delegates to tree_zeros_like',any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='tree_zeros_like' for x in ast.walk(padding)))
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'synthetic_default_batches':first,'synthetic_exact_multiple':{'actual_batches':2,'len_estimate':3},'declared_eval':{'batch':704,'max_eval_batches':None,'num_validation_sequences':None,'builder_shuffle_argument':'omitted; archived default false','PTB_cache':dc['components']['paloma/ptb-llama3']['cache_dir'],'Twitter_cache':dc['components']['paloma/twitterAAE_HELM_fixed-llama3']['cache_dir']},'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [E,L,LOG,T,META]},'actual_historical_execution_SHA':None,'actual_cache_content_hash':None,'actual_sharded_loader':None,'actual_eval_metric_replay':None,'actual_GPU_behavior':None}
(R/'analysis/eval_identity_probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Evaluation identity CPU checks:',len(checks))
