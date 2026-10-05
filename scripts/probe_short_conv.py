"""Original ShortConv reference/API/halo body with NumPy and simulated collectives.
No Pallas, BF16, gradients, JAX sharding or real distributed execution.
"""
import ast,functools,hashlib,json,pathlib,types,warnings
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/short_conv_2026_10_05';checks=[]
def extract(p,names,env):
 body=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
 for n in body:
  n.decorator_list=[]
  for f in ast.walk(n):
   if isinstance(f,ast.FunctionDef):
    f.returns=None
    for a in f.args.args+f.args.kwonlyargs:a.annotation=None
 exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(p),'exec'),env)
jnp=types.SimpleNamespace(**{k:getattr(np,k) for k in dir(np) if not k.startswith('_')});env={'jnp':jnp,'OOB_SEGMENT':-1,'functools':functools,'warnings':warnings,'DEFAULT_BATCH_AXES':('replica_dcn','data','expert')}
extract(D/'reference.py',{'short_conv_reference'},env);ref=env['short_conv_reference']
state={}
def shard_map(fn,**kwargs):
 def wrapped(w,x,seg):
  n=kwargs['mesh'].shape['context'];chunks=np.split(x,n,axis=1);segs=np.split(seg,n,axis=1) if seg is not None else [None]*n;state.update(chunks=chunks,segs=segs)
  out=[]
  for rank in range(n):state['rank']=rank;out.append(fn(w,chunks[rank],segs[rank]))
  return np.concatenate(out,axis=1)
 return wrapped
def ppermute(value,axis,permutation):
 rank=state['rank'];source=next((s for s,d in permutation if d==rank),None)
 if source is None:return np.zeros_like(value)
 a=state['chunks'][source] if value.ndim==3 else state['segs'][source]
 return a[:,-value.shape[1]:].copy()
env.update(jax=types.SimpleNamespace(lax=types.SimpleNamespace(axis_size=lambda a:len(state['chunks']),axis_index=lambda a:state['rank'],ppermute=ppermute)),shard_map=shard_map,P=lambda *a:a,reshard=lambda a,s:a,_partitioned_dims=lambda a,m:None,_assert_local_axes=lambda *a,**k:None)
extract(D/'api.py',{'_active_batch_axes','_short_conv_sharded','_round_up','_as_sequence','short_conv'},env)
def check(n,c):assert c,n;checks.append(n)
x=np.array([1,10,100,1000,10000,100000],np.float32).reshape(1,6,1);w=np.ones((3,1),np.float32);unique=np.array([[0,0,1,1,2,2]],np.int32);cases=[]
for name,seg in [('unpacked',None),('packed',unique),('reused',np.array([[0,1,0,2,2,2]],np.int32))]:cases.append({'name':name,'segment_ids':None if seg is None else seg.tolist(),'output':ref(w,x,seg).reshape(-1).tolist()})
check('Packed boundaries suppress preceding document taps',cases[1]['output']==[1,11,100,1100,10000,110000])
check('No segment metadata mixes preceding causal positions',cases[0]['output']==[1,11,111,1110,11100,111000])
check('Non-contiguous reused IDs reconnect an older matching segment',cases[2]['output'][2]==101)
identity=np.array([[1],[0],[0]],np.float32)
check('Identity initialization hides cross-document tap effects',np.array_equal(ref(identity,x,None),ref(identity,x,unique)))
check('Lag-zero tap stays active for padding-like negative IDs',ref(w,np.array([[[7.]]],np.float32),np.array([[-1]],np.int32)).item()==7)
mesh=types.SimpleNamespace(shape={'context':2});halo_cases=[]
for name,seg in [('unpacked',None),('cross_shard_same_document',np.zeros((1,6),np.int32)),('boundary_inside_shard',unique),('boundary_on_shard',np.array([[0,0,0,1,1,1]],np.int32))]:
 got=env['_short_conv_sharded'](w,x,seg,local_call=ref,mesh=mesh,batch_axes=('context',),seq_axis='context',padded_local_seq=8);want=ref(w,x,seg);halo_cases.append({'name':name,'got':got.reshape(-1).tolist(),'reference':want.reshape(-1).tolist()});check('Original halo body matches pooled reference: '+name,np.array_equal(got,want))
# Missing halo is a deliberately broken comparator, not an original backend.
no_halo=np.concatenate([ref(w,c,None) for c in np.split(x,2,axis=1)],axis=1)
check('Missing halo damages a continuous document at shard boundary',not np.array_equal(no_halo,ref(w,x,None)) and no_halo[0,3,0]==1000)
class Blocks:
 s_block_size=4
 @staticmethod
 def get_default():return Blocks()
env.update(ShortConvBlockSizes=Blocks,get_abstract_mesh=lambda:None,pallas_short_conv_available=lambda:False,_default_implementations=lambda:('reference',))
api=env['short_conv'];check('Default CPU substitute selects original reference',np.array_equal(api(w,x,unique),ref(w,x,unique)))
with warnings.catch_warnings(record=True) as ws:
 got=api(w,x,unique,implementation=('pallas_gpu','reference'));check('Ordered backend list warns then falls back',len(ws)==1 and np.array_equal(got,ref(w,x,unique)))
try:api(w,x,unique,implementation='pallas_gpu');bad=False
except RuntimeError:bad=True
check('Explicit unavailable backend fails rather than falling back',bad)
try:api(w.astype(np.float64),x,unique);bad=False
except ValueError:bad=True
check('Mixed weight and activation dtype is rejected at API',bad)
check('Round-up supports tile padding without changing retained prefix',env['_round_up'](5,4)==8)
config=[]
for p in sorted((R/'sources/scale_2026_10_05').glob('config_*.json')):
 c=json.loads(json.loads(p.read_text())['data']['project']['run']['config'])['model']['value'];config.append({'run':p.stem[7:],'sconv_settings':{k:v for k,v in c.items() if 'sconv' in k or 'conv' in k}})
check('Six ladder configs enable all three SConv sites at width four',len(config)==6 and all(c['sconv_settings']=={'sconv':True,'sconv_kernel':4,'sconv_sites':['k','attn','mlp']} for c in config))
paths=list(D.glob('*.py'))+[R/'sources/contracts_2026_10_05/model.py'];out={'checks_passed':len(checks),'checks':checks,'cases':cases,'halo_cases':halo_cases,'configurations':config,'scope':'original FP32 reference, halo body and dispatch API with NumPy and simulated shard_map/ppermute; sharding guards replaced by no-ops','source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'actual_pallas_result':None,'actual_bf16_result':None,'actual_gradient_result':None,'actual_distributed_result':None,'historical_execution_sha':None};(R/'analysis/short_conv_probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('ShortConv checks:',len(checks))
