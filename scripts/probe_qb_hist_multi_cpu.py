"""Original histogram wrapper over four real virtual CPU devices, one host.
Synthetic margins/query; no model margin construction, GPU, network or task loss.
"""
import ast,hashlib,json,pathlib,types
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,PartitionSpec as P,AxisType,NamedSharding
R=pathlib.Path(__file__).resolve().parents[1];H=R/'scripts/probe_qb_hist_real_cpu.py'
ctx={'__file__':str(H),'__name__':'hist_helpers'};src=H.read_text();exec(compile(src[:src.index('mesh=Mesh(')],str(H),'exec'),ctx);ns=ctx['ns'];M=ctx['M']
assert len(jax.devices())==4,(jax.devices())
def mesh(devs):return Mesh(np.array(devs).reshape(1,len(devs),1),('replica_dcn','data','expert'),axis_types=(AxisType.Explicit,)*3)
meshes={1:mesh(jax.devices()[:1]),4:mesh(jax.devices())}
def run(margins,valid,bins=10000,devices=4):
 mm=meshes[devices];axes=('replica_dcn','data','expert')
 with jax.set_mesh(mm):
  margins=jax.device_put(np.array(margins,np.float32),NamedSharding(mm,P(axes,None)))
  valid=jax.device_put(np.array(valid,bool),NamedSharding(mm,P(axes)))
  beta,lo,hi=ns['_qb_beta_hist'](margins,valid,mm,num_experts_per_token=2,num_experts=3,n_bins=bins);jax.block_until_ready(beta)
  bias=ns['center'](beta)
  shards=[{'device_id':sh.device.id,'rows':int(sh.data.shape[0]),'valid_count':int(np.asarray(valid.addressable_shards[i].data).sum())} for i,sh in enumerate(margins.addressable_shards)]
  return {'devices':devices,'bins':bins,'beta':beta.tolist(),'bias':bias.tolist(),'lo':float(lo),'hi':float(hi),'bin_width':(float(hi)-float(lo))/bins,'shards':shards,'beta_finite':bool(jnp.all(jnp.isfinite(beta))),'bias_finite':bool(jnp.all(jnp.isfinite(bias)))}
m=np.stack([np.arange(12)/10,np.arange(12)/10+.2,np.arange(12)/10-.2],axis=1).astype(np.float32);valid=np.ones(12,bool)
wide=m.copy();wide[-1,1]=1e6
perm=np.array([11,0,7,1,9,3,8,4,10,2,6,5])
rows={'base_one':run(m,valid,devices=1),'base_four':run(m,valid),'permuted_four':run(m[perm],valid[perm]),'wide_one':run(wide,valid,devices=1),'wide_four':run(wide,valid),'wide_more_bins':run(wide,valid,bins=100000)}
padded=np.zeros((12,3),np.float32);padded[:3]=m[:3];padded[3:]=[np.nan,np.inf,-np.inf];vp=np.array([1]*3+[0]*9,bool)
rows['three_valid_one']=run(padded,vp,devices=1);rows['three_valid_four']=run(padded,vp)
rows['all_invalid_four']=run(np.zeros((12,3)),np.zeros(12,bool))
from probe_router_weight_path_cpu import route,recipe,P as ROUTER
query_logits=jnp.array([[.1,.6,0.]],jnp.float32)
for name,row in rows.items():
 _,ids,w,_=route(recipe,jnp.ones((1,1),jnp.float32),query_logits,jnp.array(row['bias']))
 row['query_expert_ids']=ids.tolist();row['query_weights']=w.tolist()
# A transparent exact sorted order statistic; separate reference convention from histogram interpolation.
rank=int(np.ceil(valid.sum()*2/3));exact=np.sort(m,axis=0)[::-1][rank-1]
wide_exact=np.sort(wide,axis=0)[::-1][rank-1]
checks=[]
def check(n,v):assert v,n;checks.append({'name':n,'passed':True})
def same(a,b):return all(a[k]==b[k] for k in ['beta','bias','lo','hi','query_expert_ids','query_weights'])
check('Four real local CPU devices execute original histogram shard_map',jax.default_backend()=='cpu' and len(rows['base_four']['shards'])==4 and [s['rows'] for s in rows['base_four']['shards']]==[3]*4)
check('Same ordinary data gives equal one/four-device beta bias and query route',same(rows['base_one'],rows['base_four']))
check('Ordinary permutation across shards preserves histogram and route',same(rows['permuted_four'],rows['base_four']))
check('Outlier case gives equal one/four-device pooled estimate',same(rows['wide_one'],rows['wide_four']))
check('Three valid rows on one shard aggregate without padded poison',same(rows['three_valid_one'],rows['three_valid_four']) and [s['valid_count'] for s in rows['three_valid_four']['shards']]==[3,0,0,0] and rows['three_valid_four']['beta_finite'])
check('All four shards empty valid gives zero thresholds',rows['all_invalid_four']['beta']==[0.,0.,0.] and rows['all_invalid_four']['lo']==rows['all_invalid_four']['hi']==0.)
check('Other-expert outlier changes unchanged expert-zero estimate',np.array_equal(m[:,0],wide[:,0]) and rows['base_four']['beta'][0]!=rows['wide_four']['beta'][0])
check('Range expansion coarsens shared bins and more bins changes unchanged expert estimate',rows['wide_four']['bin_width']>rows['base_four']['bin_width']*10000 and rows['wide_more_bins']['beta'][0]!=rows['wide_four']['beta'][0])
check('Exact order statistic remains unchanged after replacing already maximal other-expert margin',np.array_equal(exact,wide_exact))
check('More bins reduces expert-zero reference error here but does not recover exact threshold',abs(rows['wide_more_bins']['beta'][0]-float(exact[0]))<abs(rows['wide_four']['beta'][0]-float(exact[0])) and abs(rows['wide_more_bins']['beta'][0]-float(exact[0]))>1)
check('Finite pooled estimates need not reproduce base next query route',all(r['beta_finite'] and r['bias_finite'] for r in rows.values()) and rows['base_four']['query_expert_ids']!=rows['wide_four']['query_expert_ids'])
files=[pathlib.Path(__file__),H,M,R/'scripts/probe_router_weight_path_cpu.py',ROUTER]
o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'jaxlib':__import__('jaxlib').__version__,'numpy':np.__version__},'device_inventory':[{'id':d.id,'platform':d.platform,'kind':d.device_kind} for d in jax.devices()],'launch_XLA_FLAGS':'--xla_force_host_platform_device_count=4','margin_input':m.tolist(),'outlier_input':wide.tolist(),'exact_reference':{'convention':'descending sorted value at ceil(N*K/E), separate from interpolation','rank':rank,'beta':exact.tolist(),'outlier_beta':wide_exact.tolist()},'cases':rows,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'substitutions':['Synthetic margins and fixed router query, no full model margin construction','Original route block uses identity reshard and partition-spec placeholder; no capacity/dispatch','Four virtual CPU devices on one host, no network/GPU validation'],'actual_Hero_outlier_event':None,'actual_multi_host_collectives':None,'actual_GPU_execution':None,'actual_full_model_margin_construction':None,'actual_domain_causal_effect':None,'actual_histogram_performance_cost':None}
(R/'analysis/qb_hist_multi_cpu.json').write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');print('Multi-device original histogram controls',len(checks));print(json.dumps(rows,indent=2))
