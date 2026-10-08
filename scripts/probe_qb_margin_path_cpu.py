"""Original model router projection/top-k/margin statements -> real four-CPU histogram.
Synthetic one-hot activations/router matrices; not complete Transformer or production data.
Identity reshard only for router fragment, histogram uses actual sharding/collectives.
"""
import ast,hashlib,json,pathlib,types
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,PartitionSpec as P,AxisType,NamedSharding
R=pathlib.Path(__file__).resolve().parents[1];H=R/'scripts/probe_qb_hist_real_cpu.py';ctx={'__file__':str(H),'__name__':'margin_helpers'};src=H.read_text();exec(compile(src[:src.index('mesh=Mesh(')],str(H),'exec'),ctx);ns=ctx['ns'];M=ctx['M']
cls=next(n for n in ast.parse(M.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MoEMLP');call=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
start=next(i for i,n in enumerate(call.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='router_logits' for t in n.targets));end=next(i for i,n in enumerate(call.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mesh' for t in n.targets))
margin=next(n for n in call.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='s_minus_alpha' for t in n.targets))
fn=ast.parse('def projected(self,x_flat,x):\n pass').body[0];fn.body=call.body[start:end]+[margin,ast.parse('return router_logits,qb_alpha,s_minus_alpha,selected_experts,combine_weights').body[0]]
ns.update(reshard=lambda x,p:x,_token_spec=lambda:P(),_ROUTING_RENORM_SUM=2.5)
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(M),'exec'),ns)
assert len(jax.devices())==4
mesh=Mesh(np.array(jax.devices()).reshape(1,4,1),('replica_dcn','data','expert'),axis_types=(AxisType.Explicit,)*3);axes=('replica_dcn','data','expert')
def projection(matrix,bias=None):
 mat=jnp.array(matrix,jnp.float32);x=jnp.eye(len(matrix),dtype=jnp.float32);obj=types.SimpleNamespace(router=mat,router_bias=jnp.zeros(3,jnp.float32) if bias is None else jnp.array(bias,jnp.float32),cfg=types.SimpleNamespace(num_experts_per_token=2))
 return ns['projected'](obj,x,x)
def hist(m):
 with jax.set_mesh(mesh):
  m=jax.device_put(np.asarray(m),NamedSharding(mesh,P(axes,None)));v=jax.device_put(np.ones(m.shape[0],bool),NamedSharding(mesh,P(axes)))
  b,lo,hi=ns['_qb_beta_hist'](m,v,mesh,num_experts_per_token=2,num_experts=3,n_bins=10000);jax.block_until_ready(b)
 return b,float(lo),float(hi)
base=np.stack([np.arange(12)/10+.4,np.arange(12)/10+.6,np.arange(12)/10+.2],axis=1).astype(np.float32);rows={};query=[[.1,.6,0.]]
for name,value in [('base',None),('outlier_2',2.),('outlier_10',10.),('outlier_100',100.),('outlier_1e6',1e6)]:
 matrix=base.copy()
 if value is not None:matrix[-1,1]=value
 logits,alpha,margins,ids,weights=projection(matrix);beta,lo,hi=hist(margins);bias=ns['center'](beta)
 _,_,_,query_ids,query_weights=projection(query,bias)
 exact=np.sort(np.asarray(margins),axis=0)[::-1][7]
 rows[name]={'router_matrix':matrix.tolist(),'actual_projected_logits':logits.tolist(),'alpha':alpha.tolist(),'margins':margins.tolist(),'training_expert_ids':ids.tolist(),'beta':beta.tolist(),'centered_bias':bias.tolist(),'lo':lo,'hi':hi,'bin_width':(hi-lo)/10000,'exact_reference_beta':exact.tolist(),'expert0_reference_error':abs(float(beta[0])-float(exact[0])),'query_ids':query_ids.tolist(),'query_weights':query_weights.tolist()}
# Nonzero incoming bias changes alpha: margins need not be nonnegative or have minimum zero.
logits,alpha,margins,ids,w=projection(base,[-.15,-.15,.3]);beta,lo,hi=hist(margins)
biased={'logits':logits.tolist(),'alpha':alpha.tolist(),'margins':margins.tolist(),'beta':beta.tolist(),'margin_min':float(jnp.min(margins))}
checks=[]
def check(n,v):assert v,n;checks.append({'name':n,'passed':True})
check('Original model projection reproduces synthetic router matrix exactly',all(r['actual_projected_logits']==r['router_matrix'] for r in rows.values()))
check('Zero incoming bias K2/E3 makes alpha row minimum and margin row minimum zero',all(np.allclose(np.asarray(r['alpha']).ravel(),np.min(np.asarray(r['actual_projected_logits']),axis=1)) and np.array_equal(np.min(np.asarray(r['margins']),axis=1),np.zeros(12)) for r in rows.values()))
check('Changing already maximum expert1 logit keeps alpha expert0 and expert2 margins',all(r['alpha']==rows['base']['alpha'] and np.array_equal(np.asarray(r['margins'])[:,[0,2]],np.asarray(rows['base']['margins'])[:,[0,2]]) for r in rows.values()))
check('Exact rank8 per-expert reference unchanged under maximum-logit replacement',all(np.allclose(r['exact_reference_beta'],rows['base']['exact_reference_beta'],rtol=0,atol=1e-7) for r in rows.values()))
check('Original projected outlier path changes shared grid and other expert beta',rows['outlier_1e6']['beta'][0]!=rows['base']['beta'][0] and rows['outlier_1e6']['bin_width']>rows['base']['bin_width']*10000)
check('Same original projected query changes expert set after histogram pending view',sorted(rows['base']['query_ids'][0])!=sorted(rows['outlier_1e6']['query_ids'][0]))
check('All synthetic inputs and outputs finite in projected sweep',all(np.isfinite(np.array(r['actual_projected_logits'])).all() and np.isfinite(np.array(r['beta'])).all() and np.isfinite(np.array(r['centered_bias'])).all() for r in rows.values()))
check('Nonzero incoming bias permits negative margins',biased['margin_min']<0)
check('Projection source uses raw unbiased logits for margins not sigmoid probabilities','router_logits - qb_alpha' in ast.unparse(margin) and 'sigmoid' not in ast.unparse(margin))
files=[pathlib.Path(__file__),H,M]
o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'jaxlib':__import__('jaxlib').__version__,'numpy':np.__version__},'device_count':len(jax.devices()),'query_input_logits':query,'exact_reference_convention':'descending rank ceil(12*2/3)=8, distinct from histogram interpolation','cases':rows,'incoming_nonzero_bias_case':biased,'extracted_projected_source':ast.unparse(fn),'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'substitutions':['One-hot synthetic activations and router matrices; no Transformer embeddings/attention/norm or real batches','Identity reshard and token spec only in original router fragment; histogram executes real four virtual CPU collectives','No capacity/expert dispatch, optimization or task loss'],'actual_Hero_margin_distribution':None,'actual_real_data_domain_effect':None,'actual_GPU_execution':None,'actual_full_Transformer_execution':None,'actual_multi_host_collectives':None,'actual_training_quality_effect':None}
(R/'analysis/qb_margin_path_cpu.json').write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');print('Original projected margin controls',len(checks));print(json.dumps({n:{k:r[k] for k in ['beta','bin_width','expert0_reference_error','query_ids']} for n,r in rows.items()},indent=2))
