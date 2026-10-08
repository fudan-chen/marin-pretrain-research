"""Original attention-derived validity + causal target mask + projected margin/histogram.
Artificial answer-only weights, margins/router inputs; no actual Hero answer-only data or full loss.
Four virtual CPU devices one host. AttentionMask class sentinel for None/dense branches only.
"""
import ast,hashlib,json,pathlib,types
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,PartitionSpec as P,AxisType,NamedSharding
R=pathlib.Path(__file__).resolve().parents[1];H=R/'scripts/probe_qb_margin_path_cpu.py';ctx={'__file__':str(H),'__name__':'valid_helpers'};src=H.read_text();exec(compile(src[:src.index('assert len(jax.devices())')],str(H),'exec'),ctx)
# Original projected function already extracted; load fixture projection wrapper without running prior cases.
proj=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='projection');exec(compile(ast.Module(body=[proj],type_ignores=[]),str(H),'exec'),ctx);projection=ctx['projection'];ns=ctx['ns']
A=R/'sources/boundaries_2026_10_05/attention_core.py';E=R/'sources/boundaries_2026_10_05/examples.py';T=R/'sources/main_incident_2026_10_07/train.py';M=ctx['M']
def extract(p,name,container=None):
 body=ast.parse(p.read_text()).body
 if container:body=next(n for n in body if isinstance(n,ast.ClassDef) and n.name==container).body
 node=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==name);node.decorator_list=[]
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])),str(p),'exec'),ns)
ns['AttentionMask']=type('AttentionMaskSentinel',(),{})
extract(A,'token_validity_from_attention_mask');extract(E,'causal_loss_mask','GrugLmExample')
assert len(jax.devices())==4
mesh=Mesh(np.array(jax.devices()).reshape(1,4,1),('replica_dcn','data','expert'),axis_types=(AxisType.Explicit,)*3);axes=('replica_dcn','data','expert')
weights={'dense':jnp.tile(ns['causal_loss_mask'](4),(2,1)).astype(jnp.float32),'answer_only':jnp.tile(ns['causal_loss_mask'](4,prompt_length=3),(2,1)).astype(jnp.float32),'zero_targets':jnp.zeros((2,4),jnp.float32)}
valid_fn=ns['token_validity_from_attention_mask'];allvalid=valid_fn(None,batch_size=2,sequence_length=4)
bool_mask=jnp.tile(jnp.tril(jnp.ones((4,4),bool)),(2,1,1));bool_mask=bool_mask.at[0,0,:].set(False);paddingvalid=valid_fn(bool_mask,batch_size=2,sequence_length=4)
additive=jnp.where(bool_mask,0.,-jnp.inf);additivevalid=valid_fn(additive,batch_size=2,sequence_length=4)
base=np.tile(np.array([[.4,.6,.2]],np.float32),(8,1));outlier=base.copy();outlier[0,1]=1e6
configs=[('dense',base,allvalid,weights['dense']),('answer_only',base,allvalid,weights['answer_only']),('zero_targets',base,allvalid,weights['zero_targets']),('zero_weight_prompt_outlier',outlier,allvalid,weights['answer_only']),('same_outlier_padding_excluded',outlier,paddingvalid,weights['answer_only']),('additive_same_empty_query',outlier,additivevalid,weights['answer_only'])]
rows={}
for name,matrix,valid,w in configs:
 logits,alpha,margin,ids,combine=projection(matrix)
 with jax.set_mesh(mesh):
  mar=jax.device_put(np.asarray(margin),NamedSharding(mesh,P(axes,None)));vv=jax.device_put(np.asarray(valid).reshape(-1),NamedSharding(mesh,P(axes)))
  beta,lo,hi=ns['_qb_beta_hist'](mar,vv,mesh,num_experts_per_token=2,num_experts=3,n_bins=10000);jax.block_until_ready(beta)
 rows[name]={'loss_weight':w.tolist(),'positive_targets':int(jnp.count_nonzero(w)),'target_mass':float(jnp.sum(w)),'router_valid':valid.tolist(),'router_valid_count':int(jnp.sum(valid)),'zero_weight_but_router_valid':int(jnp.sum((w==0)&valid)),'beta':beta.tolist(),'bias':ns['center'](beta).tolist(),'lo':float(lo),'hi':float(hi),'margin_first_row':margin[0].tolist(),'all_finite_beta':bool(jnp.all(jnp.isfinite(beta)))}
checks=[]
def check(n,v):assert v,n;checks.append({'name':n,'passed':True})
check('Original masks distinguish dense six answer two zero objective targets',rows['dense']['positive_targets']==6 and rows['answer_only']['positive_targets']==2 and rows['zero_targets']['positive_targets']==0)
check('Objective weighting alone leaves eight router-valid tokens and identical beta',all(rows[n]['router_valid_count']==8 and rows[n]['beta']==rows['dense']['beta'] for n in ['dense','answer_only','zero_targets']))
check('Six zero-target prompt or final positions remain router-valid in answer control',rows['answer_only']['zero_weight_but_router_valid']==6)
check('Outlier position has no direct target weight but participates in QB',rows['zero_weight_prompt_outlier']['loss_weight'][0][0]==0 and rows['zero_weight_prompt_outlier']['router_valid'][0][0] and rows['zero_weight_prompt_outlier']['beta']!=rows['answer_only']['beta'])
check('Excluding same position as actual padding removes outlier threshold effect',rows['same_outlier_padding_excluded']['router_valid_count']==7 and np.allclose(rows['same_outlier_padding_excluded']['beta'],rows['answer_only']['beta'],rtol=0,atol=1e-7))
check('Boolean no-key query and additive mask differ in original validity contract',rows['same_outlier_padding_excluded']['router_valid'][0][0]==False and rows['additive_same_empty_query']['router_valid'][0][0] and rows['additive_same_empty_query']['beta']==rows['zero_weight_prompt_outlier']['beta'])
check('Zero objective mass does not imply zero histogram target population',rows['zero_targets']['target_mass']==0 and rows['zero_targets']['router_valid_count']==8 and rows['zero_targets']['beta']!=[0.,0.,0.])
check('All declared synthetic masks and histogram outputs remain finite beta',all(r['all_finite_beta'] for r in rows.values()))
check('Frozen actual loss entry passes loss_weight and attention mask separately','batch.loss_weight' in T.read_text() and 'mask=batch.attn_mask' in T.read_text() and 'token_validity_from_attention_mask(mask' in M.read_text())
files=[pathlib.Path(__file__),H,R/'scripts/probe_qb_hist_real_cpu.py',A,E,T,M]
o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'jaxlib':__import__('jaxlib').__version__,'numpy':np.__version__},'device_count':4,'cases':rows,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'substitutions':['Artificial one-hot activations/router matrices and answer-only causal weights, not Hero data','AttentionMask sentinel only for None/boolean/additive branches; no structured-mask class execution','Original router fragment uses identity reshard/token spec; histogram actual four virtual CPU collectives','No complete loss, gradients, attention forward, capacity, expert dispatch or training update'],'actual_Hero_answer_only_training':None,'actual_Hero_router_valid_target_ratio':None,'actual_prompt_gradient_effect':None,'actual_GPU_execution':None,'actual_training_quality_effect':None,'actual_additive_mask_production_incident':None}
(R/'analysis/qb_target_validity_cpu.json').write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');print('Original target/router validity controls',len(checks));print(json.dumps(rows,indent=2))
