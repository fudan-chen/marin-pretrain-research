"""Original global QB histogram wrapper and centering on a real one-device CPU shard_map.
Synthetic margins, not original full margin construction/model or production estimator events.
Single-device psum/pmin/pmax executed: not multi-host collective validation.
"""
import ast,hashlib,json,pathlib,typing
import jax,jax.numpy as jnp,numpy as np
from jax import shard_map
from jax.sharding import Mesh,PartitionSpec as P,AxisType
R=pathlib.Path(__file__).resolve().parents[1];M=R/'sources/main_incident_2026_10_07/model.py'
names={'_seq_axis','_token_axes','_bincount_upper_quantile','_qb_beta_hist'}
ns=dict(jax=jax,jnp=jnp,shard_map=shard_map,P=P)
for node in ast.parse(M.read_text()).body:
 if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id in ['_BATCH_AXES','_SEQ_AXIS_NAME']:ns[node.target.id]=ast.literal_eval(node.value)
nodes=[n for n in ast.parse(M.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(M),'exec'),ns)
setter=next(n for n in ast.parse(M.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='apply_qb_betas')
center=ast.parse('def center(qb_betas):\n pass').body[0]
center.body=[n for n in setter.body if isinstance(n,ast.Assign)]+[ast.parse('return new_bias').body[0]]
exec(compile(ast.fix_missing_locations(ast.Module(body=[center],type_ignores=[])),str(M),'exec'),ns)
mesh=Mesh(np.array(jax.devices()).reshape(1,1,1),('replica_dcn','data','expert'),axis_types=(AxisType.Explicit,)*3)
cases={'ordinary':([[.1,.2,.3],[.4,.5,.6],[.7,.8,.9],[1.,1.1,1.2]],[1,1,1,1]),'empty_valid':([[0.,0.,0.]]*4,[0,0,0,0]),'all_equal':([[1.,1.,1.]]*4,[1,1,1,1]),'large_equal':([[3e38,3e38,3e38]]*4,[1,1,1,1]),'wide_finite':([[-3e38,0.,3e38]]*4,[1,1,1,1]),'invalid_poison':([[.1,.2,.3],[.4,.5,.6],[.7,.8,.9],[np.nan,np.inf,-np.inf]],[1,1,1,0]),'valid_nan':([[.1,.2,.3],[.4,.5,.6],[.7,.8,.9],[np.nan,.5,.6]],[1,1,1,1]),'valid_inf':([[.1,.2,.3],[.4,.5,.6],[.7,.8,.9],[np.inf,.5,.6]],[1,1,1,1])}
def clean(x):
 if isinstance(x,list):return [clean(a) for a in x]
 if isinstance(x,float) and not np.isfinite(x):return 'NaN' if np.isnan(x) else '+Inf' if x>0 else '-Inf'
 return x
rows={}
with jax.set_mesh(mesh):
 for name,(m,v) in cases.items():
  margins=jnp.array(m,jnp.float32);valid=jnp.array(v,bool)
  beta,lo,hi=ns['_qb_beta_hist'](margins,valid,mesh,num_experts_per_token=2,num_experts=3,n_bins=10000)
  jax.block_until_ready(beta);bias=ns['center'](beta)
  rows[name]={'margins':clean(margins.tolist()),'valid':valid.tolist(),'beta':clean(beta.tolist()),'lo':clean(float(lo)),'hi':clean(float(hi)),'beta_finite':bool(jnp.all(jnp.isfinite(beta))),'bias':clean(bias.tolist()),'bias_finite':bool(jnp.all(jnp.isfinite(bias)))}
checks=[]
def check(n,v):assert v,n;checks.append({'name':n,'passed':True})
check('Original wrapper uses real CPU single-device shard map and collectives',jax.default_backend()=='cpu' and len(jax.devices())==1)
check('Ordinary histogram finite',rows['ordinary']['beta_finite'] and rows['ordinary']['bias_finite'])
check('Empty valid returns zeros and zero range',rows['empty_valid']['beta']==[0.,0.,0.] and rows['empty_valid']['lo']==rows['empty_valid']['hi']==0.)
check('All-equal ordinary range remains finite',rows['all_equal']['beta_finite'] and rows['all_equal']['bias_finite'])
check('Invalid poison omitted from valid range and yields finite beta',rows['invalid_poison']['beta_finite'] and rows['invalid_poison']['bias_finite'] and rows['invalid_poison']['lo']!= 'NaN')
check('Valid NaN is not sanitized by histogram',not rows['valid_nan']['beta_finite'])
check('Valid infinity is not sanitized by histogram',not rows['valid_inf']['beta_finite'])
check('Finite extreme equal margins yield finite beta but nonfinite centered bias',rows['large_equal']['beta_finite'] and not rows['large_equal']['bias_finite'])
check('Finite wide margin subtraction exposes overflow boundary',not rows['wide_finite']['beta_finite'])
o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'jaxlib':__import__('jaxlib').__version__,'numpy':np.__version__},'cases':rows,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [M,pathlib.Path(__file__)]},'actual_Hero_extreme_margin_event':None,'actual_multi_device_collectives':None,'actual_original_margin_construction':None,'actual_GPU_execution':None,'actual_estimator_causal_training_effect':None}
(R/'analysis/qb_hist_real_cpu.json').write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');print('Original histogram CPU checks',len(checks));print(json.dumps(rows,indent=2,allow_nan=False))
