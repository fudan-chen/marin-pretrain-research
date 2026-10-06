"""CPU input envelope for the frozen PR9833 division statements.
Synthetic scalar expert y=3; not a ragged MoE, GPU, or training reproduction.
"""
import hashlib,json,pathlib
import jax,jax.numpy as jnp
from probe_routing_gradient_proposal_cpu import case,P,statements
R=pathlib.Path(__file__).resolve().parents[1]
def main():
 rows=[];checks=[]
 def check(label,condition):
  assert condition,label
  checks.append(label)
 for name,dtype in [('float16',jnp.float16),('bfloat16',jnp.bfloat16),('float32',jnp.float32)]:
  for we in [-4,-10,-30,-60]:
   for de in [-4,-20,-60,-90]:
    r=case(f'{name}_w2^{we}_d2^{de}',2.**we,2.**de,dtype)
    r.update(weight_exponent=we,cotangent_exponent=de,dtype_name=name)
    r['input_cast_zero']=r['w']==0 or r['dout']==0
    r['positive_represented_inputs_lost_gradient']=not r['input_cast_zero'] and r['rounded_weighted_cotangent']==0 and r['exact_reference_dS']>0
    rows.append(r)
 logits=jnp.asarray([-50.,0.],jnp.float32);s=jax.nn.sigmoid(logits);weights=s/jnp.sum(s)
 positive=case('positive_renormalized_sigmoid_bf16',float(weights[0]),2.**-70,jnp.bfloat16)
 positive['synthetic_logits']=[-50.,0.];positive['normalized_weights']=[float(x) for x in weights]
 def chain(logits,dout,dtype):
  x=jnp.asarray(logits,jnp.float32)
  def normalized(z):
   v=jax.nn.sigmoid(z);return v/jnp.sum(v)
  weights,pullback=jax.vjp(normalized,x)
  r=case('router_chain',float(weights[0]),dout,dtype)
  exact=pullback(jnp.asarray([r['exact_reference_dS'],0.],jnp.float32))[0]
  proposed=pullback(jnp.asarray([r['proposal_division_dS'],0.],jnp.float32))[0]
  return {'logits':logits,'scalar_expert':r,'float32_router_vjp_reference':[float(v) for v in exact],'float32_router_vjp_proposal':[float(v) for v in proposed]}
 chains=[chain([-50.,0.],2.**-70,jnp.bfloat16),chain([-7.,0.],2.**-20,jnp.float16)]
 check('Extreme bf16 local mismatch is zero after this float32 router VJP in both paths',all(v==0 for r in chains[:1] for k in ['float32_router_vjp_reference','float32_router_vjp_proposal'] for v in r[k]))
 check('Normal sigmoid fp16 product loss survives this float32 router VJP',any(v!=0 for v in chains[1]['float32_router_vjp_reference']) and all(v==0 for v in chains[1]['float32_router_vjp_proposal']))
 check('CPU backend is executed',jax.default_backend()=='cpu')
 check('Grid has 48 controls across three dtypes',len(rows)==48 and len({r['dtype_name'] for r in rows})==3)
 check('Every represented positive pair has nonzero float32 reference',all(r['exact_reference_dS']>0 for r in rows if not r['input_cast_zero']))
 for name in ['float16','bfloat16','float32']:
  check(name+' has at least one represented-positive product lost on this CPU',any(r['positive_represented_inputs_lost_gradient'] for r in rows if r['dtype_name']==name))
 check('Positive renormalized sigmoid control loses product despite nonzero bf16 operands',positive['w']>0 and positive['dout']>0 and positive['rounded_weighted_cotangent']==0 and positive['proposal_division_dS']==0 and positive['exact_reference_dS']>0)
 check('Ordinary bf16 grid control retains exact power-of-two reference',rows[16]['proposal_division_dS']==rows[16]['exact_reference_dS'])
 summary={name:{'controls':16,'input_cast_zero':sum(r['input_cast_zero'] for r in rows if r['dtype_name']==name),'represented_positive_product_loss':sum(r['positive_represented_inputs_lost_gradient'] for r in rows if r['dtype_name']==name)} for name in ['float16','bfloat16','float32']}
 out={'scope':__doc__,'runtime':{'jax':jax.__version__,'jaxlib':__import__('jaxlib').__version__,'backend':jax.default_backend(),'devices':[str(x) for x in jax.devices()]},'checks_passed':len(checks),'checks':checks,'exact_patch_statements':statements,'source_sha256':{str(P.relative_to(R)):hashlib.sha256(P.read_bytes()).hexdigest()},'grid':rows,'summary':summary,'positive_sigmoid_control':positive,'router_chain_controls':chains,'limitations':['Artificial scalar row-dot producer; only the division/mask patch statements execute unchanged.','Extreme inputs show existence, not prevalence. Hardware/backend subnormal treatment can differ.','Grid is deliberately chosen, not a training distribution; counts are not failure probabilities.','Float32 reference is not the production EXACT backend.'],'actual_GPU_reproduction':None,'actual_Hero_input_distribution':None,'actual_Hero_loss_effect':None}
 (R/'analysis/routing_gradient_envelope_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'checks_passed':len(checks),'summary':summary,'positive_sigmoid_control':positive},indent=2))
if __name__=='__main__':main()
