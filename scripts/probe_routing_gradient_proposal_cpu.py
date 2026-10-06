"""Execute two exact division/masking statements from PR9833's patch on CPU.
The row-dot producer is an artificial scalar linear expert, not the ragged MoE kernel.
"""
import ast,hashlib,json,pathlib,types
import jax,jax.numpy as jnp
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/engineering_current_2026_10_07';P=D/'pull_9833_files.json'
files=json.loads(P.read_text());patch=next(f['patch'] for f in files if f['filename'].endswith('ep_ragged_all_to_all.py'))
lines=[line[1:].strip() for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++')]
statements=[next(l for l in lines if l.startswith('divisible =')),next(l for l in lines if l.startswith('weights_cotangent ='))]
code=compile(ast.parse('\n'.join(statements)),str(P),'exec')
def case(name,w,dout,dtype):
 wf=jnp.asarray(w,jnp.float32);d=jnp.asarray(dout,dtype);weight=jnp.asarray(w,dtype);y=jnp.asarray(3.,dtype)
 dy=(weight*d).astype(dtype);dot=dy.astype(jnp.float32)*y.astype(jnp.float32)
 ns={'jnp':jnp,'routing':types.SimpleNamespace(accepted=jnp.asarray(True)),'weights_f32':wf,'assignment_output_dot':dot}
 exec(code,ns)
 exact=jax.grad(lambda z:z*jnp.asarray(3.,jnp.float32)*jnp.asarray(dout,jnp.float32))(wf)
 return {'name':name,'dtype':str(dtype),'w':float(weight),'dout':float(d),'rounded_weighted_cotangent':float(dy),'artificial_row_dot':float(dot),'exact_reference_dS':float(exact),'proposal_division_dS':float(ns['weights_cotangent'])}
def main():
 checks=[]
 def check(n,v):assert v,n;checks.append(n)
 a=case('ordinary',.25,2.,jnp.float32);z=case('accepted_zero_weight',0.,2.,jnp.float32);h=case('positive_fp16_underflow',2**-10,2**-20,jnp.float16);b=case('same_small_product_bf16',2**-10,2**-20,jnp.bfloat16)
 check('Genuine CPU backend',jax.default_backend()=='cpu')
 check('Ordinary scalar expert gradient agrees',a['proposal_division_dS']==a['exact_reference_dS']==6.)
 check('Accepted zero weight loses the otherwise nonzero weight gradient',z['proposal_division_dS']==0 and z['exact_reference_dS']==6.)
 check('Positive normal float16 weight can have rounded zero weighted cotangent',h['w']>0 and h['rounded_weighted_cotangent']==0 and h['proposal_division_dS']==0 and h['exact_reference_dS']>0)
 check('Same small product survives in this bfloat16 control',b['rounded_weighted_cotangent']>0 and b['proposal_division_dS']==b['exact_reference_dS'])
 prs={str(n):json.loads((D/f'pull_{n}.json').read_text()) for n in [9832,9833]}
 check('Snapshot PRs are unmerged proposals',all(not p['merged'] and p['state']=='open' for p in prs.values()))
 testpatch=next(f['patch'] for f in files if f['filename'].endswith('test_grugformer_moe.py'))
 check('Upstream test excludes expert-side edge assignments from interior tolerance','inside[edge_assignments] = False' in testpatch)
 out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'backend':jax.default_backend()},'exact_patch_statements':statements,'cases':[a,z,h,b],'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,D/'pull_9832.json',D/'pull_9833.json']},'actual_ragged_MoE_execution':None,'actual_GPU_reproduction':None,'actual_Hero_deployment':None,'actual_long_window_stability':None}
 (R/'analysis/routing_gradient_proposal_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('PR formula CPU/source controls:',len(checks),'passed')
if __name__=='__main__':main()
