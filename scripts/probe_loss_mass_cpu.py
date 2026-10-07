"""Original causal mask, model next-token loss method and Grug reducer on CPU.
Synthetic domain-one-hot hidden states; original CE reference explicitly replaces xla_fast_bwd dispatcher.
No Transformer forward, Hero cache, distributed reduction or production gradient attribution.
"""
import ast,hashlib,json,pathlib,types
import jax,jax.numpy as jnp,numpy as np
import probe_mixture_identity_cpu as mix
R=mix.ROOT;D=R/'sources/contracts_2026_10_05';E=R/'sources/boundaries_2026_10_05/examples.py';G=R/'sources/scale_2026_10_05/grug_loss.py';M=D/'model.py';ns={'jax':jax,'jnp':jnp,'named_call':lambda f:f,'_current_mesh':lambda:None,'_CE_BLOCK_SIZES':None}
def extract(p,names,container=None,strip=False):
 body=ast.parse(p.read_text()).body
 if container:body=next(n for n in body if isinstance(n,ast.ClassDef) and n.name==container).body
 nodes=[n for n in body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
 if strip:
  for n in nodes:n.decorator_list=[]
 exec(compile(ast.fix_missing_locations(ast.Module(body=ast.parse('from __future__ import annotations').body+nodes,type_ignores=[])),str(p),'exec'),ns)
extract(E,{'causal_loss_mask'},'GrugLmExample',True);mask=ns['causal_loss_mask']
extract(D/'api.py',{'_apply_reduction'});extract(D/'reference.py',{'_default_logsumexp','_apply_logit_soft_cap','linear_softmax_cross_entropy_loss_reference'})
ns['_cross_entropy_logsumexp']=ns['_default_logsumexp'];ref=ns['linear_softmax_cross_entropy_loss_reference'];reduce=ns['_apply_reduction'];backend_calls=[]
def delegate(x,labels,w,**kw):
 backend_calls.append({'requested_implementation':kw['implementation'],'executed':'original CPU reference','z_loss':kw['logsumexp_weight']})
 losses,lse=ref(x,labels,w,dtype=kw['dtype']);z=kw['logsumexp_weight'];losses=losses if z is None else losses+z*lse**2
 return reduce(losses,kw['reduction'],kw['weight'])
ns['fused_cross_entropy_loss_and_logsumexp_penalty']=delegate;extract(G,{'_psum_over_axes','fused_linear_softmax_cross_entropy_loss'});extract(M,{'next_token_loss'},'Transformer')
class Model:
 next_token_loss=ns['next_token_loss']
 def __init__(self,head,domain_ids):self.output_proj=head;self.domain_ids=domain_ids
 def __call__(self,tokens,mask=None):return jnp.broadcast_to(jax.nn.one_hot(self.domain_ids,2)[:,None,:],tokens.shape+(2,)),{}
checks=[];cases={}
def ck(n,v):assert v,n;checks.append({'name':n,'passed':True})
def case(name,counts=(1,1),sparse_b=True,b_scale=1.,theta=(0.,0.),all_scale=1.):
 ids=jnp.array([0]*counts[0]+[1]*counts[1]);tokens=jnp.ones((sum(counts),4),dtype=jnp.int32)
 a=mask(4).astype(jnp.float32)*all_scale;b=mask(4,prompt_length=3 if sparse_b else None).astype(jnp.float32)*b_scale*all_scale
 weights=jnp.stack([a]*counts[0]+[b]*counts[1]);head=jnp.array([[theta[0],0.],[theta[1],0.]],dtype=jnp.float32)
 fn=lambda h:Model(h,ids).next_token_loss(tokens,weights,logsumexp_weight=None)
 loss,grad=jax.value_and_grad(fn)(head)
 mass=[float(jnp.sum(weights[ids==i])) for i in range(2)];positive=[int(jnp.sum(weights[ids==i]>0)) for i in range(2)]
 row={'sequence_counts':list(counts),'input_slots':sum(counts)*4,'nonzero_target_positions':positive,'weighted_target_mass':mass,'weighted_target_share':[m/sum(mass) for m in mass],'loss':float(loss),'first_logit_gradient':np.asarray(grad[:,0]).tolist(),'mask_a':np.asarray(a).tolist(),'mask_b':np.asarray(b).tolist(),'z_loss_weight':None}
 cases[name]=row;return row
uniform=case('uniform_half_weight',all_scale=.5)
both=case('dense_both',sparse_b=False);sparse=case('equal_sequences_sparse_b');scaled=case('equal_sequences_half_weight_b',b_scale=.5);different=case('different_domain_losses',theta=(0.,float(np.log(3))));block_equal=case('block8_equal_sequences',counts=(4,4));comp=case('block8_inverse_density',counts=(2,6))
ck('Original causal mask yields three dense targets and one answer target',both['mask_a']==[1.,1.,1.,0.] and sparse['mask_b']==[0.,0.,1.,0.])
ck('Equal sequence counts yield equal objective mass for equal density',both['sequence_counts']==[1,1] and both['weighted_target_share']==[.5,.5])
ck('Equal sequence counts with sparse B produce 3 to 1 loss mass',sparse['sequence_counts']==[1,1] and sparse['weighted_target_mass']==[3.,1.] and sparse['weighted_target_share']==[.75,.25])
ck('Same logits preserve mean loss despite changed objective shares',np.isclose(both['loss'],np.log(2),atol=1e-6) and both['loss']==sparse['loss'])
ck('Original method and reducer gradients expose changed per-domain coefficients',np.allclose(both['first_logit_gradient'],[.25,.25],atol=1e-6) and np.allclose(sparse['first_logit_gradient'],[.375,.125],atol=1e-6))
ck('Positive target count and weighted mass are distinct with fractional weights',scaled['nonzero_target_positions']==sparse['nonzero_target_positions'] and scaled['weighted_target_mass']==[3.,.5] and np.allclose(scaled['weighted_target_share'],[6/7,1/7]))
ck('Unequal losses use target-mass weighted mean not sequence-domain mean',np.isclose(different['loss'],1.25*np.log(2),atol=1e-6) and not np.isclose(different['loss'],1.5*np.log(2),atol=1e-6) and np.allclose(different['first_logit_gradient'],[.375,.1875],atol=1e-6))
ck('Uniform loss-weight scaling changes mass but not normalized objective or gradient',uniform['weighted_target_mass']==[1.5,.5] and uniform['nonzero_target_positions']==sparse['nonzero_target_positions'] and uniform['loss']==sparse['loss'] and np.allclose(uniform['first_logit_gradient'],sparse['first_logit_gradient'],atol=1e-6))
# Execute original integer-quota construction for both candidate sequence mixtures.
def quota(weights):
 ds=mix.Mix({'A':mix.Identity('A',100),'B':mix.Identity('B',100)},weights,8,key=7)
 return mix.counts(ds)
q0=quota({'A':.5,'B':.5});q1=quota({'A':.25,'B':.75})
ck('Original sampler integer quota matches equal and inverse-density sequence mixtures',q0=={'A':4,'B':4} and q1=={'A':2,'B':6})
ck('Inverse-density sequence mix balances objective mass in stationary fixture',comp['weighted_target_mass']==[6.,6.] and comp['weighted_target_share']==[.5,.5] and np.allclose(comp['first_logit_gradient'],[.25,.25],atol=1e-6))
ck('Same nominal input budget can contain less total target mass',comp['input_slots']==block_equal['input_slots']==32 and sum(comp['weighted_target_mass'])==12 and sum(block_equal['weighted_target_mass'])==16)
meta=json.loads((R/'sources/live_2026_10_07/meta.json').read_text());components=meta['config']['data']['value']['components'];declared={'snapshot':'sources/live_2026_10_07/meta.json','declared_components_count':len(components),'text_key_text_count':sum(c.get('format')=={'text_key':'text'} for c in components.values()),'actual_domain_target_density':None,'includes_training_and_evaluation_declarations':True}
ck('Archived formats do not establish an actual answer-only Hero mixture',declared['declared_components_count']==declared['text_key_text_count']==223)
ck('Backend substitution recorded rather than claimed as xla_fast_bwd execution',len(backend_calls)>0 and all(c['requested_implementation']=='xla_fast_bwd' and c['executed']=='original CPU reference' and c['z_loss'] is None for c in backend_calls))
paths=[pathlib.Path(__file__),E,G,M,D/'api.py',D/'reference.py',mix.P,R/'scripts/probe_mixture_identity_cpu.py',R/'sources/live_2026_10_07/meta.json',R/'analysis/loss_mass_source_binding.json']
out={'checks_passed':len(checks),'checks':checks,'cases':cases,'integer_quotas':{'equal':q0,'inverse_density':q1},'declared_formats':declared,'backend_calls':backend_calls,'scope':__doc__,'runtime':{'jax':jax.__version__,'jaxlib':__import__('jaxlib').__version__,'numpy':np.__version__,'backend':jax.default_backend()},'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'adapters':['AST extraction, original next_token_loss method on supplied one-hot hidden-state model adapter','Original Grug wrapper mesh=None; actual fused dispatcher replaced with explicit original CPU CE reference','Original CPU default logsumexp; z-loss disabled','Artificial prompt_length3 and fractional weight, no actual Hero field values','Original MixtureDataset with finite identity stores and null CPU mesh','Domain-disjoint projection rows make CE gradient coefficients separable; not general gradient-norm attribution'],'actual_Hero_target_mass_by_domain':None,'actual_Hero_gradient_share':None,'actual_hardware_compute_cost':None,'actual_causal_mixture_effect':None,'actual_xla_fast_bwd_execution':None}
(R/'analysis/loss_mass_cpu.json').write_text(json.dumps(out,indent=2)+'\n');print('Loss mass CPU:',len(checks),'passed')
