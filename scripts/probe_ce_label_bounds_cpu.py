"""Pinned original CE reference/streaming/reducer and input validator: label bounds CPU controls."""
import ast,json,pathlib,hashlib
import jax,jax.numpy as jnp,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/contracts_2026_10_05'
ns={'jax':jax,'jnp':jnp}
def extract(path,names):
 nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert len(nodes)==len(names)
 for n in nodes:n.decorator_list=[]
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(path),'exec'),ns)
extract(D/'api.py',['_validate_inputs','_apply_reduction'])
extract(D/'reference.py',['_default_logsumexp','_apply_logit_soft_cap','linear_softmax_cross_entropy_loss_reference','linear_softmax_cross_entropy_loss_streaming'])
ns['_cross_entropy_logsumexp']=ns['_default_logsumexp'];ns['_cross_entropy_logaddexp']=jnp.logaddexp
ref=ns['linear_softmax_cross_entropy_loss_reference'];stream=ns['linear_softmax_cross_entropy_loss_streaming'];reduce=ns['_apply_reduction'];validate=ns['_validate_inputs']
head=jnp.array([[0.,1.,2.]],jnp.float32);checks=[];cases={}
def ck(name,v):
 if not v:raise RuntimeError(name)
 checks.append({'name':name,'passed':True})
def scalar(v):
 v=float(v);return v if np.isfinite(v) else 'nan' if np.isnan(v) else 'inf' if v>0 else '-inf'
def loss_fn(labels,weight,backend):
 labels=jnp.array(labels,jnp.int32);weight=jnp.array(weight,jnp.float32);x=jnp.ones((len(labels),1),jnp.float32)
 def fn(w):
  loss,_=ref(x,labels,w) if backend=='full' else stream(x,labels,w,block_size=2)
  return reduce(loss,'mean',weight)
 return fn,x,labels
for label in [0,2,-1,3,100,-100]:
 row={}
 for backend in ['full','streaming']:
  fn,x,lab=loss_fn([label],[1.],backend);validate(x,lab,head)
  v,g=jax.value_and_grad(fn)(head)
  row[backend]={'loss':scalar(v),'loss_finite':bool(jnp.isfinite(v)),'gradient':[[scalar(a) for a in rr] for rr in np.asarray(g)],'gradient_sum':scalar(jnp.sum(g)),'validator_accepted':True}
 cases[str(label)]=row
for backend in ['full','streaming']:
 fn,_,_=loss_fn([0,3],[1.,0.],backend);v,g=jax.value_and_grad(fn)(head)
 cases['masked_invalid_'+backend]={'loss':scalar(v),'gradient':[[scalar(a) for a in rr] for rr in np.asarray(g)],'gradient_finite':bool(jnp.all(jnp.isfinite(g)))}
fn,_,_=loss_fn([3],[1.],'full');eps=.01
shift_diff=scalar((fn(head+eps)-fn(head-eps))/(2*eps));jv,jg=jax.jit(jax.value_and_grad(fn))(head)
cases['positive_oob_shift']={'finite_difference_directional':shift_diff,'autodiff_directional':scalar(jnp.sum(jg)),'jit_loss':scalar(jv),'jit_gradient':np.asarray(jg).tolist(),'epsilon':eps}
def prepare(labels,weights,vocab):
 labels=np.asarray(labels);weights=np.asarray(weights)
 if not np.issubdtype(labels.dtype,np.integer):raise TypeError('labels must be integer before conversion')
 if labels.shape!=weights.shape:raise ValueError('label and weight shapes differ')
 bad=(labels<0)|(labels>=vocab)
 if np.any(bad & (weights!=0)):raise ValueError('active target outside vocabulary')
 return np.where(weights==0,0,labels).astype(np.int32)
rejections={}
for label in [-1,3,100,-100]:
 try:prepare([label],[1.],3)
 except ValueError as e:rejections[str(label)]=type(e).__name__
cast_rejections={}
for name,labels in [('fractional',[2.9]),('wide_integer',[2**32])]:
 try:prepare(labels,[1.],3)
 except (TypeError,ValueError) as e:cast_rejections[name]=type(e).__name__
safe=prepare([0,3],[1.,0.],3);prepared={}
for backend in ['full','streaming']:
 fn,_,_=loss_fn(safe,[1.,0.],backend);v,g=jax.value_and_grad(fn)(head);prepared[backend]={'loss':scalar(v),'gradient':np.asarray(g).tolist()}
cases['candidate_sanitized_inactive']={'labels':safe.tolist(),'backends':prepared}
ck('Valid target CPU backends agree in loss and gradient',all(np.isclose(cases[str(i)]['full']['loss'],cases[str(i)]['streaming']['loss'],rtol=2e-6) and np.allclose(cases[str(i)]['full']['gradient'],cases[str(i)]['streaming']['gradient'],atol=2e-7) for i in [0,2]))
ck('Original shape dtype validator accepts all selected invalid integer labels',all(cases[str(i)][b]['validator_accepted'] for i in [-1,3,100,-100] for b in ['full','streaming']))
ck('Positive out of vocabulary full reference returns finite last-class loss',all(np.isclose(cases[str(i)]['full']['loss'],cases['2']['full']['loss']) for i in [3,100]))
ck('Positive out of vocabulary full reference gradient lacks target subtraction',all(np.isclose(cases[str(i)]['full']['gradient_sum'],1.,atol=2e-7) for i in [3,100]) and np.isclose(cases['2']['full']['gradient_sum'],0.,atol=2e-7))
ck('Full reference invalid forward shift derivative differs from autodiff',abs(shift_diff)<1e-4 and np.isclose(cases['positive_oob_shift']['autodiff_directional'],1.,atol=2e-7))
ck('JIT preserves selected positive invalid finite forward and gradient gap',np.isclose(scalar(jv),cases['3']['full']['loss']) and np.allclose(np.asarray(jg),cases['3']['full']['gradient']))
ck('Negative one wraps in full reference but is infinite in streaming',np.isclose(cases['-1']['full']['loss'],cases['2']['full']['loss']) and cases['-1']['streaming']['loss']=='inf')
ck('Selected positive invalid labels yield streaming infinite loss',all(cases[str(i)]['streaming']['loss']=='inf' for i in [3,100]))
ck('Masking invalid streaming target after CE does not avoid NaN loss',cases['masked_invalid_streaming']['loss']=='nan' and isinstance(cases['masked_invalid_full']['loss'],float))
ck('Inactive sanitized labels restore backend agreement',safe.tolist()==[0,0] and np.isclose(prepared['full']['loss'],prepared['streaming']['loss'],rtol=2e-6) and np.allclose(prepared['full']['gradient'],prepared['streaming']['gradient'],atol=2e-7))
ck('Offline candidate rejects selected active invalid labels',len(rejections)==4 and all(v=='ValueError' for v in rejections.values()))
ck('Candidate checks labels before narrowing casts',cast_rejections=={'fractional':'TypeError','wide_integer':'ValueError'})
ck('Offline candidate preserves valid target labels',np.array_equal(prepare([0,2],[1.,1.],3),np.array([0,2],np.int32)))
ck('Sanitized masked case agrees with single valid target',np.isclose(prepared['full']['loss'],cases['0']['full']['loss'],rtol=2e-6) and np.allclose(prepared['full']['gradient'],cases['0']['full']['gradient'],atol=2e-7))
paths=[D/'reference.py',D/'api.py',D/'model.py'];out={'scope':'Original shape validator, full and streaming reference CE and reducer on tiny CPU tensors; autodiff and JIT, not fused dispatcher, GPU or production tokenizer input.','checks_passed':len(checks),'checks':checks,'cases':cases,'candidate_rejections':rejections,'candidate_cast_rejections':cast_rejections,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'probe_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'runtime':{'jax':jax.__version__,'numpy':np.__version__,'backend':jax.default_backend()},'substitutions':['Select original default CPU logsumexp','Select jnp.logaddexp CPU branch; no TPU accuracy branch or fused dispatcher'],'official_background':'https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html','candidate_scope':'Offline NumPy label check and explicit inactive-label replacement only; not full input validator or production JIT hook','actual_Hero_invalid_target_event':None,'actual_fused_GPU_execution':None,'actual_training_loss_effect':None,'upstream_patch_applied':False}
(R/'analysis/ce_label_bounds_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print('CE label bounds controls:',len(checks))
