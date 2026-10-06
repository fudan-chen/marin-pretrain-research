"""Conditional affine fit of observed eval aggregates; not actual denominator recovery.
The chronological holdout is post-hoc. Numerical perturbations are not confidence intervals.
"""
import csv,json,pathlib,hashlib
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];RUN='hero-fa4sm100-nomask-step146k';checks=[]
def ck(n,v):assert v,n;checks.append(n)
def fit(X,y):
 X=np.asarray(X,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
 if X.ndim!=2 or y.shape!=(X.shape[0],) or X.shape[1]<2 or not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('finite matrix/vector with matching rows required')
 A=X[:,:-1]-X[:,[-1]];b=y-X[:,-1];z,_,rank,sv=np.linalg.lstsq(A,b,rcond=None)
 if rank!=X.shape[1]-1:raise ValueError('affine weights not identifiable under fixed-weight assumption')
 w=np.r_[z,1-z.sum()]
 return {'weights':w,'rank':int(rank),'condition':float(sv[0]/sv[-1]),'max_residual':float(np.max(abs(X@w-y)))}
p={};paths=[R/'analysis/mix_trajectory_points.csv']
def put(key,value):
 if key in p:assert p[key]==value,key
 p[key]=value
for r in csv.DictReader(paths[0].open()):
 if r['run']==RUN:put((int(r['step']),r['subset'],r['metric']),float(r['value']))
for day in ['06','07']:
 f=R/f'sources/live_2026_10_{day}/eval.json';paths.append(f);j=json.loads(f.read_text());assert j['run']==RUN
 for spec,rows in zip(j['specs'],j['series']):
  key=spec['keys'][-1];a=key.split('/');n=a[2] if len(a)==4 else '__parent__';m=a[-1]
  for r in rows:
   if key in r:put((r['_step'],n,m),r[key])
names=sorted({n for s,n,m in p if n!='__parent__'});steps=sorted({s for s,n,m in p});required=[(n,m) for n in names for m in ['loss','bpb']]+[('__parent__',m) for m in ['micro_loss','bpb']];complete=[s for s in steps if all((s,n,m) in p for n,m in required)];ck('Same named branch yields 23 complete sixteen-domain observations',len(names)==16 and len(complete)==23)
train=[s for s in complete if s<=197999];hold=[s for s in complete if s>197999];ck('Post-hoc chronological partition uses 18 earlier and five later steps',len(train)==18 and hold==[200999,203999,206999,209999,212999])
metrics=[('loss','micro_loss'),('bpb','bpb')]
def matrix(ss,metrics):
 return np.array([[p[s,n,m] for n in names] for s in ss for m,parent in metrics]),np.array([p[s,'__parent__',parent] for s in ss for m,parent in metrics])
X,y=matrix(train,metrics);H,hy=matrix(hold,metrics);f=fit(X,y);w=f['weights'];ck('Fitted affine weights are positive and sum to one',np.all(w>0) and abs(w.sum()-1)<1e-14)
held=float(np.max(abs(H@w-hy)));allX,ally=matrix(complete,metrics);allfit=fit(allX,ally)
families=[]
for mm in [[metrics[0]],[metrics[1]]]:
 a,b=matrix(train,mm);ff=fit(a,b);families.append({'metric':mm[0][0],'weights':ff['weights'].tolist(),'max_difference_from_joint':float(max(abs(ff['weights']-w))),'rank':ff['rank'],'condition':ff['condition']})
rng=np.random.default_rng(20261007);ws=np.array([fit(X+rng.uniform(-2e-7,2e-7,X.shape),y+rng.uniform(-2e-7,2e-7,y.shape))['weights'] for _ in range(100)])
idx=names.index('ptb-llama3');a,b=209999,212999;delta=np.array([p[b,n,'loss']-p[a,n,'loss'] for n in names]);contributions=w*delta
B=np.vstack([np.ones(16),[p[b,n,'loss'] for n in names],[p[b,n,'bpb'] for n in names]]);_,sing,V=np.linalg.svd(B,full_matrices=True);null=V[3];valid=abs(null)>1e-12;eps=.5*min(w[valid]/abs(null[valid]));alt1=w+eps*null;alt2=w-eps*null
ck('Distinct positive weights share exactly the same predicted latest aggregates',min(alt1)>0 and min(alt2)>0 and max(abs(B@alt1-B@alt2))<1e-13 and max(abs(alt1-alt2))>.01)
# Independent known-weight system recovery and rank/shape failures.
z=np.random.default_rng(7).uniform(.5,3,(30,4));truth=np.array([.1,.2,.3,.4]);recovered=fit(z,z@truth)['weights'];ck('Known synthetic affine system recovers declared weights',max(abs(recovered-truth))<1e-13)
for label,xx,yy in [('rank deficient',np.ones((20,4)),np.ones(20)),('nonfinite',np.array([[np.nan,1],[2,3]]),np.ones(2)),('shape mismatch',np.ones((3,2)),np.ones(2))]:
 try:fit(xx,yy);bad=False
 except ValueError:bad=True
 ck('Reject '+label,bad)
paths.append(R/'sources/scale_2026_10_05/eval.py')
result={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'run':RUN,'subsets':names,'fit_steps':train,'posthoc_holdout_steps':hold,'weights':dict(zip(names,map(float,w))),'fit_rank':f['rank'],'affine_condition_number':f['condition'],'max_fit_residual':f['max_residual'],'max_posthoc_holdout_residual':held,'all23_max_residual':allfit['max_residual'],'all23_weights':dict(zip(names,map(float,allfit['weights']))),'separate_metric_fits':families,'numerical_perturbation':{'seed':20261007,'trials':100,'uniform_amplitude':2e-7,'is_confidence_interval':False,'max_absolute_weight_shift':float(np.max(abs(ws-w))),'ptb_min':float(ws[:,idx].min()),'ptb_max':float(ws[:,idx].max())},'window':{'start':a,'end':b,'conditional_micro_CE_delta':float(contributions.sum()),'observed_micro_CE_delta':p[b,'__parent__','micro_loss']-p[a,'__parent__','micro_loss'],'conditional_ptb_micro_contribution':float(contributions[idx]),'ptb_macro_contribution':float(delta[idx]/16),'ptb_macro_over_conditional_micro':float((1/16)/w[idx])},'conditional_contributions':dict(zip(names,map(float,contributions))),'latest_nonidentifiability':{'weight_set_one':alt1.tolist(),'weight_set_two':alt2.tolist(),'max_weight_difference':float(max(abs(alt1-alt2))),'max_predicted_parent_difference':float(max(abs(B@alt1-B@alt2)))},'actual_domain_denominators':None,'actual_fixed_weight_identity':None,'actual_Hero_input_identity':None,'actual_mixture_causal_effect':None,'source_sha256':{str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}}
(R/'analysis/eval_weight_inference.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('Conditional aggregate checks:',len(checks),'holdout residual:',held)
# One conditional estimate versus the macro convention; never label as measured counts.
import matplotlib,re
matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='marin-eval-affine-20261007'
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,6.7));yy=np.arange(16);ax.barh(yy,100*w,color=['#b85137' if n=='ptb-llama3' else '#245987' for n in names]);ax.axvline(6.25,color='#444',linestyle='--',lw=1,label='Macro: 6.25% per subset');ax.set_yticks(yy,[n.replace('-llama3','') for n in names],fontsize=8);ax.invert_yaxis();ax.set_xlabel('Conditional micro weight estimate (%)');ax.set_title('Fixed-weight affine fit, 18 earlier eval steps\nActual denominators unobserved; post-hoc holdout is not an independent trial',fontsize=11);ax.legend(loc='lower right',fontsize=8);ax.grid(axis='x',alpha=.15);fig.tight_layout();fig.savefig(R/'assets/eval_weight_inference.png',dpi=150);fig.savefig(R/'assets/eval_weight_inference.svg',metadata={'Date':None});plt.close(fig)
f=R/'assets/eval_weight_inference.svg';s=f.read_text()
for id in sorted(set(re.findall(r'id="([^"]+)"',s)),key=len,reverse=True):s=s.replace('id="'+id+'"','id="evalweight-'+id+'"').replace('#'+id+'"','#evalweight-'+id+'"').replace('#'+id+')','#evalweight-'+id+')')
f.write_text('\n'.join(x.rstrip() for x in s.splitlines())+'\n')
