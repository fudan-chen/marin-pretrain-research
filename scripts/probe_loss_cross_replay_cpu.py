"""Descriptive four-cell replay on common synthetic targets; no causal Hero estimate."""
import hashlib,json
import jax.numpy as jnp
import numpy as np
import probe_loss_composition_cpu as base
R=base.R

def supported_allocation(cells, same_support):
 if not same_support or any(v is None or not np.isfinite(v) for v in cells.values()):
  return None
 return decompose(cells)

def decompose(c):
 a,b,c10,d=[c[x] for x in ['00','01','10','11']]
 composition=b-a;prediction=c10-a;interaction=d-c10-b+a
 return dict(total=d-a,composition_at_old_prediction=composition,prediction_at_old_weights=prediction,interaction=interaction,prediction_at_new_weights=d-b,composition_at_new_prediction=d-c10,symmetric_composition=(composition+d-c10)/2,symmetric_prediction=(prediction+d-b)/2)

def main():
 checks=[]
 def check(n,v):
  assert v,n
  checks.append(n)
 heads=[jnp.array([[2.,-1.]]),jnp.array([[-2.,-1.]])]
 tokens=jnp.array([[0,0,0,0],[1,1,1,1]])
 weights=[jnp.array([[1.,0,0,0],[1.,1,1,0]]),jnp.array([[1.,1,1,0],[1.,0,0,0]])]
 observations={}
 for objective,alpha in [('pure_ce',None),('ce_plus_output_z',1e-4)]:
  cells={}
  per_domain=[]
  for h in heads:
   x=np.asarray(h)[0].astype(float)
   ce=np.logaddexp(*x)-x
   penalty=0 if alpha is None else alpha*np.logaddexp(*x)**2
   per_domain.append((ce+penalty).tolist())
  for i,h in enumerate(heads):
   for j,w in enumerate(weights):
    cells[str(i)+str(j)]=float(base.ns['next_token_loss'](base.Provider(h),tokens,w,logsumexp_weight=alpha))
    mass=np.asarray(w).sum(axis=1)
    check(objective+' '+str(i)+str(j)+' matches target-weight denominator',abs(cells[str(i)+str(j)]-np.dot(mass,per_domain[i])/mass.sum())<1e-6)
  d=decompose(cells)
  check(objective+' baseline allocation closes',abs(d['total']-d['composition_at_old_prediction']-d['prediction_at_old_weights']-d['interaction'])<1e-12)
  check(objective+' symmetric allocation closes',abs(d['total']-d['symmetric_composition']-d['symmetric_prediction'])<1e-12)
  observations[objective]={'cells':cells,'allocation':d,'per_domain_objective':per_domain}
 p=observations['pure_ce']
 check('Observed training-weight change conceals fixed-old-weight improvement',p['allocation']['total']<0 and p['allocation']['prediction_at_old_weights']<0 and p['allocation']['prediction_at_new_weights']>0)
 check('Prediction contribution changes with reference distribution',abs(p['allocation']['prediction_at_old_weights']-p['allocation']['prediction_at_new_weights']-(-p['allocation']['interaction']))<1e-12)
 check('Nonzero interaction forbids two baseline deltas alone',abs(p['allocation']['interaction'])>.4)
 z=observations['ce_plus_output_z']
 penalty_cells={k:z['cells'][k]-p['cells'][k] for k in p['cells']}
 check('Output penalty separately reconciles all cells',all(v>0 for v in penalty_cells.values()))
 missing={'00':p['cells']['00'],'01':None,'10':p['cells']['10'],'11':p['cells']['11']}
 check('Missing prediction on common support leaves allocation unavailable',supported_allocation(missing,True) is None)
 check('Different target support prevents numerical allocation',supported_allocation(p['cells'],False) is None)
 sources={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [base.D/'api.py',base.D/'reference.py',base.G,base.M,R/'scripts/probe_loss_composition_cpu.py']}
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'observations':observations,'output_penalty_cells':penalty_cells,'weighted_target_mass':[[1,3],[3,1]],'denominators':[4,4],'common_support':{'target_ids':['A:0','A:1','A:2','B:0','B:1','B:2'],'same_labels_and_context':True,'both_predictions_available':True},'missing_cell_control':{'cells':missing,'allocation':None},'explicit_substitutions':['V72 fixed hidden/router provider','CPU original reference replaces GPU dispatch','synthetic heads and targets, no parameter update'], 'source_sha256':sources,'actual_Hero_loss_attribution':None,'actual_mixture_training_experiment':None}
 (R/'analysis/loss_cross_replay_cpu.json').write_text(json.dumps(o,indent=2)+'\n')
 print(json.dumps({'checks':len(checks),'pure_ce':p},indent=2))
if __name__=='__main__':main()
