"""Algebraic identifiability audit of the 67 archived Paloma timestamps.
No causal effect fit. Synthetic inserted timestamps demonstrate design rank only.
"""
import csv,hashlib,json,pathlib
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];CSV=R/'analysis/mix_trajectory_points.csv';TRAJ=R/'analysis/mix_trajectory.json';COMMENTS=R/'sources/engineering_current_2026_10_07/comments_8506.json'
rows=list(csv.DictReader(CSV.open()));a=json.loads(TRAJ.read_text());steps=np.array(sorted(set(int(r['step']) for r in rows)),dtype=np.int64);comments={c['id']:c for c in json.loads(COMMENTS.read_text())}
events={'mix':{'step':108000,'comment_id':5684891157},'execution':{'step':108778,'comment_id':5689628437}}
def design(ss):
 ss=np.asarray(ss);m=(ss>=108000).astype(float);e=(ss>=108778).astype(float);t=(ss-108000)/1000.
 return np.column_stack([np.ones(len(ss)),m,e]),np.column_stack([np.ones(len(ss)),t,m,e,np.maximum(t,0),np.maximum(t-.778,0)])
def main():
 checks=[]
 def check(n,v):assert v,n;checks.append(n)
 L,S=design(steps);null_level=np.array([0,1,-1.]);null_slope=np.array([0,0,-.778,0,1,-1.]);interval=steps[(steps>=108000)&(steps<108778)]
 check('67 complete inherited-history-filtered timestamps match original audit',len(steps)==67 and steps.tolist()==a['complete_steps'] and len(rows)==a['selected_points'])
 check('Frozen source comments bind both step boundaries','step-108000' in comments[5684891157]['body'] and 'step-108778' in comments[5689628437]['body'])
 check('No retained archived evaluation timestamp separates events',len(interval)==0 and max(steps[steps<108000])==107999 and min(steps[steps>=108778])==110999)
 check('Two event level columns are exactly identical',np.array_equal(L[:,1],L[:,2]) and np.linalg.matrix_rank(L)==2)
 check('Explicit level null direction leaves all predictions unchanged',np.max(np.abs(L@null_level))==0)
 check('Segmented level/slope design is rank four out of six',np.linalg.matrix_rank(S)==4 and np.max(np.abs(S@null_slope))<1e-12)
 n=np.array([0,0,1,-1,0,0.]);check('Segmented level null direction is exact',np.max(np.abs(S@n))==0)
 # Arbitrary coefficients are an algebraic control, not fitted effects.
 beta=np.array([0.,-.015625,.03125]);alternatives=[beta+u*null_level for u in [-1.,0.,1.]]
 check('Arbitrary opposite effect allocations yield identical predictions',all(np.array_equal(L@alternatives[0],L@b) for b in alternatives[1:]))
 augmented={}
 for name,extra in [('one_intermediate',[108500]),('two_intermediate',[108100,108700])]:
  l,s=design(sorted(steps.tolist()+extra));augmented[name]={'synthetic_inserted_steps':extra,'level_rank':int(np.linalg.matrix_rank(l)),'segmented_rank':int(np.linalg.matrix_rank(s))}
 check('Synthetic one/two intermediate controls improve algebraic rank only',augmented['one_intermediate']['level_rank']==3 and augmented['one_intermediate']['segmented_rank']==5 and augmented['two_intermediate']['segmented_rank']==6)
 columns=['intercept','mix_after','execution_after'];segcolumns=['intercept','time_1000','mix_after','execution_after','mix_post_time_1000','execution_post_time_1000']
 with (R/'analysis/mix_event_design.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['step']+segcolumns);w.writerows([[int(s)]+S[i].tolist() for i,s in enumerate(steps)])
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'events':events,'retained_steps':steps.tolist(),'intermediate_retained_steps':interval.tolist(),'nearest_pre_step':107999,'nearest_post_step':110999,'level_design':{'columns':columns,'rank':2,'column_count':3,'null_vector':null_level.tolist(),'null_residual':float(np.max(np.abs(L@null_level)))},'segmented_design':{'columns':segcolumns,'rank':4,'column_count':6,'slope_null_vector':null_slope.tolist(),'null_residual':float(np.max(np.abs(S@null_slope)))},'arbitrary_coefficient_controls_not_estimates':[{'coefficients':b.tolist(),'prediction_vector':(L@b).tolist()} for b in alternatives],'synthetic_design_controls':augmented,'actual_independent_mix_effect':None,'actual_independent_execution_effect':None,'actual_counterfactual_training':None,'actual_historical_eval_identity':None,'limitations':['Only archived retained timestamps; no claim that the live service has no intermediate evals.','Rank failure concerns this additive event level/slope specification, not a theorem about all possible mechanistic models.','Full rank alone would not establish causal identification, parallel controls, or matching historical evaluation identity.','Other deployment events, learning dynamics and data-dependent response remain confounders.'],'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CSV,TRAJ,COMMENTS]}}
 (R/'analysis/mix_event_identifiability.json').write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'checks_passed':len(checks),'level_rank':2,'segmented_rank':4,'synthetic_controls':augmented},indent=2))
if __name__=='__main__':main()
