"""Audit archived exact Paloma points; descriptive trajectory, no causal estimator."""
import pathlib,json,hashlib,csv,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt']='marin-mix-trajectory-v25'
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1]
lineage=[('hero-12d8b6f0-dee637',0,58014),('hero-wd-gate-router-p02-step58k',58014,81716),('hero-ragged_a2a-nccl2307-ep-step81k',81716,108000),('hero-mix-996f4891-step108k',108000,108778),('hero-nopdl-step108k',108778,121638),('hero-main-step121638',121638,146139),('hero-fa4sm100-nomask-step146k',146139,None)]
rows=[];census=[];hashes={}
for run,lo,hi in lineage:
 p=ROOT/'sources/wandb'/('%s_eval.json'%run)
 if not p.exists():
  census.append(dict(run=run,lo=lo,hi=hi,status='no_archived_eval_snapshot',raw_points=0,selected_points=0));continue
 hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 d=json.loads(p.read_text());raw=0;selected=0;excluded=set()
 assert len(d['specs'])==len(d['series'])
 for spec,series in zip(d['specs'],d['series']):
  key=spec['keys'][-1]
  if not key.startswith('eval_dropless/paloma/'):continue
  if key.split('/')[-1] not in ('loss','bpb','macro_bpb','macro_loss','micro_loss'):continue
  seen=set()
  for x in series:
   if key not in x:continue
   step=x['_step'];value=x[key];assert step not in seen and math.isfinite(value);seen.add(step);raw+=1
   if step<lo or (hi is not None and step>=hi):excluded.add(step);continue
   subset=key.split('/')[2] if len(key.split('/'))==4 else '__parent__';metric=key.split('/')[-1]
   rows.append(dict(step=step,run=run,subset=subset,metric=metric,value=value));selected+=1
 census.append(dict(run=run,lo=lo,hi=hi,status='archived_sampled_points',raw_points=raw,selected_points=selected,excluded_steps=sorted(excluded)))
lookup={}
for r in rows:
 k=(r['step'],r['subset'],r['metric']);assert k not in lookup;lookup[k]=r['value']
subsets=sorted(set(r['subset'] for r in rows if r['metric']=='loss'))
assert len(subsets)==16
steps=sorted(set(r['step'] for r in rows));complete=[s for s in steps if all((s,u,m) in lookup for u in subsets for m in ('loss','bpb'))]
assert complete==steps
windows=[(104999,107999,'pre_switch'),(107999,110999,'crosses_mix_and_execution_changes'),(110999,113999,'post_switch')]
comparisons=[]
for subset in subsets:
 out={'subset':subset}
 for a,b,label in windows:
  for m in ('loss','bpb'):
   out[label+'_'+m+'_delta']=lookup[b,subset,m]-lookup[a,subset,m]
 for m in ('loss','bpb'):
  out['late_'+m+'_delta']=lookup[197999,subset,m]-lookup[107999,subset,m]
  out['baseline_'+m]=lookup[107999,subset,m]
 comparisons.append(out)
summary={label:{m:{'decreased':sum(c[label+'_'+m+'_delta']<0 for c in comparisons),'increased':sum(c[label+'_'+m+'_delta']>0 for c in comparisons),'equal_subset_mean_delta':float(np.mean([c[label+'_'+m+'_delta'] for c in comparisons]))} for m in ('loss','bpb')} for _,_,label in windows}
sensitivity={}
for a,b,label in windows:
 deltas=[lookup[b,u,'loss']-lookup[a,u,'loss'] for u in subsets]
 sensitivity[label]={'median_subset_CE_delta':float(np.median(deltas)),'mean_CE_delta_without_ptb':float(np.mean([d for u,d in zip(subsets,deltas) if u!='ptb-llama3'])),'logged_micro_BPB_delta':lookup[b,'__parent__','bpb']-lookup[a,'__parent__','bpb'],'logged_macro_BPB_delta':lookup[b,'__parent__','macro_bpb']-lookup[a,'__parent__','macro_bpb'],'logged_macro_CE_delta':lookup[b,'__parent__','macro_loss']-lookup[a,'__parent__','macro_loss'],'logged_micro_CE_delta':lookup[b,'__parent__','micro_loss']-lookup[a,'__parent__','micro_loss']}
macro_reconstruction=[dict(step=s,CE_residual=lookup[s,'__parent__','macro_loss']-float(np.mean([lookup[s,u,'loss'] for u in subsets])),BPB_residual=lookup[s,'__parent__','macro_bpb']-float(np.mean([lookup[s,u,'bpb'] for u in subsets]))) for s in steps]
assert all((s,'__parent__',m) in lookup for s in steps for m in ('bpb','macro_bpb','macro_loss','micro_loss'))
audit=dict(macro_reconstruction=macro_reconstruction,sensitivity=sensitivity,subset_points=sum(r['subset']!='__parent__' for r in rows),parent_points=sum(r['subset']=='__parent__' for r in rows),macro_points=sum(r['subset']=='__parent__' and r['metric'].startswith('macro_') for r in rows),scope='exact archived sampled dropless Paloma points, lineage filtered, descriptive only',source_sha256=hashes,lineage_census=census,subsets=subsets,complete_steps=steps,selected_points=len(rows),windows=[dict(start=a,end=b,label=l) for a,b,l in windows],summary=summary,comparisons=comparisons,actual_mix_counterfactual=None,actual_cluster_to_paloma_mapping=None,actual_historical_eval_identity=None)
(ROOT/'analysis/mix_trajectory.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
for name,items in [('mix_trajectory_points.csv',rows),('mix_trajectory_changes.csv',comparisons)]:
 with (ROOT/'analysis'/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
fig,axes=plt.subplots(4,4,figsize=(13,10),sharex=True)
for ax,u in zip(axes.flat,subsets):
 xs=[s for s in steps if 95999<=s<=125999];ys=[lookup[s,u,'loss']-lookup[107999,u,'loss'] for s in xs]
 ax.plot(np.array(xs)/1000,ys,'o-',ms=3,lw=.9,color='#245987')
 ax.axhline(0,color='#777',lw=.5);ax.axvline(108,color='#b85137',lw=.8);ax.axvline(108.778,color='#a99b48',lw=.7,ls='--');ax.axvline(121.638,color='#888',lw=.7,ls=':')
 ax.set_title(u.replace('-llama3',''),fontsize=9);ax.grid(alpha=.15)
for ax in axes[:,0]:ax.set_ylabel('CE delta from 107999')
for ax in axes[-1,:]:ax.set_xlabel('Logged step / 1000')
fig.suptitle('Observed fixed-subset CE: mix switch 108000; execution handoffs 108778 / 121638',fontsize=11)
fig.text(.5,.012,'Exact archived dropless points; lines guide the eye. No interpolation or counterfactual estimate.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.025,1,.965]);fig.savefig(ROOT/'assets/mix_trajectory.svg',metadata={'Date':None});fig.savefig(ROOT/'assets/mix_trajectory.png',dpi=160);plt.close(fig)
print(json.dumps({'complete_checkpoints':len(steps),'selected_points':len(rows),'summary':summary,'census':census},indent=2))
