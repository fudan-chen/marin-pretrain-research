"""New exact-step public observations. Endpoint deltas are descriptive, not
mixture causal effects or statistical significance; no inherited-summary alignment.
"""
import json,pathlib,hashlib,csv,math
import numpy as np
import matplotlib
matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='marin-late-20261006'
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/live_2026_10_06';ev=json.loads((D/'eval.json').read_text());meta=json.loads((D/'meta.json').read_text());oldmeta=json.loads((R/'sources/wandb/hero-fa4sm100-nomask-step146k_meta.json').read_text());old=json.loads((R/'analysis/mix_trajectory.json').read_text());checks=[]
def ck(n,v):assert v,n;checks.append(n)
lookup={}
for spec,series in zip(ev['specs'],ev['series']):
 k=spec['keys'][-1]
 for x in series:
  if k not in x:continue
  key=(x['_step'],k);assert key not in lookup and math.isfinite(x[k]);lookup[key]=x[k]
steps=sorted({s for s,k in lookup});keys=sorted({k for s,k in lookup});subsets=sorted({k.split('/')[2] for k in keys if len(k.split('/'))==4})
ck('New snapshot retains six complete exact-step evaluations',steps==[194999,197999,200999,203999,206999,209999] and len(keys)==36 and all((s,k) in lookup for s in steps for k in keys))
prior={}
with (R/'analysis/mix_trajectory_points.csv').open() as f:
 for row in csv.DictReader(f):
  k='eval_dropless/paloma/'+('' if row['subset']=='__parent__' else row['subset']+'/')+row['metric'];prior[int(row['step']),k]=float(row['value'])
overlap=[k for k in lookup if k in prior];ck('All seventy-two overlapping values agree with frozen curve',len(overlap)==72 and all(lookup[k]==prior[k] for k in overlap))
ck('Sixteen evaluation subsets retain the archived names',subsets==old['subsets'])
a,b=197999,209999;changes=[]
for n in subsets:
 r={'subset':n}
 for m in ['loss','bpb']:
  key='eval_dropless/paloma/'+n+'/'+m;r[m+'_start']=lookup[a,key];r[m+'_end']=lookup[b,key];r[m+'_delta']=lookup[b,key]-lookup[a,key]
 changes.append(r)
ck('Fifteen endpoints improve CE and BPB, one worsens both',sum(r['loss_delta']<0 and r['bpb_delta']<0 for r in changes)==15 and [r['subset'] for r in changes if r['loss_delta']>0]==['twitterAAE_HELM_fixed-llama3'])
parents={m:{'start':lookup[a,'eval_dropless/paloma/'+m],'end':lookup[b,'eval_dropless/paloma/'+m],'delta':lookup[b,'eval_dropless/paloma/'+m]-lookup[a,'eval_dropless/paloma/'+m]} for m in ['macro_loss','micro_loss','macro_bpb','bpb']}
residuals=[]
for s in steps:
 for leaf,parent in [('loss','macro_loss'),('bpb','macro_bpb')]:
  residuals.append(lookup[s,'eval_dropless/paloma/'+parent]-sum(lookup[s,'eval_dropless/paloma/'+n+'/'+leaf] for n in subsets)/16)
ck('Equal-subset macro reconstructs at every new checkpoint',max(map(abs,residuals))<5e-7)
ptb=next(r for r in changes if r['subset']=='ptb-llama3');sensitivity={m:{'equal_subset_delta_without_ptb':sum(r[m+'_delta'] for r in changes if r['subset']!='ptb-llama3')/15,'ptb_share_of_equal_subset_delta':ptb[m+'_delta']/sum(r[m+'_delta'] for r in changes)} for m in ['loss','bpb']}
ck('Macro improvement also remains after excluding ptb',all(v['equal_subset_delta_without_ptb']<0 for v in sensitivity.values()))
twit=[lookup[s,'eval_dropless/paloma/twitterAAE_HELM_fixed-llama3/loss'] for s in steps if s>=a]
ck('Twitter endpoint increase is not monotonic deterioration',twit[-1]>twit[0] and any(y<x for x,y in zip(twit,twit[1:])))
ptbseries=[lookup[s,'eval_dropless/paloma/ptb-llama3/loss'] for s in steps if s>=a]
ck('PTB endpoint improvement is not monotonic improvement',ptbseries[-1]<ptbseries[0] and any(y>x for x,y in zip(ptbseries,ptbseries[1:])))
cfg=meta['config'];trainer=cfg['trainer']['value']['trainer'];batch=trainer['train_batch_size'];length=cfg['model']['value']['max_seq_len'];s=meta['summaryMetrics'];oldsum=oldmeta['summaryMetrics'];total=s['throughput/total_tokens'];update=s['_step']+1
ck('Latest declared configuration is unchanged and still 4K',cfg==oldmeta['config'] and length==4096)
ck('Summary nominal tokens equal completed steps times declared batch length',total==update*batch*length)
ck('Snapshot advanced beyond old summary',s['_step']>oldsum['_step'] and total>oldsum['throughput/total_tokens'])
ck('New evaluation endpoint precedes summary training step',b<s['_step'])
result={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'snapshot':{'run':meta['name'],'public_state':meta['state'],'heartbeat_at':meta['heartbeatAt'],'summary_step':s['_step'],'completed_updates':update,'nominal_tokens':total,'nominal_tokens_since_old_snapshot':total-oldsum['throughput/total_tokens'],'share_of_18T_target':total/18e12,'max_seq_len':length,'config_equal_to_old_snapshot':True,'summary_train_loss':s['train/loss'],'latest_complete_eval_step':b,'eval_lag_updates':s['_step']-b},'complete_steps':steps,'new_eval_steps':[x for x in steps if x not in old['complete_steps']],'overlap_values_unchanged':len(overlap),'window':{'start':a,'end':b,'updates':b-a,'nominal_positions':(b-a)*batch*length},'parent_deltas':parents,'subset_deltas':changes,'sensitivity':sensitivity,'max_macro_reconstruction_residual':max(map(abs,residuals)),'actual_eval_sample_identity':None,'actual_execution_SHA':None,'actual_mixture_counterfactual':None,'statistical_significance':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'meta.json',D/'eval.json',D/'dense.json']}}
(R/'analysis/live_analysis_2026_10_06.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
with (R/'analysis/live_subset_changes_2026_10_06.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(changes[0]));w.writeheader();w.writerows(changes)
fig,axes=plt.subplots(1,2,figsize=(12,7),sharey=True);yy=np.arange(16)
for ax,m,label in zip(axes,['loss','bpb'],['Cross entropy change','Logged BPB change']):
 vals=[r[m+'_delta'] for r in changes];ax.barh(yy,vals,color=['#b85137' if v>0 else '#245987' for v in vals]);ax.axvline(0,color='#555',lw=.6);ax.set_xlabel(label+' (end minus start)');ax.grid(axis='x',alpha=.15)
axes[0].set_yticks(yy,[n.replace('-llama3','') for n in subsets],fontsize=8);axes[0].invert_yaxis();fig.suptitle('Observed endpoints: step 197999 → 209999\n15 improve, 1 worsens; not a mixture causal effect',fontsize=12);fig.tight_layout();fig.savefig(R/'assets/live_2026_10_06.png',dpi=160);fig.savefig(R/'assets/live_2026_10_06.svg',metadata={'Date':None});plt.close(fig)
p=R/'assets/live_2026_10_06.svg';text=p.read_text();import re
ids=re.findall(r'id="([^"]+)"',text)
for id in sorted(set(ids),key=len,reverse=True):text=text.replace('id="'+id+'"','id="live-'+id+'"').replace('#'+id+'"','#live-'+id+'"').replace('#'+id+')','#live-'+id+')')
p.write_text(text)
print('Live observations:',len(checks),'checks; summary step',s['_step']);print(json.dumps(sensitivity,indent=2))
