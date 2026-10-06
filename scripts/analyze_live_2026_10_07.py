"""A separately dated exact-step observation, not a mixture intervention or error estimate."""
import pathlib,json,hashlib,math
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/live_2026_10_07';oldD=R/'sources/live_2026_10_06';checks=[]
def ck(n,v):assert v,n;checks.append(n)
def points(folder):
 j=json.loads((folder/'eval.json').read_text());out={}
 for spec,series in zip(j['specs'],j['series']):
  k=spec['keys'][-1]
  for x in series:
   if k in x:
    t=(x['_step'],k);assert t not in out and math.isfinite(x[k]);out[t]=x[k]
 return out
p=points(D);old=points(oldD);meta=json.loads((D/'meta.json').read_text());om=json.loads((oldD/'meta.json').read_text());keys=sorted({k for s,k in p});ss=sorted({s for s,k in p});complete=[s for s in ss if all((s,k) in p for k in keys)];ck('Public request preserves thirty-six keys and complete exact-step intersections',len(keys)==36 and len(complete)>=2)
overlap=p.keys()&old.keys();ck('Every retrieved old evaluation value is unchanged',len(overlap)>0 and all(p[x]==old[x] for x in overlap));newsteps=[s for s in complete if s not in {s for s,k in old}];ck('Snapshot adds a complete evaluation beyond old endpoint',len(newsteps)>0 and max(complete)>209999)
a=209999;b=max(complete);ck('Endpoints use actual exact-step fields',all((s,k) in p for s in [a,b] for k in keys))
subsets=sorted({k.split('/')[2] for k in keys if len(k.split('/'))==4});ck('Sixteen source subsets retain old names',len(subsets)==16 and subsets==sorted({k.split('/')[2] for s,k in old if len(k.split('/'))==4}))
changes=[]
for n in subsets:
 r={'subset':n}
 for m in ['loss','bpb']:
  k='eval_dropless/paloma/'+n+'/'+m;r[m+'_start']=p[a,k];r[m+'_end']=p[b,k];r[m+'_delta']=p[b,k]-p[a,k]
 changes.append(r)
parents={m:{'start':p[a,'eval_dropless/paloma/'+m],'end':p[b,'eval_dropless/paloma/'+m],'delta':p[b,'eval_dropless/paloma/'+m]-p[a,'eval_dropless/paloma/'+m]} for m in ['macro_loss','micro_loss','macro_bpb','bpb']}
res=[p[s,'eval_dropless/paloma/'+parent]-sum(p[s,'eval_dropless/paloma/'+n+'/'+m] for n in subsets)/16 for s in complete for m,parent in [('loss','macro_loss'),('bpb','macro_bpb')]];ck('Macro values reconstruct from fixed sixteen subset means',max(map(abs,res))<5e-7)
cfg=meta['config'];s=meta['summaryMetrics'];length=cfg['model']['value']['max_seq_len'];batch=cfg['trainer']['value']['trainer']['train_batch_size'];ck('Declared recipe stays byte-value equal to Oct6 configuration',cfg==om['config']);ck('Summary nominal token clock matches declared batch and length',s['throughput/total_tokens']==(s['_step']+1)*batch*length);ck('Summary and latest complete eval remain separate',b<s['_step'])
ptb=next(x for x in changes if x['subset']=='ptb-llama3');sens={m:{'equal_subset_delta_without_ptb':sum(x[m+'_delta'] for x in changes if x['subset']!='ptb-llama3')/15,'ptb_delta':ptb[m+'_delta']} for m in ['loss','bpb']}
result={'checks_passed':len(checks),'checks':checks,'scope':__doc__,'snapshot':{'run':meta['name'],'state':meta['state'],'heartbeat_utc':meta['heartbeatAt'],'summary_step':s['_step'],'nominal_tokens':s['throughput/total_tokens'],'nominal_tokens_since_oct6':s['throughput/total_tokens']-om['summaryMetrics']['throughput/total_tokens'],'share_of_18T':s['throughput/total_tokens']/18e12,'max_seq_len':length,'config_equal_to_oct6':True,'latest_complete_eval_step':b,'eval_lag_updates':s['_step']-b},'complete_steps':complete,'partial_steps':[s for s in ss if s not in complete],'new_complete_eval_steps':newsteps,'overlap_values_unchanged':len(overlap),'window':{'start':a,'end':b,'updates':b-a,'nominal_positions':(b-a)*batch*length},'parent_deltas':parents,'subset_deltas':changes,'sensitivity':sens,'improving_CE_endpoints':sum(x['loss_delta']<0 for x in changes),'worsening_CE_endpoints':sum(x['loss_delta']>0 for x in changes),'source_sha256':{str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [D/'meta.json',D/'eval.json',oldD/'meta.json',oldD/'eval.json']},'actual_eval_input_identity':None,'actual_execution_SHA':None,'actual_mixture_counterfactual':None,'statistical_significance':None}
(R/'analysis/live_analysis_2026_10_07.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['snapshot','new_complete_eval_steps','overlap_values_unchanged','parent_deltas','sensitivity','improving_CE_endpoints','worsening_CE_endpoints']},indent=2))
# A new separately dated figure; older observation plots stay untouched.
import numpy as np,matplotlib,re
matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='marin-20261007-endpoints'
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,7),sharey=True);y=np.arange(len(changes))
for ax,m,label in zip(axes,['loss','bpb'],['CE change','Logged BPB change']):
 values=[x[m+'_delta'] for x in changes];ax.barh(y,values,color=['#b85137' if x>0 else '#245987' for x in values]);ax.axvline(0,color='#444',lw=.6);ax.set_xlabel(label+' (end minus start)');ax.grid(axis='x',alpha=.15)
axes[0].set_yticks(y,[x['subset'].replace('-llama3','') for x in changes],fontsize=8);axes[0].invert_yaxis();fig.suptitle(f'Observed endpoints: step {a} → {b}\nMacro CE rises while micro CE falls; not a mixture causal effect',fontsize=12);fig.tight_layout();fig.savefig(R/'assets/live_2026_10_07.png',dpi=150);fig.savefig(R/'assets/live_2026_10_07.svg',metadata={'Date':None});plt.close(fig)
f=R/'assets/live_2026_10_07.svg';s=f.read_text()
for id in sorted(set(re.findall(r'id="([^"]+)"',s)),key=len,reverse=True):s=s.replace('id="'+id+'"','id="oct7-'+id+'"').replace('#'+id+'"','#oct7-'+id+'"').replace('#'+id+')','#oct7-'+id+')')
f.write_text('\n'.join(x.rstrip() for x in s.splitlines())+'\n')
