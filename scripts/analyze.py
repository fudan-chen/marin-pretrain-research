import json,pathlib,csv,re,statistics,datetime,hashlib
from bs4 import BeautifulSoup
ROOT=pathlib.Path(__file__).resolve().parents[1];S=ROOT/'sources';A=ROOT/'analysis';A.mkdir(exist_ok=True)
def save(name,data): (A/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))
soup=BeautifulSoup((S/'data_overview.html').read_text(),'html.parser')
names={int(x.select_one('.cluster-id').text):x.select_one('summary strong').text for x in soup.select('details.cluster')}
raw=json.loads((S/'harrier_spec_12d8b6f0.json').read_text());avail=raw['available_tokens'];total=sum(avail.values())
meta=json.loads((S/'wandb/hero-fa4sm100-nomask-step146k_meta.json').read_text());config=meta['config'];data=config['data']['value'];model=config['model']['value'];t=config['trainer']['value']['trainer'];steps=t['num_train_steps'];batch=t['train_batch_size'];tpb=batch*model['max_seq_len'];stages=data['train_weights']
weights=[{k:v for k,v in w.items() if re.fullmatch(r'c\d\dq\d',k)} for _,w in stages]
assert len(weights)==3
for w in weights:assert len(w)==200 and abs(sum(w.values())-1)<1e-9
bounds=[a for a,_ in stages]+[steps];budgets=[(bounds[i+1]-bounds[i])*tpb for i in range(3)]
quality=[]
for q in range(5):
 keys=[k for k in avail if k[-1]==str(q)]
 quality.append({'q':q,'store_tokens':sum(avail[k] for k in keys),'store_pct':100*sum(avail[k] for k in keys)/total,**{'phase%d_pct'%i:100*sum(weights[i][k] for k in keys) for i in range(3)}})
domains=[]
for d in range(40):
 keys=['c%02dq%d'%(d,q) for q in range(5)]
 r={'id':d,'name':names[d],'store_tokens':sum(avail[k] for k in keys),'store_pct':100*sum(avail[k] for k in keys)/total}
 for i in range(3):r['phase%d_pct'%i]=100*sum(weights[i][k] for k in keys)
 r['delta_main_pp']=r['phase1_pct']-r['phase0_pct'];r['delta_cool_pp']=r['phase2_pct']-r['phase1_pct'];r['initial_vs_store']=r['phase0_pct']/r['store_pct'];domains.append(r)
cells=[]
examples=[]
for d in range(40):
 cl=soup.select_one('#cluster-%d'%d)
 for q in range(5):
  k='c%02dq%d'%(d,q);sec=cl.select_one('.quality.q%d'%q)
  tok=sum(b*w[k] for b,w in zip(budgets,weights));row={'cell':k,'domain_id':d,'domain':names[d],'quality':q,'available_tokens':avail[k],'store_pct':100*avail[k]/total,'planned_tokens':tok,'planned_epochs':tok/avail[k]}
  for i,w in enumerate(weights):row['phase%d_pct'%i]=100*w[k]
  cells.append(row)
  texts=[x.select_one('pre').get_text() if x.select_one('pre') else x.get_text() for x in sec.select('details.example')]
  examples.append({'cell':k,'examples':texts,'source_legends':[x.get_text(' ',strip=True) for x in sec.select('.bucket-source-legend li')]})
actual={}
for p in sorted((S/'wandb').glob('*_meta.json')):
 m=json.loads(p.read_text());conf=m['config'];w=conf['data']['value']['train_weights'];actual[p.name[:-10]]={'stages':[[x,len(y),sum(y.values())] for x,y in w],'state':m['state'],'heartbeat':m['heartbeatAt'],'summary_step':m['summaryMetrics'].get('global_step'),'seq_len':conf['model']['value']['max_seq_len']}
lineage=[('hero-12d8b6f0-dee637',0,58014),('hero-wd-gate-router-p02-step58k',58014,81716),('hero-ragged_a2a-nccl2307-ep-step81k',81716,108000),('hero-mix-996f4891-step108k',108000,108778),('hero-nopdl-step108k',108778,121638),('hero-main-step121638',121638,146139),('hero-fa4sm100-nomask-step146k',146139,10**9)]
series={};measurements=[]
for run,lo,hi in lineage:
 for kind in ['dense','eval','router']:
  p=S/'wandb'/('%s_%s.json'%(run,kind))
  if not p.exists():continue
  j=json.loads(p.read_text())
  for spec,points in zip(j['specs'],j['series']):
   metric=spec['keys'][-1]
   ps=[{'step':x['_step'],'value':x[metric],'run':run} for x in points if lo<=x.get('_step',-1)<hi and metric in x]
   series.setdefault(metric,[]).extend(ps)
dedup=[]
for k,v in series.items():
 groups={}
 for row in sorted(v,key=lambda x:x['step']):
  key=(row['step'],row['run'])
  if key in groups:
   if abs(groups[key]['value']-row['value'])>1e-10:dedup.append({'metric':k,'step':row['step'],'earlier':groups[key]['value'],'later':row['value']})
  groups[key]=row
 series[k]=list(groups.values())
for p in (S/'wandb').glob('window_*.json'):
 j=json.loads(p.read_text());r={'window':p.stem,'run':j['run'],'range':j['requested_range'],'metrics':{}}
 for spec,ps in zip(j['specs'],j['series']):
  ps=[x for x in ps if j['requested_range'][0]<=x['_step']<j['requested_range'][1]]
  k=spec['keys'][-1];vs=[x[k] for x in ps if k in x]
  r['metrics'][k]={'n':len(vs),'mean':statistics.mean(vs) if vs else None,'median':statistics.median(vs) if vs else None}
 measurements.append(r)
def pair(a,b,end):
 ja=json.loads((S/'wandb'/('window_'+a+'.json')).read_text());jb=json.loads((S/'wandb'/('window_'+b+'.json')).read_text());out={}
 for spa,pa,spb,pb in zip(ja['specs'],ja['series'],jb['specs'],jb['series']):
  k=spa['keys'][-1];ma={x['_step']:x[k] for x in pa if k in x};mb={x['_step']:x[k] for x in pb if k in x};ix=sorted(x for x in ma.keys()&mb.keys() if x<end);ds=[mb[x]-ma[x] for x in ix]
  out[k]={'n':len(ix),'mean_new_minus_old':statistics.mean(ds),'max_absolute_delta':max(abs(x) for x in ds),'ratio_medians':statistics.median(mb[x] for x in ix)/statistics.median(ma[x] for x in ix)} if ix else {'n':0}
 return out
pairs={'ep':pair('ep_control','ep_new',81916),'kernels':pair('kernel_control','kernel_new',146339)}
original_maxdiff=max(abs(raw['phase0_weights'][k]-weights[0][k]) for k in avail)
summary={'retrieved_date_local':'2026-10-04','total_store_tokens':total,'tokens_per_step':tpb,'planned_steps':steps,'phase_boundaries_steps':bounds,'phase_budgets_tokens':budgets,'planned_tokens':sum(budgets),'max_planned_epochs':max(c['planned_epochs'] for c in cells),'max_initial_weight_diff_vs_pinned_spec':original_maxdiff,'current_snapshot':actual,'paired_windows':pairs,'windows':measurements,'duplicate_value_conflicts':dedup}
for name,rows in [('domain_weights',domains),('quality_weights',quality),('cell_weights',cells)]:
 save(name+'.json',rows)
 with (A/(name+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
save('summary.json',summary);save('series.json',series);save('sample_texts.json',examples)
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('QUALITY',quality)
print('BIGGEST INCREASE',sorted(domains,key=lambda x:-x['delta_main_pp'])[:5]);print('BIGGEST DECREASE',sorted(domains,key=lambda x:x['delta_main_pp'])[:5]);print('TOP EXPOSURE',sorted(cells,key=lambda x:-x['planned_epochs'])[:5])
