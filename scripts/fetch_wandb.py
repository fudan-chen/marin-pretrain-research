import urllib.request,json,pathlib,concurrent.futures,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]; S=ROOT/'sources'; (S/'wandb').mkdir(exist_ok=True)
RUNS=['hero-12d8b6f0-dee637','hero-wd-gate-router-p02-step58k','hero-ragged_a2a-nccl2307-ep-step81k','hero-mix-996f4891-step108k','hero-fa4sm100-nomask-step146k','hero-nopdl-step108k','hero-main-step121638']
def gql(q,variables):
 req=urllib.request.Request('https://api.wandb.ai/graphql',data=json.dumps({'query':q,'variables':variables}).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=50) as r: j=json.load(r)
 if 'errors' in j: raise ValueError(j['errors'])
 return j['data']['project']['run']
def pull(run):
 v={'e':'marin-community','p':'marin_moe','r':run}
 meta=gql('query($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e){run(name:$r){name displayName state createdAt heartbeatAt config summaryMetrics historyKeys}}}',v)
 for k in ['config','summaryMetrics']: meta[k]=json.loads(meta[k]) if isinstance(meta[k],str) else meta[k]
 (S/'wandb'/('%s_meta.json'%run)).write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 hk=meta['historyKeys']; hk=hk.get('keys',hk); keys=list(hk.keys()) if isinstance(hk,dict) else hk
 print(run,'state',meta['state'],'summary step',meta['summaryMetrics'].get('_step'),'keys',len(keys),flush=True)
 (S/'wandb'/('%s_keys.json'%run)).write_text(json.dumps(keys,indent=2))
 groups={
 'dense':['_step','train/cross_entropy_loss','train/loss','grad/norm/total','throughput/tokens_per_second','throughput/mfu','optim/learning_rate','optim/adam_lr','throughput/total_tokens'],
 'eval':['_step']+[k for k in keys if k.startswith(('eval/paloma/','eval_dropless/paloma/')) and k.endswith(('loss','bpb'))],
 'mixture':['_step']+[k for k in keys if k.startswith('mixture/')],
 'router':['_step']+[k for k in keys if any(t in k for t in ['drop','routing_entropy']) and not any(t in k for t in ['/layer_','/layer/'])]
 }
 for label,ks in groups.items():
  ks=[k for k in ks if k in keys or k=='_step']
  if len(ks)<2:continue
  specs=[{'keys':['_step',k],'samples':1500} for k in ks if k!='_step']
  h=gql('query($e:String!,$p:String!,$r:String!,$specs:[JSONString!]!){project(name:$p,entityName:$e){run(name:$r){sampledHistory(specs:$specs)}}}',dict(v,specs=[json.dumps(x) for x in specs]))['sampledHistory']
  out={'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':run,'sampled':True,'specs':specs,'series':[[json.loads(x) if isinstance(x,str) else x for x in a] for a in h]}
  (S/'wandb'/('%s_%s.json'%(run,label))).write_text(json.dumps(out,ensure_ascii=False))
 return {'run':run,'keys':len(keys),'summary':meta['summaryMetrics']}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 for r in ex.map(pull,RUNS): print(r['run'],'DONE',flush=True)
