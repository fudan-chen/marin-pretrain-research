"""Append a separately dated public W&B snapshot. Never replace older curves.
GitHub receipt compares bodies/comments; source payloads retain only W&B results.
"""
import pathlib,urllib.request,json,hashlib,datetime,concurrent.futures
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/live_2026_10_06';RUN='hero-fa4sm100-nomask-step146k';assert not D.exists()
def gql(query,variables):
 req=urllib.request.Request('https://api.wandb.ai/graphql',data=json.dumps({'query':query,'variables':variables}).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=45) as f:j=json.load(f)
 if j.get('errors'):raise RuntimeError(j['errors'])
 return j['data']['project']['run']
v={'e':'marin-community','p':'marin_moe','r':RUN}
meta=gql('query($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e){run(name:$r){name state heartbeatAt config summaryMetrics}}}',v)
for k in ['config','summaryMetrics']:meta[k]=json.loads(meta[k]) if isinstance(meta[k],str) else meta[k]
keys=sorted(k for k in meta['summaryMetrics'] if k.startswith('eval_dropless/paloma/') and k.split('/')[-1] in ('loss','bpb','macro_loss','micro_loss','macro_bpb'))
assert len(keys)==36
specs=[{'keys':['_step',k],'minStep':194999,'maxStep':212000,'samples':250} for k in keys]
def history(ss):
 h=gql('query($e:String!,$p:String!,$r:String!,$specs:[JSONString!]!){project(name:$p,entityName:$e){run(name:$r){sampledHistory(specs:$specs)}}}',dict(v,specs=[json.dumps(s) for s in ss]))['sampledHistory'];assert len(h)==len(ss)
 return {'run':RUN,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sampled':True,'specs':ss,'series':[[json.loads(x) if isinstance(x,str) else x for x in row] for row in h]}
densekeys=['train/cross_entropy_loss','train/loss','grad/norm/total','throughput/total_tokens','throughput/tokens_per_second','throughput/mfu','optim/learning_rate','moe/drop_fraction']
with concurrent.futures.ThreadPoolExecutor(2) as pool:
 a=pool.submit(history,specs);b=pool.submit(history,[{'keys':['_step',k],'minStep':199000,'maxStep':212000,'samples':500} for k in densekeys]);ev=a.result();dense=b.result()
objects={'meta.json':meta,'eval.json':ev,'dense.json':dense};rows=[];payload=[]
for name,obj in objects.items():
 raw=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode();payload.append((name,raw));rows.append({'file':'live_2026_10_06/'+name,'url':'https://api.wandb.ai/graphql','public_run_url':'https://wandb.ai/marin-community/marin_moe/runs/'+RUN,'request_scope':'anonymous GraphQL metadata or sampledHistory; specifications embedded in payload','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
D.mkdir()
for name,raw in payload:(D/name).write_bytes(raw)
p=R/'sources/source_manifest.json';manifest=json.loads(p.read_text());manifest.extend(rows);p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(R/'analysis/live_acquisition_2026_10_06.json').write_text(json.dumps({'files':rows,'old_snapshots_preserved':True},indent=2)+'\n')
print('Snapshot:',meta['summaryMetrics']['_step'],meta['summaryMetrics']['throughput/total_tokens'],'eval points:',[len(x) for x in ev['series'][:2]])
