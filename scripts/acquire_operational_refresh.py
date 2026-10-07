"""Public issue refresh and sampled W&B operational history, saved separately."""
import concurrent.futures,datetime,hashlib,json,pathlib,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources';D=S/'operational_2026_10_07';assert not D.exists();BASE='https://api.github.com/repos/marin-community/marin/'
prior=json.loads((S/'archive_manifest.json').read_text());rows=[];payload={}
def get(name,url):
 req=urllib.request.Request(url,headers={'User-Agent':'marin-public-research','Accept':'application/vnd.github+json'})
 with urllib.request.urlopen(req,timeout=35) as f:b=f.read();link=f.headers.get('Link','');status=f.status
 assert 'rel="next"' not in link,('Need additional page',url)
 return name,b,json.loads(b),{'file':D.name+'/'+name,'url':url,'http_status':status,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pagination_next':False}
urls={f'issue_{n}.json':BASE+f'issues/{n}' for n in [8435,8506,8870]};urls.update({f'comments_{n}.json':BASE+f'issues/{n}/comments?per_page=100' for n in [8435,8506,8870]})
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as p:results=list(p.map(lambda item:get(*item),urls.items()))
for name,b,j,rec in results:payload[name]=(b,j);rows.append(rec)
for n in [8435,8506,8870]:assert payload[f'issue_{n}.json'][1]['comments']==len(payload[f'comments_{n}.json'][1])
RUN='hero-fa4sm100-nomask-step146k';v={'e':'marin-community','p':'marin_moe','r':RUN}
def gql(name,query,variables):
 req=urllib.request.Request('https://api.wandb.ai/graphql',data=json.dumps({'query':query,'variables':variables}).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=45) as f:b=f.read();status=f.status
 j=json.loads(b);assert not j.get('errors'),j
 rows.append({'file':D.name+'/'+name,'url':'https://api.wandb.ai/graphql','public_run_url':'https://wandb.ai/marin-community/marin_moe/runs/'+RUN,'http_status':status,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'query':query,'variables':variables})
 payload[name]=(b,j);return j['data']['project']['run']
meta=gql('wandb_meta.json','query($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e){run(name:$r){name state heartbeatAt config summaryMetrics}}}',v);summary=json.loads(meta['summaryMetrics']) if isinstance(meta['summaryMetrics'],str) else meta['summaryMetrics']
spec={'keys':['_step','_timestamp','throughput/total_tokens','throughput/duration','throughput/iteration_time'],'minStep':146000,'maxStep':int(summary['_step'])+1,'samples':1000}
gql('wandb_operational_history.json','query($e:String!,$p:String!,$r:String!,$specs:[JSONString!]!){project(name:$p,entityName:$e){run(name:$r){sampledHistory(specs:$specs)}}}',dict(v,specs=[json.dumps(spec)]))
comparisons=[]
for n in [8435,8506,8870]:
 oldpath=S/'engineering_v77_2026_10_07'/f'comments_{n}.json'
 if not oldpath.exists():oldpath=S/'engineering_v77_2026_10_07'/f'issue_{n}_comments.json'
 old={x['id']:x for x in json.loads(oldpath.read_text())};new={x['id']:x for x in payload[f'comments_{n}.json'][1]}
 comparisons.append({'issue':n,'old_file':str(oldpath.relative_to(R)),'new_file':'sources/'+D.name+f'/comments_{n}.json','old_count':len(old),'new_count':len(new),'added_ids':sorted(new.keys()-old.keys()),'removed_ids':sorted(old.keys()-new.keys()),'changed_bodies':[i for i in old.keys()&new.keys() if old[i]['body']!=new[i]['body']]})
old=[x for x in prior['files'] if x['file']!='source_manifest.json'];assert all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in old)
D.mkdir()
for name,(b,j) in payload.items():(D/name).write_bytes(b)
p=S/'source_manifest.json';m=json.loads(p.read_text());m.extend(rows);p.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
(R/'analysis/operational_refresh_acquisition.json').write_text(json.dumps({'records':rows,'prior_integrity_manifest':prior,'prior_bytes_preserved':len(old),'comment_comparisons':comparisons,'history_spec':spec,'history_sampled':True,'actual_production_deployed_revision':None,'actual_initiating_hang_cause':None},indent=2)+'\n');print('Archived',len(rows),'payloads; comments',[(x['issue'],x['new_count'],x['added_ids']) for x in comparisons],'latest step',summary['_step'])
