"""Anonymous, read-only acquisition of two run inventories, code manifests, logs and five source files.
Existing snapshot bytes are verified on rerun; no pickle is downloaded or executed.
"""
import concurrent.futures,datetime,hashlib,json,pathlib,urllib.parse,urllib.request,base64
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/run_code_2026_10_08';D.mkdir(exist_ok=True)
M=R/'sources/source_manifest.json';records=json.loads(M.read_text());added=[]
query='query($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e){run(name:$r){name commit config files(first:100){edges{node{name sizeBytes url}} pageInfo{hasNextPage endCursor}}}}}'
runs={'producer':'h100-ladder-d512-ep8-bs1024-791tpp-10pct-20260826-rno2a','continuation':'h100-d512-mix-proportional-zero-c00-seed0-from10pct-20260826-rno2a'}
def acquire(name,url,payload=None):
 p=D/name;rel=str(p.relative_to(R/'sources'))
 if p.exists():
  rec=next(x for x in records if x['file']==rel);assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'];return p
 req=urllib.request.Request(url,data=json.dumps(payload).encode() if payload else None,headers={'Content-Type':'application/json','User-Agent':'Marin-research-code-provenance'})
 with urllib.request.urlopen(req,timeout=50) as response:b=response.read()
 p.write_bytes(b);rec={'file':rel,'url':url,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 if payload:rec['request']=payload
 added.append(rec);M.write_text(json.dumps(records+added,ensure_ascii=False,indent=2)+'\n');print(name,len(b),flush=True);return p
for role,run in runs.items():
 p=acquire(role+'_inventory.json','https://api.wandb.ai/graphql',{'query':query,'variables':{'e':'marin-community','p':'marin_moe','r':run}})
 raw=json.loads(p.read_text())['data']['project']['run'];assert raw['name']==run and not raw['files']['pageInfo']['hasNextPage']
 fs=[e['node'] for e in raw['files']['edges']]
 manifest=next(f for f in fs if f['name'].endswith('wandb_manifest.json') and f['sizeBytes']>1000000)
 acquire(role+'_code_manifest.json',manifest['url']);acquire(role+'_output.log.txt',next(f['url'] for f in fs if f['name']=='output.log'))
# Both manifests retain independent origins. Download only the five analysis-relevant files.
contents=json.loads((D/'continuation_code_manifest.json').read_text())['contents']
for name,path in {'mixture.py':'data/mixture.py','loader.py':'data/loader.py','datasets.py':'data/text/datasets.py','schedule.py':'schedule.py','train_lm.py':'main/train_lm.py'}.items():
 e=contents['lib/levanter/src/levanter/'+path];digest=base64.b64decode(e['digest']).hex()
 url='https://api.wandb.ai/artifactsV2/default/marin-community/marin_moe/source-marin_moe-_callable_runner.py/'+urllib.parse.quote(e['birthArtifactID'],safe='')+'/'+digest+'/'+name
 p=acquire(name,url);assert len(p.read_bytes())==e['size'] and base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode()==e['digest']
if added:M.write_text(json.dumps(records+added,ensure_ascii=False,indent=2)+'\n')
(R/'analysis/run_code_acquisition_v122.json').write_text(json.dumps({'records':[x for x in records+added if x['file'].startswith('run_code_2026_10_08/')]},indent=2)+'\n')
print('Source acquisition complete',len(added))
