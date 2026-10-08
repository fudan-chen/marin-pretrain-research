"""Acquire selected historical entrypoint/key sources and declared dependencies from public artifacts."""
import base64,datetime,hashlib,json,pathlib,urllib.parse,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/historical_key_2026_10_08';D.mkdir(exist_ok=True);M=R/'sources/source_manifest.json';records=json.loads(M.read_text());added=[]
m=json.loads((R/'sources/run_code_2026_10_08/continuation_code_manifest.json').read_text())['contents']
paths={'train_base.py':'experiments/grug/base/train.py','train_moe.py':'experiments/grug/moe/train.py','train_ep.py':'experiments/grug/moe_hero_ep/train.py','checkpointing.py':'experiments/grug/checkpointing.py','jax_utils.py':'lib/levanter/src/levanter/utils/jax_utils.py','jax_version.py':'.venv/lib/python3.12/site-packages/jax/version.py'}
def acquire(name,url,entry=None):
 p=D/name;rel=str(p.relative_to(R/'sources'))
 if p.exists():assert next(x['sha256'] for x in records if x['file']==rel)==hashlib.sha256(p.read_bytes()).hexdigest()
 else:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Marin-historical-key-research'}),timeout=45) as response:b=response.read()
  p.write_bytes(b);rec={'file':rel,'url':url,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  if entry:rec['artifact_path']=entry
  added.append(rec);M.write_text(json.dumps(records+added,ensure_ascii=False,indent=2)+'\n');print(name,len(b),flush=True)
 return p
for name,key in paths.items():
 e=m[key];url='https://api.wandb.ai/artifactsV2/default/marin-community/marin_moe/source-marin_moe-_callable_runner.py/'+urllib.parse.quote(e['birthArtifactID'],safe='')+'/'+base64.b64decode(e['digest']).hex()+'/'+pathlib.PurePosixPath(key).name
 p=acquire(name,url,key);assert len(p.read_bytes())==e['size'] and base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode()==e['digest']
for role in ['producer','continuation']:
 run=json.loads((R/'sources/run_code_2026_10_08'/(role+'_inventory.json')).read_text())['data']['project']['run'];url=next(e['node']['url'] for e in run['files']['edges'] if e['node']['name']=='requirements.txt');acquire(role+'_requirements.txt',url)
(R/'analysis/historical_key_acquisition_v123.json').write_text(json.dumps({'records':[x for x in records+added if x['file'].startswith('historical_key_2026_10_08/')]},indent=2)+'\n');print('Historical key acquisition complete',len(added))
