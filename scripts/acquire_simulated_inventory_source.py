"""Acquire historical shuffle/slice and PRP source bytes referenced by public run artifact."""
import pathlib,json,hashlib,base64,datetime,urllib.parse,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/simulated_inventory_2026_10_08';D.mkdir(exist_ok=True);M=R/'sources/source_manifest.json';records=json.loads(M.read_text());added=[];m=json.loads((R/'sources/run_code_2026_10_08/continuation_code_manifest.json').read_text())['contents']
for name in ['dataset.py','_prp.py']:
 key='lib/levanter/src/levanter/data/'+name;e=m[key];p=D/name;rel=str(p.relative_to(R/'sources'));url='https://api.wandb.ai/artifactsV2/default/marin-community/marin_moe/source-marin_moe-_callable_runner.py/'+urllib.parse.quote(e['birthArtifactID'],safe='')+'/'+base64.b64decode(e['digest']).hex()+'/'+name
 if p.exists():assert next(x['sha256'] for x in records if x['file']==rel)==hashlib.sha256(p.read_bytes()).hexdigest()
 else:
  with urllib.request.urlopen(url,timeout=45) as response:b=response.read()
  p.write_bytes(b);rec={'file':rel,'url':url,'artifact_path':key,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};added.append(rec);M.write_text(json.dumps(records+added,ensure_ascii=False,indent=2)+'\n');print(name,len(b),flush=True)
 assert len(p.read_bytes())==e['size'] and base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode()==e['digest']
(R/'analysis/simulated_inventory_acquisition_v124.json').write_text(json.dumps({'records':[x for x in records+added if x['file'].startswith('simulated_inventory_2026_10_08/')]},indent=2)+'\n')
