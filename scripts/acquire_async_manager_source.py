"""Acquire uploaded JAX serialization dependency and verify both public code manifests."""
import pathlib,json,hashlib,base64,datetime,urllib.parse,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/async_manager_2026_10_08';D.mkdir(exist_ok=True)
key='.venv/lib/python3.12/site-packages/jax/experimental/array_serialization/serialization.py'
manifests={role:json.loads((R/('sources/run_code_2026_10_08/'+role+'_code_manifest.json')).read_text())['contents'][key] for role in ['producer','continuation']}
e=manifests['continuation'];url='https://api.wandb.ai/artifactsV2/default/marin-community/marin_moe/source-marin_moe-_callable_runner.py/'+urllib.parse.quote(e['birthArtifactID'],safe='')+'/'+base64.b64decode(e['digest']).hex()+'/serialization.py'
p=D/'serialization.py';mf=R/'sources/source_manifest.json';records=json.loads(mf.read_text());rel=str(p.relative_to(R/'sources'))
if p.exists():
 prior=next(x for x in records if x['file']==rel)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=prior['sha256']:raise RuntimeError('existing source mismatch')
else:
 with urllib.request.urlopen(url,timeout=45) as response:b=response.read()
 for entry in manifests.values():
  if len(b)!=entry['size'] or base64.b64encode(hashlib.md5(b).digest()).decode()!=entry['digest']:raise RuntimeError('manifest digest mismatch')
 p.write_bytes(b);records.append({'file':rel,'url':url,'artifact_path':key,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});mf.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
(R/'analysis/async_manager_acquisition_v129.json').write_text(json.dumps({'artifact_path':key,'matches_both_run_manifests':all(x['size']==p.stat().st_size and x['digest']==base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode() for x in manifests.values()),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'records':[x for x in records if x['file']==rel]},indent=2)+'\n')
print('Acquired historical JAX serialization:',p.stat().st_size,'bytes; both manifests bound')
