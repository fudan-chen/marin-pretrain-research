"""Append five pinned checkpoint implementation sources, refusing overwrite."""
import concurrent.futures,datetime,hashlib,json,pathlib,sys,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/checkpoint_commit_2026_10_05';SHA='84869ae8c91ffe64e9f761c5bd714542eb1876e0'
if sys.version_info<(3,12):raise SystemExit('Use Python 3.12+')
if D.exists():raise SystemExit('Refusing overwrite')
paths=['lib/levanter/src/levanter/checkpoint.py','lib/levanter/src/levanter/checkpoint_manifest.py','lib/levanter/src/levanter/tensorstore_serialization.py','lib/rigging/src/rigging/filesystem/atomic.py','lib/rigging/src/rigging/filesystem/storage_path.py']
def fetch(path):
 url=f'https://raw.githubusercontent.com/marin-community/marin/{SHA}/{path}'
 with urllib.request.urlopen(url,timeout=45) as response:body=response.read()
 compile(body,path,'exec');return path,url,body
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(fetch,paths))
D.mkdir();rows=[]
for path,url,body in results:
 name=pathlib.Path(path).name;(D/name).write_bytes(body);rows.append({'file':'checkpoint_commit_2026_10_05/'+name,'url':url,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
p=R/'sources/source_manifest.json';manifest=json.loads(p.read_text());manifest.extend(rows);p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');(R/'analysis/checkpoint_commit_acquisition.json').write_text(json.dumps({'revision':SHA,'files':rows},ensure_ascii=False,indent=2)+'\n')
print('Archived pinned checkpoint sources:',len(rows))
