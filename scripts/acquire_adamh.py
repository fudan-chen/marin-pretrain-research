"""Append pinned AdamH source; refuse overwrite. Use Python 3.12 HTTPS."""
import pathlib,json,hashlib,datetime,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/adamh_2026_10_05';SHA='84869ae8c91ffe64e9f761c5bd714542eb1876e0'
assert not D.exists(),'Refusing archive overwrite'
u=f'https://raw.githubusercontent.com/marin-community/marin/{SHA}/experiments/grug/moe_hero_ep/adamh.py'
with urllib.request.urlopen(u,timeout=45) as response:b=response.read()
compile(b,'adamh.py','exec');D.mkdir();(D/'adamh.py').write_bytes(b)
r={'file':'adamh_2026_10_05/adamh.py','url':u,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};p=R/'sources/source_manifest.json';j=json.loads(p.read_text());j.append(r);p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');(R/'analysis/adamh_acquisition.json').write_text(json.dumps({'revision':SHA,'files':[r]},indent=2)+'\n')
