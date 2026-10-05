"""Acquire one pinned optimizer source, preserving prior archive; Python 3.12+."""
import datetime,hashlib,json,pathlib,sys,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/optimizer_2026_10_05';SHA='84869ae8c91ffe64e9f761c5bd714542eb1876e0'
if sys.version_info<(3,12):raise SystemExit('Use Python 3.12+')
if D.exists():raise SystemExit('Refusing overwrite')
u=f'https://raw.githubusercontent.com/marin-community/marin/{SHA}/experiments/grug/moe_hero_ep/optimizer.py';b=urllib.request.urlopen(u,timeout=45).read();compile(b,'optimizer.py','exec');D.mkdir();(D/'optimizer.py').write_bytes(b)
row={'file':'optimizer_2026_10_05/optimizer.py','url':u,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};p=R/'sources/source_manifest.json';j=json.loads(p.read_text());j.append(row);p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');(R/'analysis/optimizer_acquisition.json').write_text(json.dumps({'revision':SHA,'files':[row]},ensure_ascii=False,indent=2)+'\n')
