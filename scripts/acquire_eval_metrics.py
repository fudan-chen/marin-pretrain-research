"""Acquire pinned RunningMean implementation; append only, Python 3.12+."""
import datetime,hashlib,json,pathlib,sys,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/eval_metrics_2026_10_05';SHA='84869ae8c91ffe64e9f761c5bd714542eb1876e0'
if sys.version_info<(3,12):raise SystemExit('Use Python 3.12+')
if D.exists():raise SystemExit('Refusing overwrite')
u=f'https://raw.githubusercontent.com/marin-community/marin/{SHA}/lib/levanter/src/levanter/utils/stat_utils.py'
with urllib.request.urlopen(u,timeout=45) as response:b=response.read()
compile(b,'stat_utils.py','exec');D.mkdir();(D/'stat_utils.py').write_bytes(b)
row={'file':'eval_metrics_2026_10_05/stat_utils.py','url':u,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};p=R/'sources/source_manifest.json';j=json.loads(p.read_text());j.append(row);p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');(R/'analysis/eval_metric_acquisition.json').write_text(json.dumps({'revision':SHA,'files':[row]},ensure_ascii=False,indent=2)+'\n')
