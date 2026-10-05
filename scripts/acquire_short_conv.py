"""Acquire pinned ShortConv implementation and original tests; no overwrite; Python3.12+."""
import concurrent.futures,datetime,hashlib,json,pathlib,sys,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/short_conv_2026_10_05';SHA='84869ae8c91ffe64e9f761c5bd714542eb1876e0'
FILES={n:'lib/levanter/src/levanter/kernels/pallas/short_conv/'+n for n in ['__init__.py','api.py','config.py','pallas_gpu.py','reference.py']};FILES['test_short_conv.py']='lib/levanter/tests/kernels/test_short_conv.py'
def fetch(item):
 n,p=item;u=f'https://raw.githubusercontent.com/marin-community/marin/{SHA}/{p}';b=urllib.request.urlopen(u,timeout=45).read();compile(b,n,'exec');return n,b,{'file':'short_conv_2026_10_05/'+n,'url':u,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if sys.version_info<(3,12):raise SystemExit('Use Python3.12+')
if D.exists():raise SystemExit('Refusing overwrite')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,FILES.items()))
D.mkdir()
for n,b,_ in rows:(D/n).write_bytes(b)
p=R/'sources/source_manifest.json';j=json.loads(p.read_text());j.extend(r[2] for r in rows);p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');(R/'analysis/short_conv_acquisition.json').write_text(json.dumps({'revision':SHA,'files':[r[2] for r in rows]},ensure_ascii=False,indent=2)+'\n')
