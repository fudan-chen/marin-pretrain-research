"""Acquire pinned experimental fork code from C025; not production code; Python3.12+."""
import concurrent.futures,datetime,hashlib,json,pathlib,sys,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];D=R/'sources/router_precision_2026_10_05';SHA='a00cb77a491f4777a2c66edce54f47ff7b255c40';FILES=['model.py','router_shape_probe.py','launch_diagnostics.py']
def fetch(n):
 u=f'https://raw.githubusercontent.com/yonromai/marin/{SHA}/experiments/grug/moe_hero_ep/{n}';b=urllib.request.urlopen(u,timeout=45).read();compile(b,n,'exec');return n,b,{'file':'router_precision_2026_10_05/'+n,'url':u,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if sys.version_info<(3,12):raise SystemExit('Use Python3.12+')
if D.exists():raise SystemExit('Refusing overwrite')
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(fetch,FILES))
D.mkdir()
for n,b,_ in rows:(D/n).write_bytes(b)
p=R/'sources/source_manifest.json';j=json.loads(p.read_text());j.extend(r[2] for r in rows);p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');(R/'analysis/router_precision_acquisition.json').write_text(json.dumps({'repository':'yonromai/marin','revision':SHA,'deployment_status':'experimental_fork_not_deployed_per_C025','files':[r[2] for r in rows]},ensure_ascii=False,indent=2)+'\n')
