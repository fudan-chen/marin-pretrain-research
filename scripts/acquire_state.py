"""Archive callback, state and export tests at the pinned implementation revision."""
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import sys
import urllib.request

R=pathlib.Path(__file__).resolve().parents[1]
S=R/'sources';D=S/'state_2026_10_05'
SHA=json.loads((S/'decision_2026_10_04/marin_tree.json').read_text())['sha']
PATHS={
 'state_adapter.py':'lib/levanter/src/levanter/callbacks/state_adapter.py',
 'callback_core.py':'lib/levanter/src/levanter/callbacks/_core.py',
 'test_state_adapter.py':'lib/levanter/tests/test_state_adapter_callbacks.py',
 'test_checkpointing.py':'tests/test_grug_checkpointing.py',
 'test_hero.py':'tests/test_moe_hero_ep.py',
 'export_vllm.py':'experiments/grug/moe_hero_ep/ops/export_vllm.py',
 'weights.py':'experiments/grug/moe_hero_ep/weights.py',
}
def fetch(item):
 name,path=item;url='https://raw.githubusercontent.com/marin-community/marin/'+SHA+'/'+path
 with urllib.request.urlopen(url,timeout=45) as response:body=response.read()
 compile(body,name,'exec')
 return name,body,{'file':str((D/name).relative_to(S)),'url':url,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if __name__=='__main__':
 if sys.version_info<(3,12):raise SystemExit('Acquire with Python 3.12+ to syntax-check the pinned export source; build/probes can use the report environment.')
 if D.exists():raise SystemExit('Refusing overwrite')
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,PATHS.items()))
 D.mkdir()
 for name,body,_ in rows:(D/name).write_bytes(body)
 p=S/'source_manifest.json';m=json.loads(p.read_text());m.extend(x[2] for x in rows);p.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
 print('Archived',len(rows),'state-related primary files')
