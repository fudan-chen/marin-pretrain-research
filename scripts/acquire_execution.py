"""Explicit, bounded public W&B provenance acquisition; no callable deserialization."""
import datetime
import hashlib
import json
import pathlib
import urllib.request
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources';D=S/'execution_2026_10_05'
NAMES=['rav-ladder-d768-v2','rav-ladder-d1024','rav-ladder-d1536','h100-mix25-20260912-d768','h100-mix25-20260912-d1024','h100-mix25-20260912-d1536']
URL='https://api.wandb.ai/graphql'
def query(fields):
    return 'query { project(name:"marin_moe",entityName:"marin-community") {'+' '.join('r%d:run(name:"%s"){%s}'%(i,n,fields) for i,n in enumerate(NAMES))+'} }'
def request(q):
    payload={'query':q};req=urllib.request.Request(URL,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','User-Agent':'Marin-Research/11'})
    with urllib.request.urlopen(req,timeout=35) as r:body=r.read()
    d=json.loads(body)
    if d.get('errors') or len(d.get('data',{}).get('project',{}))!=6:raise RuntimeError('Public run query incomplete')
    return body,d,{'url':URL,'request_body':payload}
if __name__=='__main__':
    if D.exists():raise SystemExit('Refusing to overwrite frozen execution snapshot')
    b,m,origin=request(query('name commit createdAt state fileCount files(names:["wandb-metadata.json"],first:10){edges{node{name url sizeBytes}} pageInfo{hasNextPage}}'))
    results=[('run_execution_files.json',b,origin)]
    b,d,origin=request(query('name commit fileCount files(first:100){edges{node{name sizeBytes}} pageInfo{hasNextPage}} runner:files(names:["code/_callable_runner.py"],first:10){edges{node{name url sizeBytes}}}'))
    runs=list(d['data']['project'].values())
    if any(r['files']['pageInfo']['hasNextPage'] for r in runs):raise RuntimeError('Inventory needs pagination; do not claim complete connection')
    results.append(('run_file_inventory.json',b,origin))
    # fileCount is retained independently. In this snapshot it exceeds connection length by one.
    for r in [runs[0],runs[3]]:
        edges=r['runner']['edges']
        if len(edges)!=1 or edges[0]['node']['sizeBytes']>=100000:raise RuntimeError('Expected one small public code entry')
        node=edges[0]['node']
        with urllib.request.urlopen(node['url'],timeout=35) as response:body=response.read(100001)
        if len(body)>=100000:raise RuntimeError('Runner read exceeds bounded source limit')
        compile(body,r['name'],'exec')  # Parse syntax only; never exec or deserialize.
        results.append(('runner_'+r['name']+'.py',body,{'url':node['url'],'public_run':r['name'],'wandb_file_name':node['name']}))
    D.mkdir();manifest=json.loads((S/'source_manifest.json').read_text())
    for name,body,provenance in results:
        p=D/name;p.write_bytes(body);manifest.append({'file':str(p.relative_to(S)),'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**provenance})
    (S/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Frozen public execution-source payloads:',len(results))
