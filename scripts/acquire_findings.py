"""Freeze four candidate configs and six resumed-step windows without refreshing old sources."""
import pathlib,json,urllib.request,datetime,hashlib,concurrent.futures
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources';D=S/'findings_2026_10_04';D.mkdir(exist_ok=True)
QUERY='query($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e){run(name:$r){name displayName state config summaryMetrics}}}'
HISTORY='query($e:String!,$p:String!,$r:String!,$specs:[JSONString!]!){project(name:$p,entityName:$e){run(name:$r){sampledHistory(specs:$specs)}}}'
def fetch(item):
    name,payload=item;path=D/name
    if path.exists():return None
    url='https://api.wandb.ai/graphql'
    try:
        req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','User-Agent':'Marin-Chinese-Research/4.0'})
        with urllib.request.urlopen(req,timeout=45) as res:body=res.read()
        path.write_bytes(body)
        return {'file':str(path.relative_to(S)),'url':url,'request_body':payload,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    except Exception as e:return {'file':str(path.relative_to(S)),'url':url,'request_body':payload,'error':str(e)}
if __name__=='__main__':
    selected=json.loads((S/'mix_study_swarm-2026.09.14.1_selected_runs.json').read_text())['new'];names=[r['run_name'] for r in selected]+['h100-d512-mixprior-197c9f5ceff6b9ee-seed0-from10pct-20260829']
    items=[]
    for name in names:items.append(('config_'+name+'.json',{'query':QUERY,'variables':{'e':'marin-community','p':'marin_moe','r':name}}))
    for name in [r['run_name'] for r in selected]+['h100-d512-mix-proportional-seed%d-from10pct-20260826-rno2a'%i for i in range(3)]:
        specs=[{'keys':['_step',k],'samples':100,'minStep':380,'maxStep':430} for k in ['train/cross_entropy_loss','throughput/total_tokens']]
        items.append(('window_'+name+'.json',{'query':HISTORY,'variables':{'e':'marin-community','p':'marin_moe','r':name,'specs':[json.dumps(x) for x in specs]}}))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:results=[x for x in ex.map(fetch,items) if x is not None]
    manifest=json.loads((S/'source_manifest.json').read_text())+results;(S/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    for r in results:print(r['file'],r.get('bytes',r.get('error')))
