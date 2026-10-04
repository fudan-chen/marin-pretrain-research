"""Archive primary method papers and attempt to recover the order experiment config."""
import urllib.request,json,pathlib,datetime,hashlib,concurrent.futures,sys
R=pathlib.Path(__file__).resolve().parents[1];S=R/'sources';D=S/'practical_2026_10_04';D.mkdir(exist_ok=True)
def fetch(item):
    name,url,payload=item
    try:
        request=urllib.request.Request(url,data=json.dumps(payload).encode() if payload else None,headers={'User-Agent':'Marin-Chinese-Research/3.0','Content-Type':'application/json'} if payload else {'User-Agent':'Marin-Chinese-Research/3.0'})
        with urllib.request.urlopen(request,timeout=45) as res:body=res.read();final=res.url
        (D/name).write_bytes(body)
        return {'file':str((D/name).relative_to(S)),'url':url,'final_url':final,'request_body':payload,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    except Exception as e:return {'file':str((D/name).relative_to(S)),'url':url,'request_body':payload,'error':str(e)}
if __name__=='__main__':
    items=[(name,'https://arxiv.org/'+path,None) for name,path in [('doge.html','html/2310.15393v2'),('doremi.html','html/2305.10429v4'),('data_constrained.html','abs/2305.16264'),('data_mixing_laws.html','html/2403.16952v2'),('online_mixing.html','html/2312.02406v2')]]
    query='query($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e){run(name:$r){name displayName state config summaryMetrics}}}'
    for short in ['r-stem-pre-high','r-stem-cool-high']:
        name='h100-d512-mix-'+short+'-seed0-from10pct-20260826-rno2a'
        items.append((short+'_wandb_config_attempt.json','https://api.wandb.ai/graphql',{'query':query,'variables':{'e':'marin-community','p':'marin_moe','r':name}}))
    if '--configs' in sys.argv:
        import pyarrow.parquet as pq
        selected=pq.read_table(S/'deepening_2026_10_04/hf_observations.parquet').to_pylist()
        items=[]
        for r in selected:
            if r['group'] not in ['semantic_domain_ablation','proportional_domain_ablation','semantic_quality_ablation','proportional_quality_isolation','core_design','proportional_baseline']:continue
            name=r['run_name'];file='config_'+name+'.json'
            if (D/file).exists():continue
            items.append((file,'https://api.wandb.ai/graphql',{'query':query,'variables':{'e':'marin-community','p':'marin_moe','r':name}}))
    if '--history' in sys.argv:
        items=[]
        query='query($e:String!,$p:String!,$r:String!,$specs:[JSONString!]!){project(name:$p,entityName:$e){run(name:$r){sampledHistory(specs:$specs)}}}'
        for short in ['r-stem-pre-high','r-stem-cool-high']:
            name='h100-d512-mix-'+short+'-seed0-from10pct-20260826-rno2a'
            specs=[{'keys':['_step',k],'samples':100,'minStep':a,'maxStep':b} for a,b in [(380,430),(3100,3140)] for k in ['train/cross_entropy_loss','throughput/total_tokens']]
            items.append((short+'_step_windows.json','https://api.wandb.ai/graphql',{'query':query,'variables':{'e':'marin-community','p':'marin_moe','r':name,'specs':[json.dumps(x) for x in specs]}}))
        items.append(('trainer_initial.py','https://raw.githubusercontent.com/marin-community/marin/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/lib/levanter/src/levanter/trainer.py',None))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:results=list(ex.map(fetch,items))
    manifest=json.loads((S/'source_manifest.json').read_text())+results
    (S/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    for item in results:print(item['file'],item.get('bytes',item.get('error')))
