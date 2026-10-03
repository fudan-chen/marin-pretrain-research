"""Acquire additional public, pinned evidence without overwriting the v1 sources."""
import urllib.request,json,pathlib,datetime,hashlib,concurrent.futures
ROOT=pathlib.Path(__file__).resolve().parents[1];S=ROOT/'sources';D=S/'deepening_2026_10_04';D.mkdir(exist_ok=True)
HFREV='75c25718e2a7cf7a3b1b498b00479351f80345e0'
HFPATH='registry/v1/swarms/h100-d512-from10pct-store-4d2e363d'
MARINREV='621bf9738b3ed9536aafc34b77cf8147740f1c17'
PRODUCTION='ad754c6d67555d6369a4a312ff7388e43cfd08c1'
def get(name,url):
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Marin-Chinese-Research/2.0'})
        with urllib.request.urlopen(req,timeout=50) as r:b=r.read();final=r.url
        (D/name).write_bytes(b)
        return {'file':str((D/name).relative_to(S)),'url':url,'final_url':final,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    except Exception as e:return {'file':str((D/name).relative_to(S)),'url':url,'error':str(e)}
def batch(urls):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:rs=list(ex.map(lambda x:get(*x),urls))
    manifest=json.loads((S/'source_manifest.json').read_text())+rs;(S/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    for r in rs:print(r['file'],r.get('bytes',r.get('error')),flush=True)
    return rs
if __name__=='__main__':
    batch([('marin_head.json','https://api.github.com/repos/marin-community/marin/commits/'+MARINREV),('hf_swarm_tree.json','https://huggingface.co/api/datasets/marin-community/grug-moe-mix-swarm/tree/'+HFREV+'/'+HFPATH+'?recursive=true&limit=1000'),('swarm_analyze.py','https://storage.googleapis.com/marin-public/held/h100-mix25-paloma/swarm-2026.09.14.1/analyze.py'),('regmix_paper.html','https://arxiv.org/html/2407.01492v2')])
    head=json.loads((D/'marin_head.json').read_text())['sha']
    batch([('marin_tree.json','https://api.github.com/repos/marin-community/marin/git/trees/'+head+'?recursive=1')])
    print('Pinned Marin revision:',head,flush=True)
    batch([('hf_'+name+'.parquet','https://huggingface.co/datasets/marin-community/grug-moe-mix-swarm/resolve/'+HFREV+'/'+HFPATH+'/'+name+'.parquet') for name in ['buckets','content','migration_rejections','observations','swarm']])
    batch([('hf_registry_tree.json','https://huggingface.co/api/datasets/marin-community/grug-moe-mix-swarm/tree/'+HFREV+'/registry/v1?recursive=true&limit=1000')])
    paths=[('mixture.py',MARINREV,'lib/levanter/src/levanter/data/mixture.py'),('datasets.py',MARINREV,'lib/levanter/src/levanter/data/text/datasets.py'),('test_varying_mixture.py',MARINREV,'lib/levanter/tests/test_varying_mixture.py'),('hero-mixture-log.md',MARINREV,'docs/reports/hero-mixture-log.md'),('mixture_production.py',PRODUCTION,'lib/levanter/src/levanter/data/mixture.py'),('datasets_production.py',PRODUCTION,'lib/levanter/src/levanter/data/text/datasets.py'),('dataset_production.py',PRODUCTION,'lib/levanter/src/levanter/data/dataset.py'),('schedule_production.py',PRODUCTION,'lib/levanter/src/levanter/schedule.py'),('launch_h100_scaling_ladder.py',MARINREV,'experiments/grug/moe_hero_ep/launch_h100_scaling_ladder.py'),('python_version_production.txt',PRODUCTION,'.python-version'),('pyproject_production.toml',PRODUCTION,'pyproject.toml')]
    batch([(name,'https://raw.githubusercontent.com/marin-community/marin/'+rev+'/'+path) for name,rev,path in paths])
