import urllib.request, json, pathlib, datetime, hashlib, concurrent.futures
ROOT=pathlib.Path(__file__).resolve().parents[1]; S=ROOT/'sources'
headers={'User-Agent':'Marin-Chinese-Research/1.0','Accept':'application/vnd.github+json'}
def get(name,url):
    req=urllib.request.Request(url,headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=45) as r: data=r.read(); final=r.url
        (S/name).write_bytes(data)
        return {'file':name,'url':url,'final_url':final,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    except Exception as e: return {'file':name,'url':url,'error':str(e)}
urls=[('issue.json','https://api.github.com/repos/marin-community/marin/issues/8435'),('comments_p1.json','https://api.github.com/repos/marin-community/marin/issues/8435/comments?per_page=100&page=1'),('data_overview.html','https://storage.googleapis.com/marin-public/held/harrier-k40-cluster-overview/2026.08.18/index.html?revision=uniform-sampling'),('prior_repo.json','https://api.github.com/repos/fudan-chen/pretrain'),('prior_tree.json','https://api.github.com/repos/fudan-chen/pretrain/git/trees/main?recursive=1'),('marin_repo.json','https://api.github.com/repos/marin-community/marin'),('wandb_report.html','https://wandb.ai/marin-community/marin_moe/reports/535B-A23B-18T-Token-Hero-Run-Scaling-Ladder--VmlldzoxNzc2MDM5Ng')]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: manifest=list(ex.map(lambda x:get(*x),urls))
issue=json.loads((S/'issue.json').read_text()); pages=(issue['comments']+99)//100
for p in range(2,pages+1): manifest.append(get('comments_p%d.json'%p,'https://api.github.com/repos/marin-community/marin/issues/8435/comments?per_page=100&page=%d'%p))
comments=[]
for p in range(1,pages+1): comments+=json.loads((S/('comments_p%d.json'%p)).read_text())
(S/'comments.json').write_text(json.dumps(comments,ensure_ascii=False,indent=2))
(S/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps(manifest,ensure_ascii=False,indent=2))
print('TITLE',issue['title'],'COUNT',issue['comments'],len(comments),'UPDATED',issue['updated_at'])
with (S/'issue_thread.md').open('w') as f:
    f.write('# Issue 主帖\nURL: '+issue['html_url']+'\n'+issue['body']+'\n')
    for i,c in enumerate(comments,1): f.write('\n\n# C%03d %s %s\nURL: %s\n%s\n'%(i,c['created_at'],c['user']['login'],c['html_url'],c['body']))
