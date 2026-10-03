exec(open(__file__.replace('acquire_more.py','acquire.py')).read().split('urls=')[0])
base='https://raw.githubusercontent.com/marin-community/marin/'
commit='12d8b6f09f96ad4f0277445765c1a00dbc81d5be'
paths=['experiments/grug/moe_hero_ep/train.py','experiments/grug/moe_hero_ep/model.py','experiments/grug/moe_hero_ep/optimizer.py','experiments/grug/moe_hero_ep/launch_scaling_ladder.py','experiments/grug/moe_hero_ep/launch_datakit_moe_mix.py']
urls=[('code_'+p.split('/')[-1],base+commit+'/'+p) for p in paths]
urls += [('plot_scaling_ladder.py',base+'d23e6e9c3673435fb82d83aa6c51a607d0da6009/experiments/grug/moe_hero_ep/plot_scaling_ladder.py'),('ep_writeup.html','https://storage.googleapis.com/marin-public/rav/moe-fixed-wave-a2a-384/2026.08.17/index.html')]
for n in [8443,8480,9352,9615,8827,7856,8003,6442,5203,6739,7208,6443]:
 urls += [('issue_%d.json'%n,'https://api.github.com/repos/marin-community/marin/issues/%d'%n),('issue_%d_comments.json'%n,'https://api.github.com/repos/marin-community/marin/issues/%d/comments?per_page=100'%n)]
for p in ['README.md','docs/learning/06-reading-training-signals.md','docs/learning/04-worked-example-step-32999.md','reports/generated/hero/2026-09-03-pm.md','notes/daily/2026-10-02.md','scripts/marin_wandb.py','scripts/pull_data.py','scripts/marin_tracker/hero.py','data/hero/hero-12d8b6f0-dee637/meta.json','data/hero/hero-12d8b6f0-dee637/mixture.jsonl','data/hero/hero-12d8b6f0-dee637/eval.jsonl','data/hero/hero-12d8b6f0-dee637/dense.csv']:
 urls.append(('prior_'+p.replace('/','_'),'https://raw.githubusercontent.com/fudan-chen/pretrain/'+json.loads((S/'prior_tree.json').read_text())['sha']+'/'+p))
urls.append(('hero_related_search.json','https://api.github.com/search/issues?q=repo%3Amarin-community%2Fmarin+hero+gate+router&per_page=20'))
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
 m=list(ex.map(lambda x:get(*x),urls))
manifest=json.loads((S/'source_manifest.json').read_text())+m
(S/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
for v in m: print(v['file'],v.get('bytes',v.get('error')))
