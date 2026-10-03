exec(open(__file__.replace('acquire_data_evidence.py','acquire.py')).read().split('urls=')[0])
urls=[]
root='https://storage.googleapis.com/marin-public/held/h100-mix25-paloma/'
for path in ['final-2026.09.15.1/final_results.csv','final-2026.09.15.1/point_speedups.csv','swarm-2026.09.14.1/comparison.csv','swarm-2026.09.14.1/selected_runs.json','mixture-phases-2026.09.13.1/index.html']:
 urls.append(('mix_study_'+path.replace('/','_'),root+path))
for n in [9162,9183,9179]:
 urls.append(('issue_%d.json'%n,'https://api.github.com/repos/marin-community/marin/issues/%d'%n))
urls.append(('best_mixture_996f489106c7b922.json','https://raw.githubusercontent.com/marin-community/marin/main/experiments/grug/moe_hero_ep/best_mixture_996f489106c7b922.json'))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:m=list(ex.map(lambda x:get(*x),urls))
manifest=json.loads((S/'source_manifest.json').read_text())+m
(S/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
for v in m:print(v['file'],v.get('bytes',v.get('error')))
