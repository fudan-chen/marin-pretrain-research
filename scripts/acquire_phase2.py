exec(open(__file__.replace('acquire_phase2.py','acquire.py')).read().split('urls=')[0])
urls=[]
for n in [8506,8818]:
 urls += [('issue_%d.json'%n,'https://api.github.com/repos/marin-community/marin/issues/%d'%n),('issue_%d_comments.json'%n,'https://api.github.com/repos/marin-community/marin/issues/%d/comments?per_page=100'%n)]
for ref in ['12d8b6f09f96ad4f0277445765c1a00dbc81d5be','996f4891','main']:
 p='experiments/grug/moe_hero_ep/harrier_mix_2026_08_18.py'
 urls.append(('harrier_mix_'+ref[:8]+'.py','https://raw.githubusercontent.com/marin-community/marin/'+ref+'/'+p))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: m=list(ex.map(lambda x:get(*x),urls))
manifest=json.loads((S/'source_manifest.json').read_text())+m
(S/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
for v in m: print(v['file'],v.get('bytes',v.get('error')))
