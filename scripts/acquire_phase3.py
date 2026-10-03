exec(open(__file__.replace('acquire_phase3.py','acquire.py')).read().split('urls=')[0])
urls=[]
for n in [9126,8870,9062,9332,9333,8317,9689]:
 urls += [('issue_%d.json'%n,'https://api.github.com/repos/marin-community/marin/issues/%d'%n),('issue_%d_comments.json'%n,'https://api.github.com/repos/marin-community/marin/issues/%d/comments?per_page=100'%n)]
for ref in ['12d8b6f09f96ad4f0277445765c1a00dbc81d5be','main']:
 p='experiments/grug/moe_hero_ep/harrier_mix_2026_08_18.json'
 urls.append(('harrier_spec_'+ref[:8]+'.json','https://raw.githubusercontent.com/marin-community/marin/'+ref+'/'+p))
urls.append(('launch_datakit_moe_mix.py','https://raw.githubusercontent.com/marin-community/marin/12d8b6f09f96ad4f0277445765c1a00dbc81d5be/experiments/grug/moe/launch_datakit_moe_mix.py'))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:m=list(ex.map(lambda x:get(*x),urls))
manifest=json.loads((S/'source_manifest.json').read_text())+m
(S/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
for v in m:print(v['file'],v.get('bytes',v.get('error')))
