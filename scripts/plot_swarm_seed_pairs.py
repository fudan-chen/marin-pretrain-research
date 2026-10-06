"""Plot source-derived seed-label endpoint differences; no confidence intervals."""
import json,pathlib,re
import matplotlib
matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='marin-swarm-pairs-v62'
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1];d=json.loads((R/'analysis/swarm_seed_pairs.json').read_text());by={r['metric']:r for r in d['endpoints']}
keys=['eval_dropless/paloma/macro_bpb','eval_dropless/paloma/dolma_100_programing_languages-llama3/bpb','eval_dropless/uncheatable_eval/macro_bpb','eval_dropless/paloma/manosphere_meta_sep-llama3/bpb','eval_dropless/uncheatable_eval/bbc_news-llama3/bpb'];labels=['Paloma macro','Paloma code','Uncheatable macro','Paloma manosphere','Uncheatable BBC news']
fig,ax=plt.subplots(figsize=(11,5.6));colors=['#487d9c','#7d9860','#bd764c']
for seed,color in enumerate(colors):ax.scatter([by[k]['paired_change_pct'][seed] for k in keys],[4-i+(seed-1)*.12 for i in range(5)],label=f'Seed label {seed}',color=color,s=45)
ax.scatter([by[k]['relative_mean_change_pct'] for k in keys],list(range(4,-1,-1)),marker='D',color='#222',s=25,label='Ratio of group means')
ax.axvline(0,color='#666',linestyle='--',linewidth=1);ax.set_yticks(range(4,-1,-1),labels);ax.set_xlim(-4.3,1.4);ax.set_xlabel('New / old BPB change (%)     Lower is better');ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False);ax.set_title('Three continuation seed labels: consistent gains and consistent regressions',fontsize=12);ax.legend(loc='lower left',ncol=2,fontsize=9)
fig.text(.02,.025,'Source: frozen six d512 swarm observations, dropless endpoints. Points are per-pair changes, not error bars.\nThe swarm helped select this mixture; no independent confirmation or post-selection p-value is supplied.\n25 endpoints (23 leaves + 2 macros): 20 improve in all pairs, 2 regress in all pairs, 3 have mixed directions.',fontsize=9)
fig.subplots_adjust(left=.24,right=.98,top=.88,bottom=.25)
fig.savefig(R/'assets/swarm_seed_pairs.png',dpi=150);fig.savefig(R/'assets/swarm_seed_pairs.svg',metadata={'Date':None});plt.close(fig)
p=R/'assets/swarm_seed_pairs.svg';s=p.read_text()
for old in sorted(set(re.findall(r'id="([^"]+)"',s)),key=len,reverse=True):s=s.replace('id="'+old+'"','id="swarmpair-'+old+'"').replace('#'+old+'"','#swarmpair-'+old+'"').replace('#'+old+')','#swarmpair-'+old+')')
p.write_text('\n'.join(x.rstrip() for x in s.splitlines())+'\n');print('Swarm paired endpoint figure exported')
