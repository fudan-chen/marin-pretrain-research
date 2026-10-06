"""CPU synthetic source-function measurements; no production loss attribution."""
import pathlib,json,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1]
c=json.loads((R/'analysis/post_clip_router_cpu.json').read_text())['cases']
plt.rcParams['svg.hashsalt']='post-clip-v80'
fig,axes=plt.subplots(1,2,figsize=(10,4.3))
labels=['Keep both','Keep first','Keep second','Drop both'];keys=['both','keep_first','keep_second','neither']
axes[0].bar(range(4),[c[k]['retained_weight_mass'] for k in keys],color='#176c76')
axes[0].axhline(2.5,linestyle='--',color='#30466b');axes[0].set_ylabel('Retained routing weight sum');axes[0].set_ylim(0,2.9)
for i,k in enumerate(keys):axes[0].text(i,c[k]['retained_weight_mass']+.08,f"{c[k]['retained_weight_mass']:.3f}",ha='center')
for i,k in enumerate(keys):
 for j in range(2):axes[1].bar(i+(j-.5)*.33,c[k]['router_logit_gradient'][0][j],width=.3,color=['#176c76','#b06323'][j],label=f'Selected logit {j}' if i==0 else None)
axes[1].axhline(0,color='#777',linewidth=.8);axes[1].set_ylabel('d(sum synthetic output) / d(router logit)');axes[1].legend(frameon=False,fontsize=9)
for ax in axes:ax.set_xticks(range(4),labels,rotation=15);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Dropping a selected assignment does not erase its normalization gradient\nOriginal route + portable combine on CPU; fixed acceptance and outputs, no GPU/model loss',fontsize=11)
fig.tight_layout(rect=(0,0,1,.85));fig.savefig(R/'assets/post_clip_router.png',dpi=160);fig.savefig(R/'assets/post_clip_router.svg',metadata={'Date':None})
p=R/'assets/post_clip_router.svg';s=p.read_text()
for x in sorted(re.findall(r'id="([^"]+)"',s),key=len,reverse=True):s=s.replace('id="'+x+'"','id="postclip-'+x+'"').replace('#'+x+'"','#postclip-'+x+'"').replace('#'+x+')','#postclip-'+x+')')
p.write_text(s)
