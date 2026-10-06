"""Measured finite/nonfinite counts in local portable CPU fixture, no transport test."""
import pathlib,json,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1]
coverage=json.loads((R/'analysis/portable_expert_mlp_cpu.json').read_text())['numeric_coverage']
plt.rcParams['svg.hashsalt']='portable-expert-v78'
fig,ax=plt.subplots(figsize=(10,4.5))
labels=['Input gradient dx','Expert gradient dW13','Expert gradient dW2','Raw row dot (7 active / 3 inactive)','Selected routing-weight gradient']
for i,(k,v) in enumerate(coverage.items()):
 ax.barh(i,100*v['finite']/v['elements'],color='#176c76')
 if v['nonfinite']:ax.barh(i,100*v['nonfinite']/v['elements'],left=100*v['finite']/v['elements'],color='#b06323')
 ax.text(102,i,f"finite {v['finite']}/{v['elements']} | nonfinite {v['nonfinite']}",va='center',fontsize=10)
ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_xlim(0,152);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Share of array elements (%)');ax.spines[['top','right']].set_visible(False)
fig.suptitle('Inactive NaN is allowed in the raw row dot, then excluded by caller selection\nOriginal portable CPU/XLA path; synthetic identity row mapping, no EP collective or GPU',fontsize=11)
fig.tight_layout(rect=(0,0,1,.86));fig.savefig(R/'assets/portable_expert_mlp.png',dpi=160);fig.savefig(R/'assets/portable_expert_mlp.svg',metadata={'Date':None})
p=R/'assets/portable_expert_mlp.svg';s=p.read_text()
for x in sorted(re.findall(r'id="([^"]+)"',s),key=len,reverse=True):s=s.replace('id="'+x+'"','id="portableep-'+x+'"').replace('#'+x+'"','#portableep-'+x+'"').replace('#'+x+')','#portableep-'+x+')')
p.write_text(s)
