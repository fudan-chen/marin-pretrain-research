"""Measured acceptance from original metadata helpers on synthetic demand counts."""
import pathlib,json,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1];o=json.loads((R/'analysis/receiver_layout_cpu.json').read_text())['observations']['aggregate_room'];plt.rcParams['svg.hashsalt']='receiver-layout-v79'
fig,ax=plt.subplots(figsize=(9.5,4.4));labels=[]
for i,c in enumerate(o['chunks']):
 accepted=sum(c['active_group_sizes']);drop=c['raw_demand']-accepted;unused=o['logical_chunk_capacity']-accepted
 ax.barh(i,accepted,color='#176c76',label='Accepted' if i==0 else None)
 if drop:ax.barh(i,drop,left=accepted,color='#b06323',label='Dropped' if i==0 else None)
 if unused:ax.barh(i,unused,left=accepted,color='#c8d1d8',label='Unused logical capacity' if i==1 else None)
 ax.text(11.3,i,f"demand {c['raw_demand']} | accepted {accepted} | drop {drop}",va='center',fontsize=10)
 labels.append(f"Receiver {c['receiver']} / chunk {c['chunk']}")
ax.axvline(8,color='#30466b',linestyle='--',linewidth=1);ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlim(0,17);ax.set_xticks([0,2,4,6,8,10]);ax.set_xlabel('Assignment rows (logical capacity 8; physical buffer 10 per chunk)');ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower left',bbox_to_anchor=(0,-.32),ncol=3,frameon=False,fontsize=9)
fig.suptitle('Total demand = total logical capacity = 32; 5 assignments still drop\nOriginal CPU planning + host transfer replay; synthetic counts, no collective or GPU',fontsize=11);fig.tight_layout(rect=(0,0,1,.87));fig.savefig(R/'assets/receiver_layout.png',dpi=160);fig.savefig(R/'assets/receiver_layout.svg',metadata={'Date':None})
p=R/'assets/receiver_layout.svg';s=p.read_text()
for x in sorted(re.findall(r'id="([^"]+)"',s),key=len,reverse=True):s=s.replace('id="'+x+'"','id="receiverlayout-'+x+'"').replace('#'+x+'"','#receiverlayout-'+x+'"').replace('#'+x+')','#receiverlayout-'+x+')')
p.write_text(s)
