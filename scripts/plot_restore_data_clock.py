"""Measured sample identities in synthetic joined restore/loader controls."""
import json,pathlib,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1];o=json.loads((R/'analysis/restore_data_clock.json').read_text())['observations']
plt.rcParams['svg.hashsalt']='restore-data-clock-v70'
fig,ax=plt.subplots(figsize=(12,4.8))
keys=['restored_step_batch','marker_step_counterfactual_batch','rewritten_history_batch','mixture_boundary_batch']
labels=['Restored state clock','Marker-clock counterfactual','Rewritten batch history','Next mixture boundary']
for i,(key,label) in enumerate(zip(keys,labels)):
 row=o[key];y=3-i;color='#176c76' if row['identities'][0].startswith('A:') else '#a15e12'
 ax.text(-.8,y+.15,label+'\nstep '+str(row['step'])+' / offset '+str(row['offset']),ha='right',va='center',fontsize=10)
 for j,sample in enumerate(row['identities']):ax.scatter(j,y,s=160,color=color);ax.text(j,y+.22,sample,ha='center',fontsize=10)
ax.set_xlim(-.5,7.5);ax.set_ylim(-.4,3.6);ax.set_yticks([]);ax.set_xticks(range(8));ax.set_xlabel('Position within returned batch (not training time)')
ax.spines[['left','right','top']].set_visible(False)
fig.suptitle('Checkpoint selection clock and data-consumption clock are separate\nMeasured original methods on synthetic identity stores; no Hero tokens',fontsize=12)
fig.subplots_adjust(left=.3,right=.98,top=.77,bottom=.14)
fig.savefig(R/'assets/restore_data_clock.png',dpi=160);fig.savefig(R/'assets/restore_data_clock.svg',metadata={'Date':None})
p=R/'assets/restore_data_clock.svg';s=p.read_text();ids=re.findall(r'id="([^"]+)"',s)
for x in sorted(ids,key=len,reverse=True):s=s.replace('id="'+x+'"','id="restoreclock-'+x+'"').replace('#'+x+'"','#restoreclock-'+x+'"').replace('#'+x+')','#restoreclock-'+x+')')
p.write_text(s)
