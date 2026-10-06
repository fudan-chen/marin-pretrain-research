"""Plot measured CPU four-cell losses and descriptive arithmetic allocation."""
import json,pathlib,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1]
o=json.loads((R/'analysis/loss_cross_replay_cpu.json').read_text())['observations']['pure_ce']
plt.rcParams['svg.hashsalt']='loss-cross-v73'
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.4),gridspec_kw={'width_ratios':[1,1.4]})
c=o['cells'];matrix=np.array([[c['00'],c['01']],[c['10'],c['11']]])
ax.imshow(matrix,cmap='Blues',vmin=0,vmax=2.5)
for i in range(2):
 for j in range(2):ax.text(j,i,f'{matrix[i,j]:.6f}',ha='center',va='center',color='white' if matrix[i,j]>1.5 else '#123342')
ax.set_xticks([0,1],['Old weights\nA1 / B3','New weights\nA3 / B1'])
ax.set_yticks([0,1],['Old predictions','New predictions']);ax.set_title('Same labels / context / target support')
d=o['allocation'];vals=[d['composition_at_old_prediction'],d['prediction_at_old_weights'],d['interaction'],d['total']]
bx.barh(range(4),vals,color=['#176c76','#176c76','#ad641d','#30466b'])
for i,v in enumerate(vals):bx.text(v+(.05 if v>=0 else -.05),i,f'{v:+.6f}',ha='left' if v>=0 else 'right',va='center')
bx.set_yticks(range(4),['Composition at old predictions','Prediction at old weights','Interaction','Total']);bx.invert_yaxis();bx.axvline(0,color='#777',lw=.8);bx.set_xlim(-2.8,2.8);bx.set_xlabel('Change in mean pure CE');bx.spines[['top','right']].set_visible(False)
fig.suptitle('Four-cell replay: descriptive allocation depends on the reference distribution\nSynthetic CPU source-method control; no Hero or training-effect estimate',fontsize=11)
fig.tight_layout(rect=(0,0,1,.87));fig.savefig(R/'assets/loss_cross_replay.png',dpi=160);fig.savefig(R/'assets/loss_cross_replay.svg',metadata={'Date':None})
p=R/'assets/loss_cross_replay.svg';s=p.read_text()
for x in sorted(re.findall(r'id="([^"]+)"',s),key=len,reverse=True):s=s.replace('id="'+x+'"','id="losscross-'+x+'"').replace('#'+x+'"','#losscross-'+x+'"').replace('#'+x+')','#losscross-'+x+')')
p.write_text(s)
