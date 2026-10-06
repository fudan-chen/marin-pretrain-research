"""Measured domain counts in synthetic original-method controls."""
import json,pathlib,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1];o=json.loads((R/'analysis/mixture_boundary_logging.json').read_text())['observations'];plt.rcParams['svg.hashsalt']='mixture-boundary-v71'
keys=['normal_before','normal_after','frozen_stages_new_loader_crossing','reconverted_new_schedule_before','reconverted_new_schedule_after'];labels=['Normal / before switch','Normal / after switch','Frozen old stages + new loader','Reconverted / before switch','Reconverted / after switch']
fig,ax=plt.subplots(figsize=(11,4.7))
for i,(k,label) in enumerate(zip(keys,labels)):
 row=o[k];a=row['domain_counts'].get('A',0);b=row['domain_counts'].get('B',0);ax.barh(i,a,color='#176c76');ax.barh(i,b,left=a,color='#a15e12')
 if a:ax.text(a/2,i,'A: '+str(a),ha='center',va='center',color='white')
 if b:ax.text(a+b/2,i,'B: '+str(b),ha='center',va='center',color='white')
 ax.text(8.3,i,'logged stage '+str(row['stage_log']['values']['mixture/stage']),va='center',fontsize=10)
ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_xlim(0,10.3);ax.set_xticks([0,2,4,6,8]);ax.set_xlabel('Observed samples per returned batch (identity fixtures, not tokens)');ax.spines[['top','right']].set_visible(False)
fig.suptitle('Start-stage logs do not measure whole-batch composition\nCrossing case deliberately mismatches stored stage offsets and loader schedule',fontsize=12);fig.tight_layout(rect=(0,0,1,.87));fig.savefig(R/'assets/mixture_boundary_logging.png',dpi=160);fig.savefig(R/'assets/mixture_boundary_logging.svg',metadata={'Date':None})
p=R/'assets/mixture_boundary_logging.svg';s=p.read_text()
for x in sorted(re.findall(r'id="([^"]+)"',s),key=len,reverse=True):s=s.replace('id="'+x+'"','id="mixboundary-'+x+'"').replace('#'+x+'"','#mixboundary-'+x+'"').replace('#'+x+')','#mixboundary-'+x+')')
p.write_text(s)
