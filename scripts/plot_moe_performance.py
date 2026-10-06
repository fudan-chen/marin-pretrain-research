"""Plot author-reported sequential times and derived metrics; schematic is not a GPU trace."""
import json,pathlib,re
import matplotlib
matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='marin-moe-performance-20261007'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=pathlib.Path(__file__).resolve().parents[1];d=json.loads((R/'analysis/moe_performance_attribution.json').read_text());rows=d['timed_rows'];t=[r['seconds'] for r in rows]
labels=['main','remat','buffers','routing grad','short conv','dSwiGLU','top-k','scatter','overlap','shared GEMM','skip stores','Pallas']
fig=plt.figure(figsize=(12,8.4));gs=fig.add_gridspec(2,2,height_ratios=[1.35,1],hspace=.65,wspace=.3)
a=fig.add_subplot(gs[0,:]);a.plot(range(len(t)),t,color='#245c85',marker='o',linewidth=2);a.scatter([9],[t[9]],color='#b34a27',s=85,zorder=4)
for i,v in enumerate(t):a.annotate(f'{v:.3f}',(i,v),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
a.set_xticks(range(12),labels,rotation=30,ha='right');a.set_ylim(12.3,14.2);a.set_ylabel('Median step time (s)');a.set_title('A  Author-reported one-rack series; each row has a different cumulative code base',loc='left',fontsize=11);a.grid(axis='y',alpha=.2);a.spines[['top','right']].set_visible(False)
b=fig.add_subplot(gs[1,0]);ag=d['aggregate'];vals=[100*ag['time_reduction_fraction'],100*ag['relative_speed_gain']];b.barh([1,0],vals,color=['#58846b','#245c85']);b.set_yticks([1,0],['Step time reduction','Relative speed gain']);b.set_xlim(0,12.5)
for y,v in zip([1,0],vals):b.text(v+.12,y,f'{v:.3f}%',va='center',fontsize=10)
b.set_xlabel('Percent (different denominators)');b.set_title('B  Whole-series arithmetic, not component attribution',loc='left',fontsize=10);b.spines[['top','right']].set_visible(False)
c=fig.add_subplot(gs[1,1]);c.set_xlim(0,1);c.set_ylim(0,1);c.axis('off');c.set_title('C  Mechanism schematic; no device trace reproduced',loc='left',fontsize=10)
boxes=[(.06,.76,'Shared compute gets faster'),(.06,.46,'Less compute covers transport'),(.06,.16,'More transport becomes exposed')]
for x,y,label in boxes:
 c.add_patch(FancyBboxPatch((x,y),.88,.18,boxstyle='round,pad=0.01',facecolor='#eef2f4',edgecolor='#627887',linestyle='--'))
 c.text(x+.44,y+.09,label,ha='center',va='center',fontsize=10)
for y in [.76,.46]:c.annotate('',xy=(.5,y-.10),xytext=(.5,y-.01),arrowprops={'arrowstyle':'->','color':'#627887'})
fig.suptitle('Performance gains depend on the final graph and overlap',fontsize=14,y=.98)
fig.text(.02,.012,'Source: frozen PR #9708; GB200 64 GPUs, step 180000, seed 0, 100 steps. No error bars or raw step traces available here.\nOrange point: +0.014 s in series, but final-tip removal slowed author pairs by 0.35% / 0.22%. PR was unmerged at snapshot.',fontsize=9)
fig.subplots_adjust(left=.15,right=.98,top=.90,bottom=.16)
fig.savefig(R/'assets/moe_performance_attribution.png',dpi=150);fig.savefig(R/'assets/moe_performance_attribution.svg',metadata={'Date':None});plt.close(fig)
p=R/'assets/moe_performance_attribution.svg';s=p.read_text()
for old in sorted(set(re.findall(r'id="([^"]+)"',s)),key=len,reverse=True):s=s.replace('id="'+old+'"','id="moeperf-'+old+'"').replace('#'+old+'"','#moeperf-'+old+'"').replace('#'+old+')','#moeperf-'+old+')')
p.write_text('\n'.join(x.rstrip() for x in s.splitlines())+'\n');print('Performance figure exported: PNG and SVG')
