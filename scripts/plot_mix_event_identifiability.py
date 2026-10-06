"""Plot retained event sampling and design aliasing, not estimated loss effects."""
import json,pathlib,re
import matplotlib
matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='marin-mix-event-v61'
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
R=pathlib.Path(__file__).resolve().parents[1];d=json.loads((R/'analysis/mix_event_identifiability.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,5));a,b=axes
a.set_xlim(107.5,111.5);a.set_ylim(-.4,2.3);a.axvspan(108,108.778,color='#edcdbd',alpha=.7);a.plot([107.999,110.999],[0,0],'o',color='#245987',markersize=9)
a.vlines([108,108.778],-.2,1.5,colors=['#b75c46','#927831'],linestyles='--');a.annotate('Mix switch\n108000',xy=(108,1.45),xytext=(107.6,1.75),fontsize=10,arrowprops={'arrowstyle':'->'});a.annotate('Execution handoff\n108778',xy=(108.778,1.4),xytext=(109.05,1.7),fontsize=10,arrowprops={'arrowstyle':'->'})
a.text(108.389,.75,'No retained\nevaluation point',ha='center',fontsize=10);a.text(107.999,-.22,'107999',ha='center',fontsize=9);a.text(110.999,-.22,'110999',ha='center',fontsize=9);a.set_yticks([]);a.set_xlabel('Logged step / 1000');a.set_title('A  Nearest retained evaluation points span both changes',fontsize=10);a.spines[['top','right','left']].set_visible(False)
steps=[104999,107999,110999,113999];matrix=[[1,int(s>=108000),int(s>=108778)] for s in steps];b.imshow(matrix,cmap=ListedColormap(['#edf0f2','#9ac1d5']),vmin=0,vmax=1,aspect='auto')
b.set_xticks(range(3),['Intercept','Mix after','Execution after']);b.set_yticks(range(4),[str(s) for s in steps]);b.set_ylabel('Observed step (four-row excerpt)')
for i in range(4):
 for j in range(3):b.text(j,i,str(matrix[i][j]),ha='center',va='center',fontsize=12)
b.set_title('B  Event columns identical at all 67 retained timestamps',fontsize=10)
fig.suptitle('The archived series cannot separate these two event coefficients',fontsize=14)
fig.text(.02,.02,'Level design: rank 2 / 3 columns. Segmented level + slope design: rank 4 / 6 columns.\nThis is a design audit, not a causal estimate. No claim that the live service has no intermediate evaluations.\nIntermediate synthetic timestamp controls improve rank but do not supply counterfactual training or matching eval identity.',fontsize=9)
fig.subplots_adjust(top=.78,bottom=.28,left=.06,right=.98,wspace=.45)
fig.savefig(R/'assets/mix_event_identifiability.png',dpi=150);fig.savefig(R/'assets/mix_event_identifiability.svg',metadata={'Date':None});plt.close(fig)
p=R/'assets/mix_event_identifiability.svg';s=p.read_text()
for old in sorted(set(re.findall(r'id="([^"]+)"',s)),key=len,reverse=True):s=s.replace('id="'+old+'"','id="mixevent-'+old+'"').replace('#'+old+'"','#mixevent-'+old+'"').replace('#'+old+')','#mixevent-'+old+')')
p.write_text('\n'.join(x.rstrip() for x in s.splitlines())+'\n');print('Event design figure exported')
