"""Plot deterministic synthetic CPU integration, not a production distribution."""
import json,pathlib,re
import matplotlib
matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='marin-router-coupling-v60'
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1];d=json.loads((R/'analysis/router_coupling_update_cpu.json').read_text());f=d['fixture'];o=d['optimizer_control']
fig,axes=plt.subplots(1,2,figsize=(11,5.2));a,b=axes
a.bar([0,1,2],f['router_gradient_candidate'][0],color=['#b75c46','#487d9c','#b4bac0']);a.axhline(0,color='#666',linewidth=.8);a.set_xticks([0,1,2],['Expert ID 0','Expert ID 1','Unselected ID 2']);a.set_ylabel('Router parameter gradient');a.ticklabel_format(axis='y',style='sci',scilimits=(0,0));a.set_title('A  One assignment error reaches two selected parameters',fontsize=10);a.text(.02,.96,'Reference gradient: [0, 0, 0]\nAll weighted cotangents are positive',transform=a.transAxes,va='top',fontsize=10)
vals=[o['fresh_update_difference_norm'],o['warm_update_difference_norm']];b.bar([0,1],vals,color=['#b75c46','#487d9c']);b.set_yscale('log');b.set_ylim(1e-10,.02);b.set_xticks([0,1],['Common fresh state','Common warm state']);b.set_ylabel('One-step update difference norm');b.set_title('B  The same gradient discrepancy depends on optimizer state',fontsize=10)
a.set_ylim(-1.4e-6,1.2e-6)
b.set_title('B  Update difference depends on optimizer state',fontsize=10)
for i,v in enumerate(vals):b.text(i,v*1.8,f'{v:.3e}',ha='center',fontsize=10)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('A local rounding error must be followed through the update',fontsize=14)
fig.text(.02,.025,'Synthetic CPU: original router block + portable PR row-dot/division + original decay wrapper + real Optax.\nConstant expert outputs; FP32 parameters, BF16 activation, path/reshard stubs, constant diagnostic LR.\nFresh/warm states are artificial. No Hero checkpoint, complete expert, GPU, or loss-effect measurement.',fontsize=9)
fig.subplots_adjust(left=.10,right=.98,top=.80,bottom=.27,wspace=.35)
fig.savefig(R/'assets/router_coupling_update.png',dpi=150);fig.savefig(R/'assets/router_coupling_update.svg',metadata={'Date':None});plt.close(fig)
p=R/'assets/router_coupling_update.svg';s=p.read_text()
for old in sorted(set(re.findall(r'id="([^"]+)"',s)),key=len,reverse=True):s=s.replace('id="'+old+'"','id="routercouple-'+old+'"').replace('#'+old+'"','#routercouple-'+old+'"').replace('#'+old+')','#routercouple-'+old+')')
p.write_text('\n'.join(x.rstrip() for x in s.splitlines())+'\n')
print('Coupling/update figure exported')
