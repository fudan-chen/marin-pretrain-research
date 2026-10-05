"""Artificial optimizer-state comparison, not training outcomes."""
import pathlib,json,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1];d=json.loads((R/'analysis/adamh_probe.json').read_text());fig,ax=plt.subplots(1,2,figsize=(9,3.7));ax[0].bar(['Retain moments','Reset moments'],[d['retained_norm'],d['reset_norm']],color=['#1c8088','#c87340']);ax[0].set_ylabel('Final parameter Frobenius norm');ax[0].set_ylim(0,2.7);ax[0].set_title('Norm check passes in both paths');
for i,v in enumerate([d['retained_norm'],d['reset_norm']]):ax[0].text(i,v+.06,f'{v:.5f}',ha='center')
ax[1].bar(['Same copied state','Reset moments'],[0,d['angle_between_retained_and_reset_degrees']],color=['#1c8088','#c87340']);ax[1].set_ylabel('Angle vs retained-state output (degrees)');ax[1].set_ylim(0,4.8);ax[1].set_title('Direction reveals the state change');ax[1].text(1,d['angle_between_retained_and_reset_degrees']+.12,f"{d['angle_between_retained_and_reset_degrees']:.3f} degrees",ha='center');fig.suptitle('Artificial two-step inputs; real-array NumPy helper substitutes; fixed LR=0.1',fontsize=10);fig.tight_layout();fig.savefig(R/'assets/adamh_state.png',dpi=180);fig.savefig(R/'assets/adamh_state.svg');plt.close(fig);p=R/'assets/adamh_state.svg';s=p.read_text()
for name in sorted(re.findall(r'id="([^" ]+)"',s),key=len,reverse=True):s=s.replace('id="'+name+'"','id="ah-'+name+'"').replace('#'+name+'"','#ah-'+name+'"').replace('#'+name+')','#ah-'+name+')')
p.write_text(s)
