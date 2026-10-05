"""Artificial scalar spectrum and original matrix function with float32 substitute."""
import pathlib,json,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];d=json.loads((R/'analysis/muon_direction_probe.json').read_text());rows=d['trajectory'];x=[r['steps'] for r in rows];s=np.array([r['singular_values'] for r in rows]);fig,ax=plt.subplots(1,2,figsize=(10,4))
for i in range(4):ax[0].plot(x,s[:,i],'-o',ms=3,label=f'Sorted rank {i+1}')
ax[0].set_yscale('log');ax[0].axhline(1,color='#7d878b',ls='--');ax[0].set_title('Sorted singular values at each step');ax[0].set_xlabel('NS iterations (quintic coefficients cycle)');ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
ax[1].plot(x,[r['orthogonality_residual'] for r in rows],'-o',color='#bf673a');ax[1].scatter([5,6],[rows[5]['orthogonality_residual'],rows[6]['orthogonality_residual']],s=65,color='#155d71',zorder=3);ax[1].annotate('5 to 6 worsens residual',(6,rows[6]['orthogonality_residual']),xytext=(6.2,1.35),arrowprops={'arrowstyle':'->'},fontsize=9);ax[1].set_title('Frobenius residual ||XX^T - I||');ax[1].set_xlabel('NS iterations');ax[1].set_ylim(0,2.2);ax[1].grid(alpha=.2);fig.suptitle('Artificial diag(1, 0.2, 0.01, 1e-5); float32 substitutes BF16; no GPU result',fontsize=10);fig.tight_layout();fig.savefig(R/'assets/muon_direction.svg');fig.savefig(R/'assets/muon_direction.png',dpi=180);plt.close(fig);p=R/'assets/muon_direction.svg';s=p.read_text()
for name in sorted(re.findall(r'id="([^" ]+)"',s),key=len,reverse=True):s=s.replace('id="'+name+'"','id="md-'+name+'"').replace('#'+name+'"','#md-'+name+'"').replace('#'+name+')','#md-'+name+')')
p.write_text(s)
