"""Analytic illustration, not observed training curves."""
import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];d=json.loads((R/'analysis/muon_geometry_probe.json').read_text());eta=.2;p=np.array([1.,0.]);u=np.array([eta,np.sqrt(1-eta**2)]);v=p-eta*u;q=v/np.linalg.norm(v)
fig,ax=plt.subplots(1,2,figsize=(10,4));a=ax[0];t=np.linspace(0,2*np.pi,400);a.plot(np.cos(t),np.sin(t),color='#ccd6dc');a.plot([0,1],[0,0],color='#62757d',lw=1);a.plot([0,v[0]],[0,v[1]],color='#138a8a',lw=1);a.plot([p[0],v[0]],[p[1],v[1]],'-o',label='Before projection: eta = 0.2');a.plot([v[0],q[0]],[v[1],q[1]],'-o',label='Radial projection');a.plot([p[0],q[0]],[p[1],q[1]],'--',color='#a75025',label='Final delta = 0.20102');
for point,label in [(p,'W'),(v,'V'),(q,"W'")]:a.annotate(label,point,xytext=(5,-13),textcoords='offset points')
a.set_xlim(.65,1.07);a.set_ylim(-.28,.1);a.set_aspect('equal');a.set_title('Projection changes the step bound');a.legend(fontsize=8,loc='upper left');a.grid(alpha=.15)
b=ax[1];before=d['four_dimensional_inner_norm_before'][0];after=d['four_dimensional_inner_norm_after'][0];x=np.arange(2);b.bar(x-.17,before,.34,label='Before');b.bar(x+.17,after,.34,label='After');b.set_xticks(x,['Inner matrix 0','Inner matrix 1']);b.set_ylabel('Frobenius norm');b.set_title('Joint norm stays sqrt(5); inner norms change');b.legend();b.set_ylim(0,2.35)
for i,(old,new) in enumerate(zip(before,after)):b.text(i-.17,old+.04,f'{old:.3f}',ha='center',fontsize=9);b.text(i+.17,new+.04,f'{new:.3f}',ha='center',fontsize=9)
fig.suptitle('Artificial CPU inputs; eta=0.2; no training / SPMD measurement',fontsize=11);fig.tight_layout();fig.savefig(R/'assets/muon_geometry.svg');fig.savefig(R/'assets/muon_geometry.png',dpi=180);plt.close(fig)

import re
p=R/'assets/muon_geometry.svg';s=p.read_text();ids=re.findall(r'id="([^" ]+)"',s)
for name in sorted(ids,key=len,reverse=True):s=s.replace('id="'+name+'"','id="mg-'+name+'"').replace('#'+name+'"','#mg-'+name+'"').replace('#'+name+')','#mg-'+name+')')
p.write_text(s)
