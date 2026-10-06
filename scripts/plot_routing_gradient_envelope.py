"""Plot deliberately selected CPU scalar controls, not production incidence."""
import json,pathlib,re
import matplotlib
matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='marin-routing-envelope-v58'
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
R=pathlib.Path(__file__).resolve().parents[1];d=json.loads((R/'analysis/routing_gradient_envelope_cpu.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(12,5.3))
colors=['#e0e4e7','#b4d3c1','#b45d47']
for ax,name in zip(axes,['float16','bfloat16','float32']):
 rows=[r for r in d['grid'] if r['dtype_name']==name]
 grid=[[0 if r['input_cast_zero'] else 2 if r['positive_represented_inputs_lost_gradient'] else 1 for r in rows[i:i+4]] for i in range(0,16,4)]
 ax.imshow(grid,cmap=ListedColormap(colors),vmin=0,vmax=2)
 ax.set_xticks(range(4),['-4','-20','-60','-90']);ax.set_yticks(range(4),['-4','-10','-30','-60']);ax.set_xlabel('Cotangent exponent (base 2)');ax.set_title(name)
 for i in range(4):
  for j in range(4):ax.text(j,i,['cast 0','retained','lost'][grid[i][j]],ha='center',va='center',fontsize=9)
axes[0].set_ylabel('Weight exponent (base 2)')
fig.suptitle('Positive operands do not guarantee a represented product',fontsize=14)
fig.legend(handles=[Patch(color=c,label=t) for c,t in zip(colors,['At least one input casts to zero','Positive product retained','Positive inputs; product becomes zero'])],loc='lower center',bbox_to_anchor=(.5,.15),ncol=3,fontsize=9)
fig.text(.04,.025,'48 selected scalar CPU controls; y=3. These counts are not failure probabilities. GPU subnormal behavior is untested.\nA local dS discrepancy need not survive the router VJP: the extreme bf16 chain was zero in both paths.\nSource: frozen PR #9833 division/masking statements; row-dot producer and input distribution are synthetic.',fontsize=9)
fig.subplots_adjust(top=.80,bottom=.30,wspace=.35)
for suffix in ['png','svg']:fig.savefig(R/f'assets/routing_gradient_envelope.{suffix}',dpi=150,metadata={'Date':None} if suffix=='svg' else None)
plt.close(fig)
p=R/'assets/routing_gradient_envelope.svg';s=p.read_text()
for old in sorted(set(re.findall(r'id="([^"]+)"',s)),key=len,reverse=True):s=s.replace('id="'+old+'"','id="routeenv-'+old+'"').replace('#'+old+'"','#routeenv-'+old+'"').replace('#'+old+')','#routeenv-'+old+')')
seen={}
def unique_image_id(match):
 old=match.group(1);seen[old]=seen.get(old,0)+1
 return 'id="'+old+(('-'+str(seen[old])) if seen[old]>1 else '')+'"'
s=re.sub(r'id="(routeenv-image[^\"]+)"',unique_image_id,s)
p.write_text('\n'.join(x.rstrip() for x in s.splitlines())+'\n')
print('Routing envelope figure exported')
