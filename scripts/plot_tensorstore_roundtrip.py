"""Plot measured local IO arrays; no production values."""
import json,pathlib,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1]
o=json.loads((R/'analysis/tensorstore_roundtrip.json').read_text())
plt.rcParams['svg.hashsalt']='tensorstore-io-v65'
fig,axes=plt.subplots(1,3,figsize=(11,4))
for ax,key,title in zip(axes,['complete_child','partial_child','deleted_chunk_child'],['Full write + commit','Only first chunk written','Logical c/0/0 deleted']):
 a=np.array(o['observations'][key]['values']);ax.imshow(a,vmin=0,vmax=24,cmap='Blues')
 for (i,j),v in np.ndenumerate(a):ax.text(j,i,str(int(v)),ha='center',va='center',color='white' if v>14 else 'black')
 ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Fresh-process reads: finite values do not establish content integrity\nSynthetic 6 x 4 array; local Zarr3/OCDBT; TensorStore 0.1.69')
fig.tight_layout(rect=(0,0,1,.85));fig.savefig(R/'assets/tensorstore_roundtrip.png',dpi=160);fig.savefig(R/'assets/tensorstore_roundtrip.svg',metadata={'Date':None})
p=R/'assets/tensorstore_roundtrip.svg';s=p.read_text();ids=re.findall(r'id="([^"]+)"',s)
for x in sorted(ids,key=len,reverse=True):s=s.replace('id="'+x+'"','id="tsio-'+x+'"').replace('#'+x+'"','#tsio-'+x+'"').replace('#'+x+')','#tsio-'+x+')')
p.write_text(s)
