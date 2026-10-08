"""Declared phase timeline with explicitly hypothetical aligned bridge."""
import pathlib,json,hashlib,re
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt']='marin-ablation-resume-v121'
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1];j=json.loads((R/'analysis/ablation_resume_boundary.json').read_text());g=j['geometry'];fig,ax=plt.subplots(figsize=(11,4.8));old='#d4dde4';new='#3c8490'
for y,end,label in [(2,480,'Producer declaration'),(1,384,'Continuation declaration'),(0,432,'Hypothetical aligned bridge')]:
 ax.barh(y,end-336,left=336,height=.5,color=old);ax.barh(y,480-end,left=end,height=.5,color=new);ax.text(338,y,'original mixture',va='center',fontsize=10)
 if end<480:ax.text(end+2,y,'new mixture',va='center',fontsize=10,color='white')
ax.axvline(393,color='#b64736',lw=1.8);ax.axvline(384,color='#777',ls=':',lw=1);ax.axvline(432,color='#777',ls=':',lw=1)
ax.annotate('restore at 393',(393,2.3),(407,2.8),arrowprops={'arrowstyle':'->','color':'#b64736'},color='#a33729',fontsize=11)
ax.annotate('',xy=(384,1.45),xytext=(393,1.45),arrowprops={'arrowstyle':'|-|','color':'#a33729'});ax.text(387,1.65,'9 updates = 9,216 sequence slots',ha='center',fontsize=10,color='#a33729')
ax.set_yticks([2,1,0],['Producer declaration','Continuation declaration','Hypothetical bridge']);ax.set_xlim(336,480);ax.set_ylim(-.6,3.1);ax.set_xticks([336,384,393,432,480]);ax.tick_params(axis='x',labelrotation=35);ax.set_xlabel('Absolute update boundary; 48 updates per mixture block');ax.set_title('The restored model and the rebuilt data prefix can use different histories',pad=18,fontsize=13)
ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);fig.text(.02,.01,'Declarations from public configs; the bridge is unexecuted.\nConditional logical-ID replay does not establish historical repeated tokens or a loss effect.',fontsize=9,color='#444');fig.tight_layout(rect=[0,.09,1,1])
svg=R/'assets/ablation_resume_boundary.svg';png=R/'assets/ablation_resume_boundary.png';fig.savefig(svg,metadata={'Date':None});fig.savefig(png,dpi=160);plt.close(fig);s=svg.read_text()
for ident in sorted(set(re.findall(r'id="([^"]+)"',s)),key=len,reverse=True):s=s.replace('id="'+ident+'"','id="abresume-'+ident+'"').replace('#'+ident+'"','#abresume-'+ident+'"').replace('#'+ident+')','#abresume-'+ident+')')
s=re.sub(r'<svg\b([^>]*)>',lambda m:'<svg'+m.group(1)+' role="img" aria-labelledby="ablation-resume-title ablation-resume-desc"><title id="ablation-resume-title">Declared producer and continuation schedules disagree for nine updates before restore</title><desc id="ablation-resume-desc">Producer retains original mixture through step393. Continuation declares new weights from384. Unexecuted aligned bridge starts new weights at432. Historical repetition is unverified.</desc>',s,count=1);svg.write_text(s+'\n')
(R/'analysis/ablation_resume_boundary_figure.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256((R/'analysis/ablation_resume_boundary.json').read_bytes()).hexdigest(),'figure_sha256':hashlib.sha256(svg.read_bytes()).hexdigest(),'png_sha256':hashlib.sha256(png.read_bytes()).hexdigest(),'geometry':g,'actual_bridge_training_result':None,'actual_browser_rendering':None},indent=2)+'\n');print('Resume-boundary figure exported')
