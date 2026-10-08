"""Observed baseline-relative ablation BPB scatter; no confidence intervals or fitted effects."""
import pathlib,csv,json,hashlib,re
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1];rows=list(csv.DictReader((R/'analysis/domain_ablation_audit.csv').open()));j=json.loads((R/'analysis/ablation_context.json').read_text());groups=['proportional_domain_ablation','semantic_domain_ablation'];by={(r['group'],int(r['domain'])):r for r in rows}
matplotlib.rcParams.update({'svg.hashsalt':'marin-ablation-context-v120','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axes=plt.subplots(1,3,figsize=(13,4.8));record=[]
for ax,(m,label,stat) in zip(axes,[('paloma','Paloma macro BPB','paloma_macro_bpb'),('gsm8k','GSM8K text BPB','logprob_gsm8k_5shot'),('humaneval','HumanEval text BPB','logprob_humaneval_10shot')]):
 x=np.array([float(by[groups[0],i][m+'_change_pct']) for i in range(40)]);y=np.array([float(by[groups[1],i][m+'_change_pct']) for i in range(40)]);flip=x*y<0
 ax.scatter(x[~flip],y[~flip],s=27,c='#276181',label='Same observed sign');ax.scatter(x[flip],y[flip],s=35,c='#b74f35',marker='D',label='Opposite observed signs');bound=max(abs(np.r_[x,y]))*1.12
 ax.set_xlim(-bound,bound);ax.set_ylim(-bound,bound);ax.axhline(0,c='#777',lw=.7);ax.axvline(0,c='#777',lw=.7);ax.set_aspect('equal',adjustable='box');ax.set_title(label+'\n'+str(int(flip.sum()))+'/40 opposite signs',fontsize=11)
 for i in [13,39]:
  offset=((-38,-21) if i==13 else (28,18)) if m=='paloma' else ((-30,13) if i==13 else (6,9))
  ax.annotate('c%02d'%i,(x[i],y[i]),xytext=offset,textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-','color':'#777','lw':.6})
 ax.set_xlabel('vs proportional reference (%)');record.append({'metric':m,'domain_points':40,'opposite_sign_count':int(flip.sum()),'points':[[float(a),float(b)] for a,b in zip(x,y)],'axis_limits':[-float(bound),float(bound)]})
axes[0].set_ylabel('vs broad-technical-capped reference (%)');axes[-1].legend(loc='lower right',fontsize=7,frameon=False)
fig.suptitle('Deleting one domain reallocates its budget: responses depend on reference mixture',fontsize=13)
fig.text(.015,.015,'Each point is one domain; negative is lower BPB. One seed per ablation; no significance claim.\nAll 40 points shown per panel, no clipping. Panel scales differ. Text BPB is not answer accuracy or pass@1.',fontsize=9,color='#444')
fig.tight_layout(rect=[0,.1,1,.92]);png=R/'assets/ablation_context.png';svg=R/'assets/ablation_context.svg';fig.savefig(png,dpi=160);fig.savefig(svg,metadata={'Date':None});plt.close(fig)
s=svg.read_text()
for ident in sorted(set(re.findall(r'id="([^"]+)"',s)),key=len,reverse=True):s=s.replace('id="'+ident+'"','id="ablcontext-'+ident+'"').replace('#'+ident+'"','#ablcontext-'+ident+'"').replace('#'+ident+')','#ablcontext-'+ident+')')
s=re.sub(r'<svg\b([^>]*)>',lambda m:'<svg'+m.group(1)+' role="img" aria-labelledby="ablation-context-title ablation-context-desc"><title id="ablation-context-title">Baseline-relative ablation BPB responses, forty domains in each panel</title><desc id="ablation-context-desc">Observed sign reversals: Paloma six, GSM8K nine, HumanEval thirty-one. Single-seed observations, not significance or causal data values.</desc>',s,count=1)
svg.write_text(s+'\n')
out={'panels':record,'analysis_sha256':hashlib.sha256((R/'analysis/ablation_context.json').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((R/'analysis/domain_ablation_audit.csv').read_bytes()).hexdigest(),'figure_sha256':hashlib.sha256(svg.read_bytes()).hexdigest(),'png_sha256':hashlib.sha256(png.read_bytes()).hexdigest(),'actual_browser_rendering':None}
(R/'analysis/ablation_context_figure.json').write_text(json.dumps(out,indent=2)+'\n');print('Ablation context figure exported')
