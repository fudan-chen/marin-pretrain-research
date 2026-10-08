"""Original-source synthetic LR curves; no training loss or historical runtime evidence."""
import hashlib,json,pathlib,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1];p=R/'analysis/schedule_config_cpu.json';x=json.loads(p.read_text());curves=x['curves']
plt.rcParams.update({'svg.hashsalt':'marin-schedule-config-v134','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
for key,label,style in [('fraction_1','Original warmup=1 (full fraction)','-'),('baseline','Original warmup=0.1','--'),('explicit_steps_1_candidate','Local candidate: explicit steps=1',':')]:
 rows=curves[key];axs[0].plot([r['count'] for r in rows],[r['lr'] for r in rows],style,lw=1.8,label=label)
for key,label,style in [('baseline','Valid baseline','--'),('beyond_N_endpoint','Original cycles=[120], N=100','-')]:
 rows=[r for r in curves[key] if r['count']<=110];axs[1].plot([r['count'] for r in rows],[r['lr'] for r in rows],style,lw=1.8,label=label)
axs[1].axvline(100,color='#777',lw=1,ls=':');axs[1].annotate('Jump at nominal N=100',xy=(100,.01),xytext=(45,.0088),arrowprops={'arrowstyle':'->','color':'#555'},fontsize=9)
for ax in axs:
 ax.set_xlabel('Absolute schedule count');ax.set_ylabel('Learning rate');ax.set_ylim(-.0004,.011);ax.grid(alpha=.17);ax.legend(fontsize=8,frameon=False,loc='lower left')
axs[0].set_title('One means a fraction in the original converter');axs[1].set_title('Accepted endpoints can produce a discontinuity')
fig.suptitle('Synthetic scheduler controls, N=100 and peak LR=0.01',fontsize=13)
fig.text(.02,.025,'Pinned original scheduler + Optax0.2.5. Left dotted line uses a local unit-converter candidate.\nNo configuration-parser, GPU, training loss, or actual Hero incident validation.',fontsize=9,color='#444')
fig.tight_layout(rect=[0,.13,1,.92]);svg=R/'assets/schedule_config.svg';png=R/'assets/schedule_config.png';fig.savefig(svg,metadata={'Date':None});fig.savefig(png,dpi=160);plt.close(fig)
s=svg.read_text()
for ident in sorted(set(re.findall(r'id="([^"]+)"',s)),key=len,reverse=True):s=s.replace('id="'+ident+'"','id="schedcfg-'+ident+'"').replace('#'+ident+'"','#schedcfg-'+ident+'"').replace('#'+ident+')','#schedcfg-'+ident+')')
s=re.sub(r'<svg\b([^>]*)>',lambda m:'<svg'+m.group(1)+' role="img" aria-labelledby="schedule-config-title schedule-config-desc"><title id="schedule-config-title">Synthetic learning-rate schedule configuration controls</title><desc id="schedule-config-desc">Original warmup1 ramps throughout100steps. Explicit one-step local candidate peaks at count1. An accepted endpoint120 beyond horizon100 yields a jump at count100. No actual Hero or training loss evidence.</desc>',s,count=1)
svg.write_text('\n'.join(line.rstrip() for line in s.splitlines()).rstrip()+'\n')
(R/'analysis/schedule_config_figure.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'figure_sha256':hashlib.sha256(svg.read_bytes()).hexdigest(),'png_sha256':hashlib.sha256(png.read_bytes()).hexdigest(),'source_kind':'original_source_synthetic_cpu_controls_and_local_candidate','actual_training_loss_effect':None},indent=2)+'\n');print('Schedule figure exported')
