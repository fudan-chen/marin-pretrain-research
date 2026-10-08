"""Scientific plot of four synthetic prefixes and fixed-key window interventions; no quality ranking."""
import pathlib,json,hashlib,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1];P=R/'analysis/prefix_sampling_structure.json';j=json.loads(P.read_text());ref=j['uniform_without_replacement_reference']['expected_distinct_io_blocks'];rows=j['four_prefixes'];windows=j['window_interventions']
matplotlib.rcParams.update({'svg.hashsalt':'marin-prefix-sampling-v125','axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axes=plt.subplots(1,2,figsize=(12.4,5.2))
ax=axes[0];y=[r['distinct_io_blocks'] for r in rows];ax.scatter(range(4),y,s=65,c='#276181',zorder=3)
for i,v in enumerate(y):ax.annotate(str(v),(i,v),xytext=(0,-18),textcoords='offset points',ha='center',fontsize=10)
ax.axhline(512,c='#b35c35',ls=':',lw=1.5,label='First-window support cap: 512 blocks');ax.set_xticks(range(4),['None/q0','None/q1','zero/q0','zero/q1']);ax.set_xlim(-.45,3.45);ax.set_title('Four fixed-key synthetic prefixes');ax.set_xlabel('data_seed / artificial child label');ax.set_ylabel('Distinct physical IO index blocks')
ax=axes[1];x=[r['window_blocks'] for r in windows];y2=[r['distinct_io_blocks'] for r in windows];ax.plot(x,y2,'o-',c='#3e846b',ms=6,lw=1.3)
for xx,yy in zip(x,y2):ax.annotate(str(yy),(xx,yy),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10)
ax.set_xscale('log',base=2);ax.set_xticks(x,list(map(str,x)));ax.set_xlim(.7,6000);ax.set_title('Window intervention, one fixed child key');ax.set_xlabel('window_blocks (log2 spacing)')
for ax in axes:ax.axhline(ref,c='#666',ls='--',lw=1.2,label='Uniform-subset expected count: 787.52');ax.set_ylim(0,950);ax.grid(axis='y',alpha=.16);ax.legend(loc='upper left',fontsize=8,frameon=False)
axes[1].set_ylim(-30,950)  # Keep the four-block marker fully visible at the zero baseline.
fig.suptitle('Block-shuffle prefixes are structured subsets, not a quality score',fontsize=13)
fig.text(.02,.025,'Synthetic N=1,000,000; prefix K=879; IO block=256. No real documents or training.\nDashed line: explicit uniform-without-replacement expectation, not a confidence interval.\nWindow changes alter selected IDs; more blocks is not proof of better data, faster IO, or lower loss.',fontsize=9,color='#444')
fig.tight_layout(rect=[0,.16,1,.92]);svg=R/'assets/prefix_sampling_structure.svg';png=R/'assets/prefix_sampling_structure.png';fig.savefig(svg,metadata={'Date':None});fig.savefig(png,dpi=160);plt.close(fig)
s=svg.read_text()
for ident in sorted(set(re.findall(r'id="([^"]+)"',s)),key=len,reverse=True):s=s.replace('id="'+ident+'"','id="pfxsample-'+ident+'"').replace('#'+ident+'"','#pfxsample-'+ident+'"').replace('#'+ident+')','#pfxsample-'+ident+')')
s=re.sub(r'<svg\b([^>]*)>',lambda m:'<svg'+m.group(1)+' role="img" aria-labelledby="prefix-sampling-title prefix-sampling-desc"><title id="prefix-sampling-title">Synthetic block-shuffle prefix structure and window interventions</title><desc id="prefix-sampling-desc">Four prefixes cover 438 to 452 IO blocks, below the first-window cap512. Uniform-subset expected coverage is787.52. Fixed-key window interventions cover4,64,442,873blocks; these are not quality or performance scores.</desc>',s,count=1);svg.write_text('\n'.join(line.rstrip() for line in s.splitlines()).rstrip()+'\n')
(R/'analysis/prefix_sampling_figure.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'figure_sha256':hashlib.sha256(svg.read_bytes()).hexdigest(),'png_sha256':hashlib.sha256(png.read_bytes()).hexdigest(),'observed_prefix_distinct_blocks':[r['distinct_io_blocks'] for r in rows],'window_values':x,'window_distinct_blocks':y2,'uniform_subset_expected_distinct_blocks':ref,'source_kind':'synthetic_original_source_cpu_control_and_explicit_mathematical_reference','actual_training_loss_effect':None},indent=2)+'\n');print('Prefix-sampling scientific figure exported')
