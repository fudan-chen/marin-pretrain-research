"""Export figures from the archived public measurements. No network access."""
import json, csv, pathlib, statistics, collections, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
A=ROOT/'analysis'; OUT=ROOT/'assets'; OUT.mkdir(exist_ok=True)
font=os.environ.get('MARIN_REPORT_FONT')
if not font:
    font=next((p for p in ['/System/Library/Fonts/Supplemental/Arial Unicode.ttf','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'] if pathlib.Path(p).exists()),None)
if not font or not pathlib.Path(font).exists():
    raise RuntimeError('Set MARIN_REPORT_FONT to an available Chinese font path.')
from matplotlib import font_manager
font_manager.fontManager.addfont(font)
plt.rcParams['font.family']=FontProperties(fname=font).get_name()
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'font.size':10,'figure.facecolor':'white','axes.unicode_minus':False,'savefig.dpi':180,'svg.fonttype':'path'})
BLUE='#224f77'; ORANGE='#c07a2c'; RED='#a7443a'; GREEN='#3b7769'
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)
series=json.loads((A/'series.json').read_text())
events=[(58014,'Gate/router WD'),(81716,'Ragged EP'),(108000,'新配比'),(108778,'PDL off'),(121638,'Main'),(146139,'FA4 / masks')]
def event_lines(ax):
    for s,n in events: ax.axvline(s/1000,color='#999',lw=.7,alpha=.55)
fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
metrics=[('train/cross_entropy_loss','训练 CE / nats',1),('eval_dropless/paloma/macro_loss','固定 Paloma macro loss / nats',1),('moe/drop_fraction','专家分配丢弃比例 / %',100),('throughput/tokens_per_second','吞吐 / 百万 tokens/s',1e-6)]
for ax,(metric,label,scale) in zip(axes,metrics):
    rows=[x for x in series[metric] if x['step']>=5000]
    x=np.array([p['step']/1000 for p in rows]);y=np.array([p['value']*scale for p in rows])
    if metric.startswith('eval'):
        ax.plot(x,y,'o-',ms=3,lw=1.2,color=BLUE,label='公开固定评估点')
    else:
        ax.scatter(x,y,s=2,alpha=.14,color=BLUE,rasterized=True,label='W&B抽样点')
        buckets=collections.defaultdict(list)
        for p in rows:buckets[(p['step']//1000,p['run'])].append(p)
        pts=[(statistics.median(p['step'] for p in g)/1000,statistics.median(p['value'] for p in g)*scale) for _,g in sorted(buckets.items())]
        ax.plot([p[0] for p in pts],[p[1] for p in pts],color=BLUE,lw=1.3,label='每1000 steps内中位数（不跨run）')
    event_lines(ax);ax.set_ylabel(label);ax.legend(loc='upper right',fontsize=8)
axes[0].set_ylim(1.16,1.60)
axes[3].set_ylim(0,3.6)
for s,n in [(81716,'EP切换'),(108000,'配比切换'),(146139,'内核切换')]:
    axes[0].annotate(n,xy=(s/1000,1.58),xytext=(s/1000+1,1.58),rotation=90,fontsize=8,va='top')
axes[-1].set_xlabel('累计 optimizer step / 千步（按7段生产run拼接）')
fig.suptitle('同一条训练时间线，四种不同量',fontsize=17,y=.995)
fig.text(.06,.01,'来源：公开W&B sampledHistory。0–5k warmup未画；CE纵轴聚焦1.16–1.60。抽样点可能漏掉瞬时尖峰。\n固定评估仅用eval_dropless/paloma/macro_loss；run继承历史按生效区间剔除，未拼入失败的试部署。',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.06,1,.975]);save(fig,'training_signals')

def window(name,key):
    j=json.loads((ROOT/'sources/wandb'/('window_'+name+'.json')).read_text())
    i=next(i for i,s in enumerate(j['specs']) if s['keys'][-1]==key)
    return {p['_step']:p[key] for p in j['series'][i] if key in p}
fig,axes=plt.subplots(3,2,figsize=(12,10))
panels=[('ep_control','ep_new',81716,81916,'81k：相同批次，专家通信切换'),('mix_before','mix_after',108000,108200,'108k：相同步号，不同抽样分布'),('kernel_control','kernel_new',146139,146339,'146k：相同批次，内核整包切换')]
for r,(old,new,start,end,title) in enumerate(panels):
    for c,key in enumerate(['train/cross_entropy_loss','throughput/tokens_per_second']):
        ax=axes[r,c]
        for name,color,label in [(old,ORANGE,'旧'),(new,BLUE,'新')]:
            m=window(name,key);xs=sorted(s for s in m if start<=s<end)
            ax.plot([s-start for s in xs],[m[s]*(1e-6 if c else 1) for s in xs],color=color,lw=.8,alpha=.85,label=label)
        ax.set_title(title if c==0 else '吞吐（启动低值保留）',fontsize=11)
        ax.set_ylabel('CE / nats' if c==0 else '百万 tokens/s');ax.set_xlabel('距切换点的 updates');ax.legend(fontsize=9)
fig.suptitle('切换窗口：先判断批次与执行路径是否可比',fontsize=16)
fig.text(.05,.008,'81k与146k为200步逐step对照；108k改变配比，图中的旧/新CE不能解释为能力差值。\n146k同时改FA4、tail masks与GC，不能把全部吞吐增益归给其中一个。',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.055,1,.965]);save(fig,'intervention_windows')

domains=json.loads((A/'domain_weights.json').read_text())
labels=['%02d %s'%(d['id'],d['name']) for d in domains]
matrix=np.array([[d['phase%d_pct'%s] for s in range(3)] for d in domains])
fig,ax=plt.subplots(figsize=(11,15))
im=ax.imshow(matrix,cmap='Blues',aspect='auto',vmin=0,vmax=12)
ax.set_yticks(range(40),labels,fontsize=8);ax.set_xticks(range(3),['原配比 0–108k','新主阶段 108k–312192','Cooldown计划 312192–390251']);ax.grid(False)
for i in range(40):
    for j in range(3):ax.text(j,i,f'{matrix[i,j]:.2f}%',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>7 else '#213b50')
fig.colorbar(im,ax=ax,shrink=.5,label='训练抽样比例 / %')
ax.set_title('40个domain：实际三阶段配比（不是库存组成）',pad=15,fontsize=15)
fig.text(.02,.006,'各domain为5个质量档之和。名称沿用原报告；聚类内容并非纯净任务。Cooldown为未来计划。',fontsize=9)
fig.tight_layout(rect=[0,.025,1,1]);save(fig,'domain_mixture')

qs=json.loads((A/'quality_weights.json').read_text());x=np.arange(5);fig,ax=plt.subplots(figsize=(10,4.8))
for j,(k,label,col) in enumerate([('store_pct','库存','#bbb'),('phase0_pct','原配比',ORANGE),('phase1_pct','新主阶段',BLUE),('phase2_pct','Cooldown计划',GREEN)]):
    vals=[q[k] for q in qs];ax.bar(x+(j-1.5)*.19,vals,.18,label=label,color=col)
ax.set_xticks(x,['Q%d'%q for q in range(5)]);ax.set_ylabel('比例 / %');ax.set_title('新主阶段同时增加了Q0和Q1，Q4几乎不变');ax.legend(ncol=4,fontsize=9)
fig.tight_layout();save(fig,'quality_mixture')

rs=[r for r in csv.DictReader((ROOT/'sources/mix_study_final-2026.09.15.1_final_results.csv').open()) if r['metric']=='bpb']
rs=sorted(rs,key=lambda r:float(r['change_pct']))
fig,ax=plt.subplots(figsize=(11,7))
ys=np.arange(len(rs));values=[float(r['change_pct']) for r in rs]
ax.barh(ys,values,color=[GREEN if v<0 else RED for v in values]);ax.set_yticks(ys,[r['subset'] for r in rs],fontsize=9);ax.invert_yaxis();ax.axvline(0,color='#777',lw=.7)
ax.set_xlabel('相对旧配比BPB变化 / %（负值更好）');ax.set_title('d1536固定Paloma：整体改善，Wikipedia退化')
for i,v in enumerate(values):ax.text(v-(.03 if v<0 else -.03),i,f'{v:+.3f}%',va='center',ha='right' if v<0 else 'left',fontsize=8)
ax.set_xlim(-1.8,.75)
fig.text(.01,.004,'来源：final-2026.09.15.1 CSV。不是535B最终结果；旧/新ladder还有硬件、EP与optimizer等混杂。',fontsize=9)
fig.tight_layout(rect=[0,.035,1,1]);save(fig,'mixture_eval')
print('Exported five figures, PNG + SVG.')
