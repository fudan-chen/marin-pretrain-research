# -*- coding: utf-8 -*-
"""Three distinct information tasks: endpoint directions, fit assumptions, batching counterexample."""
import csv
import json
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap
R=pathlib.Path(__file__).resolve().parents[1];A=R/'analysis';O=R/'assets'
font=os.environ.get('MARIN_REPORT_FONT') or next((p for p in ['/System/Library/Fonts/Supplemental/Arial Unicode.ttf','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'] if pathlib.Path(p).exists()),None)
if not font:raise RuntimeError('Set MARIN_REPORT_FONT to a Chinese font.')
font_manager.fontManager.addfont(font);plt.rcParams['font.family']=FontProperties(fname=font).get_name()
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'figure.facecolor':'white','axes.unicode_minus':False,'savefig.dpi':180,'svg.fonttype':'path','svg.hashsalt':'marin-scale-2026-10-05'})
def save(fig,name):
    fig.savefig(O/(name+'.png'),bbox_inches='tight');p=O/(name+'.svg');fig.savefig(p,bbox_inches='tight',metadata={'Date':None});p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');plt.close(fig)
rs=list(csv.DictReader((A/'scale_endpoints.csv').open()));sizes=['d768','d1024','d1536']
names={'4chan':'4chan','c4_100_domains':'C4 100域','c4_en':'C4英文','dolma-v1_5':'Dolma','dolma_100_programing_languages':'编程语言文本 *','dolma_100_subreddits':'Reddit 100社区','falcon-refinedweb':'Falcon RefinedWeb','gab':'Gab','m2d2_s2orc_unsplit':'S2ORC学术文本','m2d2_wikipedia_unsplit':'Wikipedia','manosphere_meta_sep':'Manosphere','mc4':'mC4','ptb':'PTB *','redpajama':'RedPajama *','twitterAAE_HELM_fixed':'Twitter AAE','wikitext_103':'WikiText-103 *'}
subsets=list(names);z=np.array([[float(next(r['bpb_change_pct'] for r in rs if r['size']==s and r['subset']==k)) for s in sizes] for k in subsets])
fig,ax=plt.subplots(figsize=(10.4,8.7));cmap=LinearSegmentedColormap.from_list('direction',['#287c82','#faf9f5','#b96435'])
im=ax.imshow(z,cmap=cmap,vmin=-2.1,vmax=2.1,aspect='auto');ax.set_xticks(range(3),sizes);ax.set_yticks(range(16),[names[k] for k in subsets]);ax.tick_params(length=0);ax.xaxis.tick_top()
for i in range(16):
    for j in range(3):ax.text(j,i,'%+.3f%%'%z[i,j],ha='center',va='center',color='white' if abs(z[i,j])>1.1 else '#222',fontsize=10)
ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,16,1),minor=True);ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='minor',bottom=False,left=False)
ax.set_title('16个验证子集：四项在不同 ladder 上改变收益方向',fontsize=14,pad=30)
bar=fig.colorbar(im,ax=ax,fraction=.038,pad=.045);bar.set_label('新/旧 BPB − 1（%）；负值改善')
fig.text(.025,.017,'* 为方向改变的子集。各列改善数：11/16、12/16、15/16；macro：−0.405%、−0.696%、−0.716%。\n每个端点为一个旧/新 run 的观察比较，无独立重复误差条；早期配比、硬件、EP、评估batch等不同。不能据此把反转归因给模型规模。',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.09,1,1]);save(fig,'scale_direction_matrix')

ps=list(csv.DictReader((A/'scale_fit_sensitivity.csv').open()));fig,axes=plt.subplots(1,2,figsize=(12.2,5.2))
for ax,key,label in zip(axes,['macro','dolma_100_programing_languages'],['Paloma macro','编程语言文本']):
    rows=[r for r in ps if r['subset']==key];x=[float(r['floor_fraction']) for r in rows]
    ax.plot(x,[float(r['refit_recentered_speedup']) for r in rows],marker='o',color='#286a70',label='每个floor重新拟合α')
    ax.plot(x,[float(r['fixed_zero_floor_alpha_speedup']) for r in rows],marker='x',ls='--',color='#bd773e',label='沿用零floor的α（对照假设）')
    ax.set_xticks(x,['0','0.1','0.25','0.5','0.75','0.9']);ax.set_xlabel('人为选定的floor / 最小旧BPB');ax.set_ylabel('d1536端点的等效算力倍数');ax.set_title(label);ax.grid(alpha=.2);ax.legend(fontsize=9)
    for idx in [0,3,5]:ax.annotate('%.3f×'%float(rows[idx]['refit_recentered_speedup']),(x[idx],float(rows[idx]['refit_recentered_speedup'])),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
fig.suptitle('“1.20×”怎样依赖拟合假设：floor与斜率必须一起检查',fontsize=14)
fig.text(.025,.014,'只有三个旧ladder端点。横轴是诊断假设，连线仅辅助阅读；不是估计的不可约loss，也不是置信区间。\n图中倍数由同端点BPB差倒算，throughput ratio设为1；没有测得运行时间或535B算力节省。',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.14,1,.94]);save(fig,'scale_fit_assumptions')

t=json.loads((A/'scale_batch_counterexample.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11.6,5.2))
for ax,field,title in zip(axes,['batch_token_weighted_bpb','global_ratio_bpb'],['先算batch BPB，再按token平均','先累计NLL与byte，再取比值']):
    x=np.arange(2);width=.32
    for offset,model,color in [(-width/2,'old','#777'),(width/2,'new','#286a70')]:
        vals=[r[model+'_'+field] for r in t['results']];ax.bar(x+offset,vals,width,color=color,label='模型'+('旧' if model=='old' else '新'))
        for xx,v in zip(x+offset,vals):ax.text(xx,v+.017,'%.3f'%v,ha='center',fontsize=9)
    ax.set_xticks(x,['A/B分成两批','A/B合成一批']);ax.set_ylim(0,1.04);ax.set_ylabel('BPB');ax.set_title(title,fontsize=11);ax.legend(fontsize=9);ax.grid(axis='y',alpha=.15)
    changes=[r['logged_formula_change_pct' if field=='batch_token_weighted_bpb' else 'global_ratio_change_pct'] for r in t['results']]
    for xx,v in zip(x,changes):ax.text(xx,.96,'新/旧 %+.2f%%'%v,ha='center',fontsize=10,color='#b35c35' if v>0 else '#286a70')
fig.suptitle('同一组预测，仅改变分批，也能让BPB排名反转',fontsize=14)
fig.text(.025,.014,'明确的人工反例：A/B各100 tokens，byte分别100/1000；旧NLL为100/100，新为110/50。两种分批的CE都为旧1.0、新0.8。\n左图按固定公共eval.py公式做本地算术回放，右图是分批不变的全局比值；不是历史checkpoint评估或已确认的历史根因。',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.14,1,.94]);save(fig,'scale_batch_counterexample')
