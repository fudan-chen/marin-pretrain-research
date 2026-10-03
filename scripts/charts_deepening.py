# -*- coding: utf-8 -*-
"""Export paired-seed, within-domain and complete-block cursor figures from local data."""
import pathlib,json,os,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
R=pathlib.Path(__file__).resolve().parents[1];A=R/'analysis';O=R/'assets'
font=os.environ.get('MARIN_REPORT_FONT') or next((p for p in ['/System/Library/Fonts/Supplemental/Arial Unicode.ttf','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'] if pathlib.Path(p).exists()),None)
if not font:raise RuntimeError('Set MARIN_REPORT_FONT to an available Chinese font.')
font_manager.fontManager.addfont(font);plt.rcParams['font.family']=FontProperties(fname=font).get_name()
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'font.size':10,'figure.facecolor':'white','axes.unicode_minus':False,'savefig.dpi':180,'svg.fonttype':'path','svg.hashsalt':'marin-535b-2026-10-04'})
def read(p):return json.loads(p.read_text())
def save(fig,name):
    fig.savefig(O/(name+'.png'),bbox_inches='tight')
    svg=O/(name+'.svg');fig.savefig(svg,bbox_inches='tight',metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
j=read(A/'paired_seed_metrics.json');lookup={x['metric']:x for rows in j.values() if isinstance(rows,list) for x in rows if isinstance(x,dict) and 'metric' in x}
choices=[('eval_dropless/paloma/macro_bpb','Paloma macro'),('eval_dropless/paloma/dolma_100_programing_languages-llama3/bpb','Paloma 编程文本'),('eval_dropless/uncheatable_eval/github_cpp-llama3/bpb','Uncheatable C++'),('logprob_humaneval_10shot','HumanEval 文本BPB'),('logprob_gsm8k_5shot','GSM8K 文本BPB'),('eval_dropless/uncheatable_eval/bbc_news-llama3/bpb','BBC 新闻'),('belebele_mean','Belebele mean'),('include_mean','Include mean'),('mmlu_pro_5shot','MMLU-Pro 文本BPB')]
# Namespace of uncheatable is preserved as exported in the archive.
for i,(k,label) in enumerate(choices):
    if k not in lookup:
        tail=k.split('eval_dropless/')[-1];matches=[x for x in lookup if x.endswith(tail)]
        if not matches:matches=[x for x in lookup if ('github_cpp' if 'C++' in label else 'bbc_news') in x]
        assert len(matches)==1,(k,matches);choices[i]=(matches[0],label)
fig,ax=plt.subplots(figsize=(11,6.8));y=np.arange(len(choices));colors=['#224f77','#c07a2c','#3b7769']
for seed,color in enumerate(colors):ax.scatter([lookup[k]['seed%d_change_pct'%seed] for k,_ in choices],y+(seed-1)*.13,color=color,s=35,label='seed%d'%seed,zorder=3)
means=[lookup[k]['change_pct'] for k,_ in choices];ax.scatter(means,y,marker='D',s=24,color='#222',label='三seed均值之比',zorder=4)
for i,(k,_) in enumerate(choices):
    vals=[lookup[k]['seed%d_change_pct'%s] for s in range(3)];ax.plot([min(vals),max(vals)],[i,i],color='#aaa',lw=1,zorder=1)
ax.axvline(0,color='#666',lw=.8);ax.set_yticks(y,[l for _,l in choices]);ax.invert_yaxis();ax.set_xlim(-12,18);ax.set_xlabel('BPB相对旧配比变化 / %（负值更好）');ax.set_title('同一组d512代理：代码获益，数学文本BPB变差',fontsize=15,pad=16);ax.legend(loc='lower right',fontsize=9)
fig.text(.02,.008,'固定文本来自eval_dropless，任务来自grouped_bpb；均为BPB，不是准确率或pass@1。\n线段是三个seed的最小/最大值，不是置信区间。候选参与了选择；完整54项任务在正文附录。',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.07,1,1]);save(fig,'paired_seed_tradeoffs')
cells=read(A/'cell_weights.json');domains=read(A/'domain_weights_zh.json');matrix=np.array([[next(c['phase1_pct']-c['phase0_pct'] for c in cells if c['cell']=='c%02dq%d'%(d,q)) for q in range(5)] for d in range(40)])
fig,ax=plt.subplots(figsize=(10,14.8));lim=math.ceil(abs(matrix).max()*10)/10;im=ax.imshow(matrix,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto');ax.grid(False)
ax.set_yticks(range(40),['c%02d %s'%(d['id'],d['name_zh']) for d in domains],fontsize=8);ax.set_xticks(range(5),['Q%d'%q for q in range(5)])
for i in range(40):
    for q in range(5):
        v=matrix[i,q];ax.text(q,i,f'{v:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>2.2 else '#243440')
fig.colorbar(im,ax=ax,shrink=.45,label='新主阶段 − 原配比 / 百分点');ax.set_title('同一domain的五档也在迁移：200桶权重差',fontsize=15,pad=15)
fig.text(.02,.008,'配置权重，不是库存比例或取整后的有效计数。红色减少，蓝色增加。接近零的格子保留+0.00/-0.00；精确值见CSV。',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.025,1,1]);save(fig,'cell_weight_shifts')
scenarios=read(A/'cursor_audit.json')['scenarios'];fig,axes=plt.subplots(1,2,figsize=(11,5.4),sharey=True)
for ax,K,title in zip(axes,[49152,12288],['保留K=49152，历史边界→108096','缩K到12288，历史边界=108000']):
    rr=[r for r in scenarios if r['new_block_size']==K];pos=np.arange(len(rr));forward=np.array([r['forward_cursor_tokens']/1e9 for r in rr]);back=np.array([r['backward_cursor_tokens']/1e9 for r in rr]);ax.bar(pos,forward,color='#c07a2c',label='向前游标');ax.bar(pos,back,bottom=forward,color='#224f77',label='向后游标')
    for i,r in enumerate(rr):ax.text(i,r['l1_cursor_difference_tokens']/1e9+2,f"{r['l1_cursor_difference_tokens']/1e9:.3f}B",ha='center',fontsize=10)
    ax.set_xticks(pos,[str(r['step']) for r in rr]);ax.set_xlabel('假设切换step');ax.set_title(title,fontsize=11);ax.set_ylim(0,136)
axes[0].set_ylabel('完整混合块的绝对游标差之和 / B tokens');axes[1].legend(fontsize=9)
fig.suptitle('固定全局token量，逐桶历史仍会改变',fontsize=16)
fig.text(.02,.008,'B=10亿tokens。堆叠显示向前与向后偏移的和；不是实际重复token数量。\n只核对完整混合块，未重放桶内shuffle、packing、restart取模或原始token ID。',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.08,1,.94]);save(fig,'cursor_differences')
print('Exported three additional figures, PNG + SVG.')
