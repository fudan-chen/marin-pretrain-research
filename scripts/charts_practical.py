# -*- coding: utf-8 -*-
"""Render all 80 domain interventions and the four math exposure endpoints."""
import pathlib,csv,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
R=pathlib.Path(__file__).resolve().parents[1];A=R/'analysis';O=R/'assets'
font=os.environ.get('MARIN_REPORT_FONT') or next((p for p in ['/System/Library/Fonts/Supplemental/Arial Unicode.ttf','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'] if pathlib.Path(p).exists()),None)
if not font:raise RuntimeError('Set MARIN_REPORT_FONT to a Chinese font.')
font_manager.fontManager.addfont(font);plt.rcParams['font.family']=FontProperties(fname=font).get_name()
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'font.size':10,'figure.facecolor':'white','axes.unicode_minus':False,'savefig.dpi':180,'svg.fonttype':'path','svg.hashsalt':'marin-535b-2026-10-04'})
def save(fig,name):
    fig.savefig(O/(name+'.png'),bbox_inches='tight');p=O/(name+'.svg');fig.savefig(p,bbox_inches='tight',metadata={'Date':None});p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');plt.close(fig)
rs=list(csv.DictReader((A/'domain_ablation_audit.csv').open()));groups=['semantic_domain_ablation','proportional_domain_ablation'];y=np.arange(40)
fig,axes=plt.subplots(1,2,figsize=(12,14.7),sharey=True)
for ax,g,title in zip(axes,groups,['Paloma macro BPB','GSM8K文本BPB']):
    key='paloma_change_pct' if g==groups[0] else 'gsm8k_change_pct'
    for i,(family,color,label) in enumerate(zip(groups,['#225a83','#c47f2f'],['语义配比：删除并分给其他域','比例配比：删除并分给其他域'])):
        rows=sorted([r for r in rs if r['group']==family],key=lambda r:int(r['domain']));ax.scatter([float(r[key]) for r in rows],y+(i-.5)*.17,color=color,s=24,label=label,zorder=3)
    ax.axvline(0,color='#666',lw=.8);ax.set_xlabel('相对各组参照BPB变化 / %（负值更好）');ax.set_title(title,fontsize=13)
labels=['c%02d %s'%(int(r['domain']),r['domain_zh']) for r in sorted([r for r in rs if r['group']==groups[0]],key=lambda r:int(r['domain']))]
axes[0].set_yticks(y,labels,fontsize=8);axes[0].invert_yaxis();axes[1].legend(loc='lower right',fontsize=8)
fig.suptitle('删掉同一个领域，通用文本与数学文本的方向可以相反',fontsize=16)
fig.text(.02,.008,'80次删域干预，各一个seed0。每组使用自己的参照：broad-technical-capped / proportional seed0。\n不是删域的独立贡献估计；包含预算重分配。记录配置已核对，原运行代码SHA与token流未核对。GSM8K这里不是答题准确率。',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.045,1,.96]);save(fig,'domain_ablation_response')
rs=list(csv.DictReader((A/'math_q4_response.csv').open()));x=[float(r['nominal_remaining_c39q4_epochs']) for r in rs];fig,ax=plt.subplots(figsize=(10.5,5.6))
for k,color,label in [('paloma_bpb','#225a83','Paloma macro'),('gsm8k_bpb','#c47f2f','GSM8K文本BPB'),('humaneval_bpb','#3b7769','HumanEval文本BPB')]:
    b=float(rs[0][k]);ax.plot(x,[100*(float(r[k])/b-1) for r in rs],marker='o',color=color,label=label,lw=1.4)
ax.axhline(0,color='#777',lw=.8);ax.set_xticks(x,['基线\n%.3f'%x[0]]+['%s标签\n%.3f'%(label,v) for label,v in zip(['4x','6x','8x'],x[1:])]);ax.set_xlabel('恢复后c39q4的名义模拟曝光/轮；共同历史不包含在横轴中');ax.set_ylabel('相对seed0基线的BPB变化 / %');ax.set_title('增加数学Q4，结果并非随曝光量单调改善',fontsize=15,pad=14);ax.legend()
fig.text(.02,.008,'每个点只有一个seed；连线仅帮助读数，不是拟合或预测。“4x/6x/8x”是运行标签，不等于权重倍数。\n按registry 13.125T+3.75T预算计算；其他桶的权重也减少，因此不能解释成数学Q4自身的纯效应。',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.09,1,1]);save(fig,'math_q4_response')
print('Exported two practical figures, PNG + SVG.')
