# -*- coding: utf-8 -*-
"""One figure explains where order perturbations are allowed in the restored window."""
import pathlib,json,os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
R=pathlib.Path(__file__).resolve().parents[1];font=os.environ.get('MARIN_REPORT_FONT') or '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
font_manager.fontManager.addfont(font);plt.rcParams['font.family']=FontProperties(fname=font).get_name()
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'order-v8','font.size':11,'savefig.dpi':180})
p=json.loads((R/'templates/order_edge_locked.json').read_text());fig,ax=plt.subplots(figsize=(12,5.5));colors=['#b9c2c8','#4f8095','#c48b55','#b9c2c8'];intervals=[(393,432),(432,3120),(3120,3888),(3888,3930)]
for y,a in enumerate(p['arms']):
    for (lo,hi),c in zip(intervals,colors):ax.barh(y,hi-lo,left=lo,color=c,height=.62,edgecolor='white')
    ax.text((432+3120)/2,y,'前段56整块：每块数学Q4 %d'%a['interior_counts'][0]['c39q4'],color='white',ha='center',va='center')
    ax.text((3120+3888)/2,y,'后段16整块：%d'%a['interior_counts'][1]['c39q4'],color='white',ha='center',va='center',fontsize=10)
ax.set_yticks([0,1,2],['基准','前置k10','后置k10']);ax.invert_yaxis();ax.set_xlim(300,4030);ax.set_ylim(2.8,-.9);ax.set_xticks([393,3120,3930]);ax.set_xlabel('更新索引；共同窗口 [393,3930)，预算不变')
ax.annotate('块8只读尾部39,936序列\n各组共同保留',xy=(411,-.33),xytext=(780,-.75),ha='center',fontsize=10,arrowprops={'arrowstyle':'->','color':'#555'})
ax.annotate('块81只读头部43,008序列\n输入与累计游标共同保留',xy=(3908,-.33),xytext=(3300,-.75),ha='center',fontsize=10,arrowprops={'arrowstyle':'->','color':'#555'})
ax.set_title('时序变化放进完整块，两端保持共同数据状态',fontsize=16,pad=16)
fig.text(.025,.02,'本报告的未执行草案：c39q4与c26q4交换，其他198桶不变。前段 +20 / 后段 −70 序列每块，56×20 + 16×(−70)=0。\n使用公开步号推导的窗口和V7未确认配方；逻辑索引匹配还要求相同历史、游标、shuffle key与底层数据映射。不是Marin实际读取图。',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.1,1,1]);fig.savefig(R/'assets/order_edge_ledger.png',bbox_inches='tight');f=R/'assets/order_edge_ledger.svg';fig.savefig(f,bbox_inches='tight',metadata={'Date':None});f.write_text('\n'.join(s.rstrip() for s in f.read_text().splitlines())+'\n');plt.close(fig)
