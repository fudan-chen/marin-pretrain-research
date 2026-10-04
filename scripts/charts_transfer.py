# -*- coding: utf-8 -*-
"""Compare bucket and domain net redistribution on frozen candidate weights."""
import pathlib,json,os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
R=pathlib.Path(__file__).resolve().parents[1]
font=os.environ.get('MARIN_REPORT_FONT') or '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
font_manager.fontManager.addfont(font);plt.rcParams['font.family']=FontProperties(fname=font).get_name()
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'marin-transfer-v7','font.size':11,'savefig.dpi':180})
ps=json.loads((R/'analysis/transfer_data.json').read_text())['profiles'];fig,ax=plt.subplots(figsize=(11.8,5.6))
left=[p['comparisons']['nominal']['cross_domain_net_tokens']/1e9 for p in ps];right=[p['comparisons']['nominal']['within_domain_cancellation_tokens']/1e9 for p in ps]
y=list(range(4));ax.barh(y,left,color='#356f88',label='域净差所需重分配');ax.barh(y,right,left=left,color='#c48b55',label='域内相抵的重分配');ax.invert_yaxis()
ax.set_yticks(y,[p['id'][:4]+'…（%d桶变化）'%p['changed_phase_cells'] for p in ps]);ax.set_xlim(0,7100);ax.set_xlabel('名义剩余预算上的重分配 / B tokens；不是训练收益')
for i,(a,b) in enumerate(zip(left,right)):ax.text(a+b+65,i,'%.1f B'% (a+b),va='center',fontsize=10)
ax.set_title('域份额相近，也可能包含大量域内Q档迁移',fontsize=16,pad=14);ax.legend(loc='lower right',fontsize=10);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.text(.02,.02,'均相对996f；剩余名义预算13.125T + 3.75T。桶重分配 = Σ|桶曝光差|/2；域净差重分配 = Σ|域曝光差|/2。\n两者之差是域内相抵量；只是预算分解，不指定实际供体路径，不估计桶的因果贡献。',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.11,1,1]);fig.savefig(R/'assets/candidate_budget_decomposition.png',bbox_inches='tight')
p=R/'assets/candidate_budget_decomposition.svg';fig.savefig(p,bbox_inches='tight',metadata={'Date':None});p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');plt.close(fig)
