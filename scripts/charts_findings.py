# -*- coding: utf-8 -*-
"""Show all macro constituents and all observed seed0 candidate endpoints."""
import pathlib,csv,json,os
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
names={'4chan':'4chan','c4_100_domains':'C4 100域','c4_en':'C4英文','dolma-v1_5':'Dolma','dolma_100_programing_languages':'编程语言文本','dolma_100_subreddits':'Reddit 100社区','falcon-refinedweb':'Falcon RefinedWeb','gab':'Gab','m2d2_s2orc_unsplit':'S2ORC学术文本','m2d2_wikipedia_unsplit':'Wikipedia','manosphere_meta_sep':'Manosphere','mc4':'mC4','ptb':'PTB','redpajama':'RedPajama','twitterAAE_HELM_fixed':'Twitter AAE','wikitext_103':'WikiText-103'}
rs=list(csv.DictReader((A/'macro_contributions.csv').open()));rs.sort(key=lambda r:float(r['contribution_to_macro_bpb']))
values=[1000*float(r['contribution_to_macro_bpb']) for r in rs]
labels=[names[r['metric'].split('/')[2].removesuffix('-llama3')] for r in rs]
fig,ax=plt.subplots(figsize=(11.7,7.8));y=list(range(16))
ax.barh(y,values,color=['#286a70' if v<0 else '#bc773e' for v in values],height=.65,zorder=3);ax.set_yticks(y,labels);ax.invert_yaxis();ax.axvline(0,color='#555',lw=.9)
ax.set_xlim(-1.37,.82);ax.set_xlabel('对Paloma macro的贡献 / 10⁻³ BPB（子集绝对变化 ÷ 16；负值更好）')
for i,v in enumerate(values):ax.text(v+(.018 if v>=0 else -.018),i,'%+.3f'%v,ha='left' if v>=0 else 'right',va='center',fontsize=9)
ax.set_title('选中配比相对库存比例基线：收益与退步同时存在',fontsize=15,pad=13)
fig.text(.025,.016,'三组continuation seed均值。代码 + 学术贡献 −1.860 × 10⁻³，其余14项 +1.275 × 10⁻³，净值约 −0.586 × 10⁻³。\n16项等权分解；没有把整体paloma/bpb当成第17项。描述终点变化，不估计单个训练桶的因果贡献。',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.075,1,1]);save(fig,'macro_contribution_findings')

rs=list(csv.DictReader((A/'observed_pareto_frontiers.csv').open()));audit=json.loads((A/'findings_audit.json').read_text())
selected='h100-d512-mixprior-996f489106c7b922-seed0-from10pct-20260829';alternative=audit['counterexample_run']
fig,axes=plt.subplots(2,2,figsize=(12.8,10.4))
for column,(objective,key,label) in enumerate(zip(['paloma_humaneval','paloma_gsm8k'],['humaneval_bpb','gsm8k_bpb'],['HumanEval目标文本BPB','GSM8K目标文本BPB'])):
    group=[r for r in rs if r['objective']==objective];front=sorted([r for r in group if r['on_observed_frontier']=='True'],key=lambda r:float(r['paloma_bpb']))
    for row in range(2):
        ax=axes[row,column]
        ax.scatter([float(r['paloma_bpb']) for r in group],[float(r[key]) for r in group],s=13,alpha=.32,color='#6b869b',label='597个seed0终点',zorder=2)
        ax.plot([float(r['paloma_bpb']) for r in front],[float(r[key]) for r in front],color='#33786b',lw=1.4,marker='o',ms=4,label='观察前沿',zorder=3)
        for run,color,marker,text in [(selected,'#b96525','*','选中996f…'),(alternative,'#784b9a','D','替代197c…')]:
            r=next(r for r in group if r['run_name']==run);x=float(r['paloma_bpb']);v=float(r[key]);ax.scatter(x,v,color=color,marker=marker,s=140 if marker=='*' else 40,edgecolor='white',linewidth=.6,label=text,zorder=5)
            if row==1:ax.annotate(text,(x,v),xytext=(35,22 if marker=='*' else -22),textcoords='offset points',fontsize=9,color=color,arrowprops={'arrowstyle':'-','color':color,'lw':.8})
        ax.set_xlabel('Paloma macro BPB');ax.set_ylabel(label)
        if row==0:ax.set_title('全部候选；选中被%d个候选支配'%audit['pareto'][objective]['selected_seed0_dominator_count'],fontsize=11)
        else:
            ax.set_xlim(1.18,1.205);ax.set_ylim((.48,.57) if column==0 else (.70,.89));ax.set_title('局部放大（所有差结果仍显示在上排）',fontsize=11)
axes[0,0].legend(loc='upper left',fontsize=8)
fig.suptitle('观察前沿随评估目标变化；单seed的“最好”仍需复验',fontsize=15,y=.985)
fig.text(.02,.014,'固定registry，全597个mixprior seed0候选；每个点一个seed，不是候选多seed均值。两指标均越小越好。\n三指标联合前沿有38个候选；197c同时优于选中seed0，但其54项任务中31项更差，尚无独立重复确认。',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.06,1,.965]);save(fig,'observed_frontier_findings')
print('Exported two findings figures, PNG + SVG.')
