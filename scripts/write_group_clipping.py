"""Refresh the existing optimizer chapter with frozen genuine CPU controls."""
import pathlib,json,re
R=pathlib.Path(__file__).resolve().parents[1];d=json.loads((R/'analysis/group_clipping_cpu.json').read_text());a=d['original_AdamH_control'];p=R/'OPTIMIZER_GROUPS_ZH.md';s=p.read_text()
s=s.replace('探针用NumPy、平面参数树和恒等Adam替代依赖，只检查分组、专用衰减的附加项与状态转发。它没有运行真实Adam矩估计、Optax multi_transform、JAX分片或训练，不证明真实checkpoint可以无误恢复。','早期探针用NumPy、平面参数树和恒等Adam，只检查分组、衰减附加项与状态转发。V51增加真实Optax CPU分组裁剪对照和原AdamH模块检查；人工标签不证明真实模型分组，也没有执行完整Hero optimizer build、训练或checkpoint恢复。')
s=s.replace('这是源码结构与接口语义推断，本轮只检查三个调用位置，没有执行真实multi_transform验证。','早期这里只核对三个源码调用位置。V51用真实Optax 0.2.5 multi_transform执行人工标签对照，确认该组内归约语义；未执行完整Hero分组构建。')
s=s.replace('六份配置的max_grad_norm均为None，','六份历史配置及10月7日归档声明的max_grad_norm均为None，')
marker='\n## V51：真实CPU裁剪对照与状态验收\n';s=s.split(marker)[0]
s+=marker+f'''
控制实验用三个人工标签与梯度范数3、4、12，对应三条clip-only变换。标签按源码分组结构命名，不是实际Hero参数路径；真实Optax只在每个标签的active子树里计算其global norm。

|归约控制|三个组的输出范数|合并输出范数|结论范围|
|---|---|---:|---|
|不裁剪，identity分组|3、4、12|13|对应无clipping的API控制，不是完整训练更新|
|每组阈值1|1、1、1|{d['group_control']['group_clipped_combined_norm']:.6f}|每组受约束，合并范数可以超过1|
|全模型先裁剪到1|3/13、4/13、12/13|1|保留组间相对比例；不同于逐组裁剪|
|原AdamH，首步gradient阈值0.01|输入gradient范数{a['clipped_gradient_norm']:.6f}|参数update范数{a['first_parameter_update_norm']:.6f}|梯度范数界不自动成为参数update范数界；两者单位不同|
|原AdamH，首步clipped/unclipped|首次参数update差{a['first_update_difference_norm']:.3g}|moment差{a['first_moment_difference_norm']:.6f}|参数首次几乎相同，状态仍不同|
|同一后续非零gradient|后续update差{a['common_next_update_difference_norm']:.6f}|—|短期更新相似不能验收策略等价|

AdamH对Adam方向做归一化。在首步、epsilon影响很小的人工例子中，整个梯度乘正比例后，mu也同比缩放、nu按平方缩放，逐元素方向比例可能近乎抵消；但保存的mu/nu幅度已经改变。下一步固定同一参数输入、加入相同新梯度后，旧moment与新梯度的相对份额不同，差异重新进入方向。这个对照只在首步使用两种不同输入，后续参数与梯度输入共同；不是两条完整“持续开启/关闭裁剪”的训练曲线。lr=0.1、epsilon=1e−8和2×2矩阵均为人工控制，不外推到535B。

验收应保存原梯度的全模型/各组范数、每组clip系数与触发率、clipping后的梯度、moment/count、最终update/parameter比，并观察后续有效步。若microbatch采用不同有效分母，其梯度尺度也可能改变裁剪触发；应先固定归约口径，再比较裁剪策略。[梯度累积审计](GRADIENT_ACCUMULATION_ZH.md)的反例不能直接代填本次训练的触发频率。

当前公开归档max_grad_norm=None，危险/差异分支是迁移与配置变更的验收对象，不能作为已发生的Hero裁剪事故。没有执行真实create_mask、MuonH Newton–Schulz、完整multi_transform组链、调度、GPU/TPU或历史状态重放。

[10项CPU与源码检查](analysis/group_clipping_cpu.json)记录原值、人工标签、模块SHA和Optax函数来源SHA。[脚本](scripts/probe_group_clipping_cpu.py)使用真实Optax分组控制与未修改的AdamH模块；完整Hero build仍未知。使用V50的[CPU依赖](requirements-optimizer-cpu.txt)，入口为`make clipping-cpu CPU_PYTHON=/你的环境/bin/python`；通常report构建只读取冻结结果。
'''
p.write_text(s)
import matplotlib,numpy as np
matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='marin-group-clipping-20261007'
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,4));x=np.arange(3);width=.35;ax.bar(x-width/2,[1,1,1],width,label='Per-group clip to 1');ax.bar(x+width/2,[3/13,4/13,12/13],width,label='Whole-model clip to 1');ax.set_xticks(x,['muonh label','adamh label','adam label']);ax.set_ylabel('Post-clip gradient norm');ax.set_title('Real Optax CPU with artificial group labels\nLatest archived Hero recipe: max_grad_norm=None',fontsize=11);ax.legend(fontsize=9);fig.tight_layout();fig.savefig(R/'assets/group_clipping_cpu.png',dpi=150);fig.savefig(R/'assets/group_clipping_cpu.svg',metadata={'Date':None});plt.close(fig)
p=R/'assets/group_clipping_cpu.svg';s=p.read_text();ids=re.findall(r'id="([^"]+)"',s)
for old in sorted(set(ids),key=len,reverse=True):s=s.replace('id="'+old+'"','id="groupclip-'+old+'"').replace('#'+old+'"','#groupclip-'+old+'"').replace('#'+old+')','#groupclip-'+old+')')
p.write_text('\n'.join(l.rstrip() for l in s.splitlines())+'\n')
