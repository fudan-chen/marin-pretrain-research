"""Accessible source-driven state/route table; synthetic CPU values only."""
import pathlib,json,html,hashlib
R=pathlib.Path(__file__).resolve().parents[1];j=json.loads((R/'analysis/pending_resume_step_cpu.json').read_text())
rows=j['cases'];names=['uninterrupted','full_restore','clear_pending','materialize_keep_pending','materialize_clear_pending'];labels=['不中断参照','完整恢复','只清空 pending','先写入 bias，保留 pending','先写入 bias，再清空 pending']
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="480" viewBox="0 0 1200 480" role="img" aria-labelledby="pending-resume-title pending-resume-desc">','<title id="pending-resume-title">原训练步的 pending 与恢复对照</title>','<desc id="pending-resume-desc">人工三专家 CPU 模型。完整恢复和保留 pending 的预写入等价；清空 pending 后原 setter 把 bias 重设为零，改变路由及更新。数值不是 Hero 误差。</desc>','<rect width="1200" height="480" fill="#f8fafc"/>']
def text(x,y,s,size=18,color='#18344b'):
 svg.append('<text x="%s" y="%s" font-family="system-ui,sans-serif" font-size="%s" fill="%s">%s</text>'%(x,y,size,color,html.escape(str(s))))
text(28,36,'恢复验收必须走到下一次原训练更新',25);text(28,65,'人工三专家 · 固定 logits · 原 QB setter / route / train_step · 真实 Adam 与本地 OCDBT',16)
for x,t in [(28,'恢复/操作方式'),(350,'下一步 bias'),(600,'所选专家 ID'),(795,'训练 loss'),(1010,'对参照更新')]:text(x,112,t,17)
for i,(name,label) in enumerate(zip(names,labels)):
 y=146+i*48;good=name in ['uninterrupted','full_restore','materialize_keep_pending'];r=rows[name];color='#e6f1eb' if good else '#fff0df'
 svg.append('<rect x="20" y="%s" width="1160" height="43" rx="5" fill="%s"/>'%(y-25,color))
 text(28,y,label,17);text(350,y,str(r['next_forward_bias'][0]),17);text(600,y,str(r['next_forward_ids'][0]),17);text(795,y,'%.6f'%r['train_loss'],17);text(1010,y,'一致' if good else '改变',17)
text(28,407,'零 pending 不是“不做更新”：原函数始终用 -beta 并中心化，替换 router bias。',18)
text(28,444,'另行记录：原初始化 EMA 共享缓冲区在本 CPU donation 路径报错；归档 Hero EMA 关闭。',16)
svg.append('</svg>');p=R/'assets/pending_resume_step.svg';p.write_text('\n'.join(svg)+'\n')
(R/'analysis/pending_resume_plot_binding.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256((R/'analysis/pending_resume_step_cpu.json').read_bytes()).hexdigest(),'figure_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':names,'actual_Hero_error_magnitude':None},indent=2)+'\n')
