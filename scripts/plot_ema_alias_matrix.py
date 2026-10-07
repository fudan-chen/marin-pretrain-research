"""Render exact buffer-accounting observations, not memory peaks."""
import pathlib,json,html,hashlib
R=pathlib.Path(__file__).resolve().parents[1];j=json.loads((R/'analysis/ema_alias_matrix_cpu.json').read_text())
names=list(j['cases']);labels=['原初始化 + donation','仅复制 EMA + donation','全状态复制 + donation','原初始化，关闭 donation','原初始化 → 保存恢复 + donation']
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="440" viewBox="0 0 1200 440" role="img" aria-labelledby="ema-alias-title ema-alias-desc">','<title id="ema-alias-title">相同状态值，不同缓冲区关系</title>','<desc id="ema-alias-desc">人工CPU模型的五种输入值完全相同。共享EMA并donation失败；只复制EMA、全复制、关闭donation或保存恢复都成功且下一次完整输出相同。字节是输入叶的缓冲区记账，不是内存峰值。</desc>','<rect width="1200" height="440" fill="#f8fafc"/>']
def t(x,y,v,size=17):s.append('<text x="%s" y="%s" font-family="system-ui,sans-serif" font-size="%s" fill="#18344b">%s</text>'%(x,y,size,html.escape(str(v))))
t(24,37,'恢复通过，不能替代从零初始化验收',25);t(24,66,'真实原初始化 / 训练闭包 / 本地 OCDBT · 人工 CPU 模型 · 五种输入值相同',16)
for x,v in [(24,'构造路径'),(445,'逻辑叶字节'),(630,'唯一观测缓冲区字节'),(895,'共享组数'),(1050,'执行结果')]:t(x,108,v,16)
for i,(name,label) in enumerate(zip(names,labels)):
 r=j['cases'][name];a=r['input_allocation'];y=146+i*49;s.append('<rect x="18" y="%s" width="1160" height="43" rx="5" fill="%s"/>'%(y-26,'#ffe5de' if r['error'] else '#e6f1eb'))
 for x,v in [(24,label),(445,a['logical_leaf_bytes']),(630,a['unique_observed_buffer_bytes']),(895,len(a['alias_groups'])),(1050,'失败' if r['error'] else '成功')]:t(x,y,v)
t(24,412,'仅复制 EMA 保留参数 / 优化器原缓冲区；成功分支下一次完整状态相同。不是 GPU 或内存峰值测量。',16);s.append('</svg>')
p=R/'assets/ema_alias_matrix.svg';p.write_text('\n'.join(s)+'\n')
(R/'analysis/ema_alias_plot_binding.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256((R/'analysis/ema_alias_matrix_cpu.json').read_bytes()).hexdigest(),'figure_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':names,'actual_production_peak_memory':None},indent=2)+'\n')
