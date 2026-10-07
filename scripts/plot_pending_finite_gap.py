"""Observed synthetic state health across three original train steps."""
import pathlib,json,html,hashlib
R=pathlib.Path(__file__).resolve().parents[1];j=json.loads((R/'analysis/pending_finite_gap_cpu.json').read_text())
names=list(j['cases']);labels=['有限 beta 参照','注入一个 NaN beta','注入一个 +Inf beta','有限 beta = [3e38]*3']
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="560" viewBox="0 0 1200 560" role="img" aria-labelledby="pending-finite-title pending-finite-desc">','<title id="pending-finite-title">有限loss可以跨过非有限路由状态</title>','<desc id="pending-finite-desc">人工CPU三专家模型，原训练闭包执行三步。所有loss有限；异常pending或中心化溢出可使下一步bias非有限。第三步状态重新有限但参数已偏离参照。</desc>','<rect width="1200" height="560" fill="#f8fafc"/>']
def t(x,y,v,size=17,color='#18344b'):s.append('<text x="%s" y="%s" font-family="system-ui,sans-serif" font-size="%s" fill="%s">%s</text>'%(x,y,size,color,html.escape(str(v))))
t(24,38,'loss 有限，不等于路由状态健康',26);t(24,67,'人工 beta 注入 · 原 train_step / setter / route · 非 Hero 事故或误差估计',16)
for x,v in [(24,'控制'),(280,'第 1 步：输出注入 beta'),(585,'第 2 步：使用该 beta'),(890,'第 3 步：再次有限')]:t(x,110,v,17)
for k,(name,label) in enumerate(zip(names,labels)):
 y=155+k*85;t(24,y+14,label,17)
 for i,row in enumerate(j['cases'][name]['stages']):
  x=270+i*305;s.append('<rect x="%s" y="%s" width="294" height="75" rx="6" fill="%s"/>'%(x,y-23,'#e6f1eb' if row['candidate_all_flags_pass'] else '#fff0df'))
  t(x+10,y,'loss %.6f'%row['loss'],16)
  flags=[row['pending_finite'],row['stored_bias_finite'],row['applied_bias_finite']]
  t(x+10,y+22,'P / stored / next: '+' / '.join('Y' if v else 'N' for v in flags),15)
  changed=row['params_w']!=j['cases']['finite']['stages'][i]['params_w']
  t(x+10,y+43,'参数已偏离参照' if changed else '参数与参照相同',15,'#a64c20' if changed else '#215f55')
t(24,518,'原 loss-only 检查接受全部 12 步；P = pending 输入有限性，stored / next = 两种 bias 视图有限性。',16)
t(24,545,'绿色仅表示三个有限性标志通过，不证明过去更新正确；图中数值是人工平方误差。',16);s.append('</svg>')
p=R/'assets/pending_finite_gap.svg';p.write_text('\n'.join(s)+'\n')
(R/'analysis/pending_finite_plot_binding.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256((R/'analysis/pending_finite_gap_cpu.json').read_bytes()).hexdigest(),'figure_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':names,'actual_Hero_numerical_event':None},indent=2)+'\n')
