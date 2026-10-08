"""Split the preserved complete report into themed, single-chapter reading pages."""
from pathlib import Path
import ast,copy,hashlib,html,json,re
from urllib.parse import urlsplit,unquote
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parents[1];OUT=R/'reading';OUT.mkdir(exist_ok=True)
source=R/'report_standalone.html';soup=BeautifulSoup(source.read_text(),'html.parser')
sections={s['id']:s for s in soup.select('main > section[id]')}
groups=[
('start','01 · 全貌与阅读路线','先读综合判断，再看原训练记录。','synthesis-guide training-decisions-guide report engineering-map-guide translation operations operational-progress-guide'),
('data','02 · 数据配比与实验设计','从真实观测走到候选、供体、顺序与独立确认。','data mix-trajectory-guide decision-guide transfer-guide experiment-controls-guide confirmation-playbook-guide scale-guide conclusions deep practical repeat-exposure-guide order-guide batch-clock-guide'),
('input','03 · 数据管线与实际曝光','检查缓存、分词、打包、混合计数与恢复后的样本身份。','cache-guide dedup-guide quality-guide packing-fields-guide boundary-guide mixture-identity-guide mixture-range-guide assertion-contracts-guide weight-domain-guide run-code-guide'),
('loss','04 · Loss、梯度与数值精度','先诊断目标，再核对归约与实际反向路径。','loss-triage contracts-guide masked-numerics-guide loss-denominator-guide ce-label-bounds-guide ce-backward-path-guide ce-gradient-dtype-guide gradient-accumulation-guide zero-gradient-state-guide'),
('moe','05 · MoE、路由与性能','分清分配、权重、均衡估计、精度和真正的性能收益。','recent-moe-guide routing-guide qb-guide router-precision-guide short-conv-guide'),
('optimizer','06 · 优化器与学习率','分组、动量、范数约束与恢复后的学习率。','optimizer-guide optimizer-schedule-guide schedule-config-guide muon-geometry-guide muon-direction-guide adamh-state-guide state-guide'),
('eval','07 · 评估与曲线解释','核对评分格式、目标覆盖、分母、逐域变化和可比记录。','eval-metrics-guide eval-format-guide eval-target-alignment-guide eval-identity-guide eval-replay-guide eval-array-export-guide eval-weight-guide live-observation-guide live-oct7-guide'),
('recovery','08 · 故障、保存与恢复','从报警与状态健康走到异步提交、恢复及资源归还。','engineering-guide change-guide failure-boundaries-guide exception-provenance-guide observability-guide pending-resume-guide optional-resume-guide checkpoint-commit-guide checkpoint-memory-guide async-manager-guide checkpoint-failure-policy-guide gang-recovery-guide controller-recovery-guide'),
('method','09 · 规则、学习与证据','复用 rubrics 和管线；需要核查时再读审计与改进日志。','learning assessment-guide rubrics pipeline provenance delivery-audit-guide improvement'),
('tools','10 · 交互工具','候选筛选、顺序预算、工程案例、笔记与理解检查。','engineering-lab assessment-lab workbench decision-lab transfer-lab order-lab planner explorer notes')]
owner={sid:slug for slug,_,_,ids in groups for sid in ids.split()}
assert set(owner)==set(sections),(set(sections)-set(owner),set(owner)-set(sections))
ids_to_page={}
for sid,section in sections.items():
 for t in [section]+section.select('[id]'):ids_to_page.setdefault(t['id'],sid+'.html')
code_paths=set()
css='''body{margin:0;background:#f6f8fa;color:#203341;font:17px/1.85 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1000px;margin:0 auto;padding:30px 28px 100px}header{background:white;border-bottom:1px solid #dce3e8;padding:16px 28px;position:sticky;top:0;z-index:5}header nav{max-width:1100px;margin:auto;display:flex;gap:18px;flex-wrap:wrap}a{color:#176a80;text-underline-offset:3px}h1{font-size:30px;line-height:1.45}h2{margin-top:2.2em;font-size:24px}h3{font-size:20px}p{margin:1em 0}table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow:auto}th,td{border:1px solid #dce3e8;padding:10px;text-align:left;min-width:100px}th{background:#eaf1f5}pre{overflow:auto;background:#eaf0f4;padding:16px;font:13px/1.65 ui-monospace,monospace}code{overflow-wrap:anywhere}img,svg{max-width:100%;height:auto}details{margin:18px 0;border:1px solid #dce3e8;padding:12px;background:white}summary{cursor:pointer}.card{background:white;border:1px solid #dce3e8;border-radius:10px;padding:20px;margin:16px 0}.muted{color:#607282;font-size:14px}.chapter-toc{font-size:14px}.pager{display:flex;justify-content:space-between;gap:20px;margin-top:50px;border-top:1px solid #ccd7df;padding-top:20px}.code-line{display:block;white-space:pre}.code-line:target{background:#fff2ba}.ln{display:inline-block;width:4em;color:#74818b;text-decoration:none}.chapter{margin:0!important;padding:0!important}figure{margin:20px 0}figcaption{font-size:14px;color:#607282}@media(max-width:600px){main{padding:20px 16px 70px}h1{font-size:25px}header{position:static}body{font-size:16px}}'''
(OUT/'reading.css').write_text(css)
def title(section):return section.find(['h1','h2']).get_text(' ',strip=True)
def shell(t,body,extra_head='',scripts=''):
 return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(t)+' · Marin 预训练研究</title>'+extra_head+'<link rel="stylesheet" href="reading.css"></head><body><header><nav><a href="index.html">主题目录</a><a href="../report_standalone.html">完整报告（保留）</a><a href="code.html">源码与候选补丁</a><a href="tools.html">交互工具</a></nav></header><main>'+body+'</main>'+scripts+'</body></html>'
def rewrite(fragment):
 for tag in fragment.select('[href], [src]'):
  for attr in ['href','src']:
   u=tag.get(attr)
   if not u or u.startswith(('http:','https:','data:','mailto:','javascript:')):continue
   if tag.name in ['use','path'] or attr=='href' and tag.find_parent('svg'):continue
   if u.startswith('#'):
    target=ids_to_page.get(u[1:]);tag[attr]=(target+u if target else '../report_standalone.html'+u);continue
   parts=urlsplit(u);path=unquote(parts.path)
   if path.endswith('.md'):
    # Chapter source filenames are indexed from local Markdown headings below.
    match=md_to_sid.get(path)
    if match:tag[attr]=match+'.html'+('#'+parts.fragment if parts.fragment else '');continue
   local=R/path
   if attr=='href' and local.is_file() and local.suffix in {'.py','.patch','.diff'}:
    code_paths.add(path);tag[attr]='code-'+hashlib.sha256(path.encode()).hexdigest()[:16]+'.html'+('#L'+parts.fragment[1:] if re.fullmatch(r'L\d+',parts.fragment or '') else '');continue
   tag[attr]='../'+u
 return fragment
# Parse chapter metadata without executing the original report builder.
md_to_sid={}
tree=ast.parse((R/'scripts/build_html.py').read_text())
for node in ast.walk(tree):
 if isinstance(node,ast.Tuple):
  try: entry=ast.literal_eval(node)
  except (ValueError,TypeError): continue
  if len(entry)==3 and all(isinstance(x,str) for x in entry) and entry[0] in sections and entry[2].endswith('.md'):
   md_to_sid[entry[2]]=entry[0]
landing='<h1>Marin 预训练研究 · 按主题阅读</h1><p>每次只读一个问题。首次阅读先看全貌，再沿数据主线深入；遇到具体故障，再进入对应源码专题。</p><p class="muted">保留原完整报告与所有来源。人工反例、作者记录和真实训练结果的证据范围沿用原文；拆页不增加训练结论。</p><div class="card"><strong>第一次建议只读四篇</strong><ol>'+''.join('<li><a href="'+sid+'.html">'+html.escape(title(sections[sid]))+'</a></li>' for sid in ['synthesis-guide','report','data','training-decisions-guide'])+'</ol><p>每篇只记：问题 → 源码原因 → 处理方法 → 验证边界。</p></div>'
for slug,name,desc,ids in groups:
 members=ids.split();landing+='<div class="card"><h2><a href="theme-'+slug+'.html">'+name+'</a></h2><p>'+desc+'</p><span class="muted">'+str(len(members))+' 篇 / 工具</span></div>'
 body='<h1>'+name+'</h1><p>'+desc+'</p><ol>'+''.join('<li><a href="'+sid+'.html">'+html.escape(title(sections[sid]))+'</a></li>' for sid in members)+'</ol>'
 (OUT/('theme-'+slug+'.html')).write_text(shell(name,body))
 for i,sid in enumerate(members):
  if slug=='tools':continue
  section=rewrite(copy.deepcopy(sections[sid]))
  toc='<details class="chapter-toc"><summary>本篇目录</summary><ol>'+''.join('<li><a href="#'+h.get('id','')+'">'+html.escape(h.get_text(' ',strip=True))+'</a></li>' for h in section.select('h2[id],h3[id]'))+'</ol></details>'
  pager='<nav class="pager">'+('<a href="'+members[i-1]+'.html">← 上一篇</a>' if i else '<span></span>')+('<a href="'+members[i+1]+'.html">下一篇 →</a>' if i+1<len(members) else '<a href="index.html">返回主题目录</a>')+'</nav>'
  (OUT/(sid+'.html')).write_text(shell(title(section),'<p class="muted"><a href="theme-'+slug+'.html">'+name+'</a> / 本页只含一篇专题</p>'+toc+str(section)+pager))
(OUT/'index.html').write_text(shell('按主题阅读',landing))
# Preserve original interactive DOM, scripts and embedded data together; chapter pages stay script-free.
tooldoc=copy.deepcopy(soup)
for section in list(tooldoc.select('main > section[id]')):
 if owner[section['id']]!='tools':section.decompose()
for intro in tooldoc.select('main > .intro'):intro.decompose()
for aside in tooldoc.select('aside'):aside.decompose()
for script in tooldoc.find_all('script'):
 if script.string and script.get('type')!='application/json':
  text=str(script.string).replace("$('#toc-search').addEventListener", "$('#toc-search')?.addEventListener")
  # Script-generated chapter links must also leave the tools page.
  for chapter in sections:
   if owner[chapter]!='tools':
    text=text.replace("'#"+chapter+"'", "'"+chapter+".html#"+chapter+"'")
  script.string=text
for tag in tooldoc.select('main section'):rewrite(tag)
for link in tooldoc.select('link[href]'):
 if not urlsplit(link['href']).scheme:link['href']='../'+link['href']
for tag in tooldoc.select('body > a.top-link'):tag.decompose()
style=tooldoc.new_tag('style');style.string='.layout{display:block!important}main{max-width:1100px;margin:auto!important;padding:30px 24px}';tooldoc.head.append(style)
nav=BeautifulSoup('<nav style="padding:20px"><a href="index.html">← 返回主题目录</a> · <a href="../report_standalone.html">完整报告</a></nav>','html.parser').nav;tooldoc.body.insert(0,nav)
(OUT/'tools.html').write_text(str(tooldoc))
for sid in groups[-1][3].split():
 (OUT/(sid+'.html')).write_text(shell(title(sections[sid]),'<h1>'+html.escape(title(sections[sid]))+'</h1><p><a href="tools.html#'+sid+'">打开交互工具 →</a></p>'))
# Code viewers include original line numbers and raw downloads; archived sources remain unchanged.
for path in sorted(code_paths):
 text=(R/path).read_text();name='code-'+hashlib.sha256(path.encode()).hexdigest()[:16]+'.html'
 body='<h1>'+html.escape(Path(path).name)+'</h1><p class="muted">'+html.escape(path)+'</p><p>这是报告引用的源码或候选补丁；是否应用于上游或生产，以对应章节为准。</p><p><a href="../'+html.escape(path,quote=True)+'">查看 / 下载原文件</a></p><pre>'+''.join('<span class="code-line" id="L'+str(i)+'"><a class="ln" href="#L'+str(i)+'">'+str(i)+'</a>'+html.escape(line)+'</span>' for i,line in enumerate(text.splitlines(),1))+'</pre>'
 (OUT/name).write_text(shell(Path(path).name,body))
(OUT/'code.html').write_text(shell('源码与候选补丁','<h1>源码与候选补丁</h1><p>保留行号和原文件下载。候选补丁不自动等于上游已采用的修复。</p><ul>'+''.join('<li><a href="code-'+hashlib.sha256(p.encode()).hexdigest()[:16]+'.html">'+html.escape(p)+'</a></li>' for p in sorted(code_paths))+'</ul>'))
report={'complete_report_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'groups':[{'slug':g,'title':n,'sections':ids.split()} for g,n,_,ids in groups],'sections':len(sections),'code_viewers':len(code_paths),'chapter_pages_script_free':True,'interactive_tools_original_logic_with_optional_sidebar_guard':True}
(OUT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Built',len(sections),'sections,',len(groups),'themes,',len(code_paths),'code viewers; original complete report preserved')
