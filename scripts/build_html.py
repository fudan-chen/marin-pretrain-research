# -*- coding: utf-8 -*-
"""Build a file:// compatible report with embedded data and no remote assets."""
import pathlib,json,base64,html,re
import markdown
from bs4 import BeautifulSoup
ROOT=pathlib.Path(__file__).resolve().parents[1]
chapters=[('conclusions','第四轮：结论与判断','CONCLUSIONS_ZH.md'),('report','研究主报告','REPORT_ZH.md'),('data','数据配比与顺序','DATA_GUIDE_ZH.md'),('deep','第二轮：搜索、取舍与游标','DEEP_DIVE_ZH.md'),('practical','第三轮：删域、曝光与顺序','PRACTICAL_ZH.md'),('translation','#8435逐条中文解读','ISSUE_8435_ZH.md'),('operations','58条运行记录','OPERATIONS_ZH.md'),('provenance','来源与复核','PROVENANCE_ZH.md')]
sections=[];toc=[]
for slug,label,file in chapters:
    result=markdown.markdown((ROOT/file).read_text(),extensions=['tables','fenced_code','toc'])
    soup=BeautifulSoup(result,'html.parser')
    for h in soup.find_all(re.compile('^h[1-6]$')):
        old=h.get('id','heading');h['id']=slug+'-'+old
        if h.name in ['h2','h3']:toc.append((slug,label,h['id'],h.text,h.name))
    for table in soup.select('table'):
        wrap=soup.new_tag('div',attrs={'class':'table-wrap'});table.wrap(wrap)
    for a in soup.select('a[href]'):
        if a['href'].startswith('http'):a['target']='_blank';a['rel']='noopener'
    sections.append('<section class="chapter" id="%s">%s</section>'%(slug,str(soup)))
cells=json.loads((ROOT/'analysis/cell_weights.json').read_text());domains={d['id']:d['name_zh'] for d in json.loads((ROOT/'analysis/domain_weights_zh.json').read_text())}
for c in cells:c['domain_zh']=domains[c['domain_id']]
samples=[{'cell':s['cell'],'examples':[t[:300]+('…' if len(t)>300 else '') for t in s['examples']]} for s in json.loads((ROOT/'analysis/sample_texts.json').read_text())]
def data(j):return json.dumps(j,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
nav=''.join('<a class="major" href="#%s">%s</a>'%(s,html.escape(l)) for s,l,_ in chapters)
nav+='<a class="major" href="#planner">自己的配比预算工具</a><a class="major" href="#explorer">200桶查询</a><a class="major" href="#notes">我的研究笔记</a>'
details=''
for slug,label,_ in chapters:
    links=''.join('<a class="%s" href="#%s">%s</a>'%('minor' if kind=='h3' else '',ident,html.escape(text)) for s,_,ident,text,kind in toc if s==slug)
    details+='<details%s><summary>%s · 章节</summary>%s</details>'%(' open' if slug=='conclusions' else '',html.escape(label),links)
explorer='''<section class="chapter" id="explorer"><h1>200桶查询：比例、库存与计划曝光</h1><p>点击桶号查看公开样本片段。库存与计划抽样量使用不同分母；Cooldown尚未完成。</p><div class="filters"><label>搜索 <input id="bucket-search" placeholder="如 c13q4 / Agent / 数学"></label><label>语义域 <select id="bucket-domain"></select></label><label>质量档 <select id="bucket-quality"><option value="">全部Q档</option><option value="0">Q0</option><option value="1">Q1</option><option value="2">Q2</option><option value="3">Q3</option><option value="4">Q4</option></select></label><label>排序 <select id="bucket-sort"><option value="id">按桶号</option><option value="epochs">按曝光轮数降序</option><option value="delta">按主阶段增幅降序</option></select></label></div><p class="status" id="bucket-count" aria-live="polite"></p><div class="table-wrap bucket-scroll"><table><thead><tr><th>桶 / 样本</th><th>语义域</th><th>库存/B</th><th>原配比</th><th>新主阶段</th><th>Cooldown计划</th><th>全程计划/B</th><th>计划轮数</th></tr></thead><tbody id="bucket-rows"></tbody></table></div><div class="sample-panel" id="sample-panel" hidden></div><p><a href="analysis/cell_weights.csv" download>下载完整200桶CSV</a> · <a href="analysis/domain_weights.csv" download>下载40域CSV</a> · <a href="analysis/summary.json">查看计算摘要</a></p></section>'''
notes='''<section class="chapter notes" id="notes"><h1>我的研究笔记</h1><p class="status">内容仅保存在当前浏览器的本地存储，可导出为文本。更换浏览器、清除网站数据或改用独立HTML时可能需要重新导入。</p><textarea id="local-notes" aria-label="我的研究笔记" placeholder="记录你想验证的假设、具体证据、对自己项目的调整……"></textarea><p id="notes-status" class="status" aria-live="polite"></p><button id="export-notes" class="action">导出笔记</button></section>'''
template='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Marin 535B预训练研究 · 中文报告</title><style>CSS</style></head><body id="top"><div class="layout"><aside class="sidebar"><div class="brand">Marin 535B<br>预训练研究</div><div class="meta">公开快照：2026-10-04<br>重点：故障机制、Loss、数据顺序</div><input id="toc-search" aria-label="筛选章节目录" placeholder="筛选章节标题"><nav>NAV</nav>DETAILS</aside><main><header class="intro"><h1>从一次大规模预训练里，学会定位问题与调整数据</h1><p>逐条读工程记录，把改动、症状、原因和验证对应起来；重新计算配比和曲线，明确哪里仍缺少因果证据。</p><div class="facts"><span>主帖 + 27条评论</span><span>58条生产事件</span><span>40域 × 5质量档</span><span>7段生产run</span><span>934条配比观测</span></div><div class="actions"><a href="#report">从三个案例开始</a><a href="#data">直接看数据分析</a><a href="#deep">第二轮新增结论</a><a href="REPORT_ZH.md">Markdown主报告</a><a href="report_standalone.html" download>下载单文件HTML</a><button class="action" id="print-report">打印 / 保存PDF</button></div></header>SECTIONS EXPLORER NOTES<div class="footer">文字、图与查询器可离线阅读。原网页与来源链接需要网络。每个数值对应归档文件；本机未执行535B训练。</div></main></div><a class="top-link" href="#top">回到目录</a><script type="application/json" id="bucket-data">CELLS</script><script type="application/json" id="sample-data">SAMPLES</script><script>JS</script></body></html>'''
template=template.replace('SECTIONS EXPLORER NOTES','SECTIONS PLANNER EXPLORER NOTES')
template=template.replace('<script>JS</script>','<script type="application/json" id="planner-example">PLANEXAMPLE</script><script>PLANNERCORE</script><script>JS</script><script>PLANNERUI</script>')
template=template.replace('<a href="#deep">第二轮新增结论</a>','<a href="#deep">第二轮新增结论</a><a href="#practical">第三轮：如何设计自己的实验</a><a href="#planner">输入自己的预算</a>')
template=template.replace('<span>934条配比观测</span>','<span>934条配比观测</span><span>129个运行配置复核</span>')
template=template.replace('<a href="#report">从三个案例开始</a>','<a href="#conclusions">先读最新结论与判断</a><a href="#report">从三个案例开始</a>')
mapping={'CSS':(ROOT/'assets/report.css').read_text(),'NAV':nav,'DETAILS':details,'SECTIONS':'\n'.join(sections),'EXPLORER':explorer,'NOTES':notes,'CELLS':data(cells),'SAMPLES':data(samples),'JS':(ROOT/'assets/report.js').read_text(),
         'PLANNER':(ROOT/'assets/planner.html').read_text(),'PLANEXAMPLE':data(json.loads((ROOT/'analysis/planner_example.json').read_text())),
         'PLANNERCORE':(ROOT/'assets/planner-core.js').read_text(),'PLANNERUI':(ROOT/'assets/planner-ui.js').read_text()}
page=re.sub('|'.join(sorted(mapping,key=len,reverse=True)),lambda m:mapping[m[0]],template)
# Link existing markdown artifacts to their corresponding chapters in this combined report.
for slug,_,file in chapters:page=page.replace('href="'+file+'"','href="#'+slug+'"')
# Retain a downloadable markdown link in the header.
page=page.replace('<a href="#report">Markdown主报告</a>','<a href="REPORT_ZH.md" download>Markdown主报告</a>')
(ROOT/'index.html').write_text(page)
standalone=page
for p in (ROOT/'assets').glob('*.png'):
    standalone=standalone.replace('src="assets/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"')
(ROOT/'report_standalone.html').write_text(standalone)
print('HTML built:',len(page.encode()),'bytes; standalone:',len(standalone.encode()),'bytes')
