# -*- coding: utf-8 -*-
"""Build a file:// compatible report with embedded data and no remote assets."""
import pathlib,json,base64,html,re
import markdown
from bs4 import BeautifulSoup
ROOT=pathlib.Path(__file__).resolve().parents[1]
chapters=[('learning','理解文档与自测','LEARNING_GUIDE_ZH.md'),('rubrics','18条判断规则','RUBRICS_ZH.md'),('pipeline','七阶段决策管线','PIPELINE_ZH.md'),('improvement','持续改进队列','IMPROVEMENT_LOG_ZH.md'),('conclusions','第四轮：结论与判断','CONCLUSIONS_ZH.md'),('report','研究主报告','REPORT_ZH.md'),('data','数据配比与顺序','DATA_GUIDE_ZH.md'),('deep','第二轮：搜索、取舍与游标','DEEP_DIVE_ZH.md'),('practical','第三轮：删域、曝光与顺序','PRACTICAL_ZH.md'),('translation','#8435逐条中文解读','ISSUE_8435_ZH.md'),('operations','58条运行记录','OPERATIONS_ZH.md'),('provenance','来源与复核','PROVENANCE_ZH.md')]
chapters.insert(0,('decision-guide','候选选择与规则回放','DECISION_GUIDE_ZH.md'))
chapters.insert(0,('transfer-guide','预算分解与供体实验','TRANSFER_GUIDE_ZH.md'))
chapters.insert(0,('order-guide','顺序、边缘块与索引','ORDER_GUIDE_ZH.md'))
chapters.insert(0,('engineering-guide','工程排障、反证与验收','ENGINEERING_GUIDE_ZH.md'))
chapters.insert(0,('assessment-guide','理解检查、锚点与分歧','ASSESSMENT_GUIDE_ZH.md'))
chapters.insert(0,('scale-guide','跨规模比较、评估口径与确认','SCALE_TRANSFER_ZH.md'))
chapters.insert(0,('contracts-guide','源码接口、梯度与恢复验收','IMPLEMENTATION_CONTRACTS_ZH.md'))
chapters.insert(0,('state-guide','训练状态、精度与切换时刻','TRAIN_STATE_ZH.md'))
readings={x['file']:x for x in json.loads((ROOT/'analysis/figure_readings.json').read_text())}
sections=[];toc=[]
for slug,label,file in chapters:
    result=markdown.markdown((ROOT/file).read_text(),extensions=['tables','fenced_code','toc'])
    soup=BeautifulSoup(result,'html.parser')
    for h in soup.find_all(re.compile('^h[1-6]$')):
        old=h.get('id','heading');h['id']=slug+'-'+old
        if h.name in ['h2','h3']:toc.append((slug,label,h['id'],h.text,h.name))
    for table in soup.select('table'):
        wrap=soup.new_tag('div',attrs={'class':'table-wrap'});table.wrap(wrap)
        row_count=len(table.select('tbody tr'))
        if row_count>30:
            d=soup.new_tag('details',attrs={'class':'large-table-details'});wrap.wrap(d)
            summary=soup.new_tag('summary');summary.string='展开完整表：%d行，原值与来源保留'%row_count;d.insert(0,summary)
    for fig in soup.select('img[src]'):
        note=readings.get(fig['src'])
        if not note:continue
        fig['id']='figure-'+pathlib.Path(fig['src']).stem
        box=soup.new_tag('div',attrs={'class':'figure-reading'})
        for label,key in [('看什么','question'),('怎么读','reading'),('边界','boundary')]:
            p=soup.new_tag('p');strong=soup.new_tag('strong');strong.string=label+'：';p.append(strong);p.append(note[key]);box.append(p)
        p=soup.new_tag('p')
        for label,key in [('原值/计算入口','values'),('完整解释','chapter')]:
            a=soup.new_tag('a',href=note[key]);a.string=label;p.append(a)
        box.append(p);fig.parent.insert_after(box)
    for a in soup.select('a[href]'):
        if a['href'].startswith('http'):a['target']='_blank';a['rel']='noopener'
    sections.append('<section class="chapter" id="%s">%s</section>'%(slug,str(soup)))
cells=json.loads((ROOT/'analysis/cell_weights.json').read_text());domains={d['id']:d['name_zh'] for d in json.loads((ROOT/'analysis/domain_weights_zh.json').read_text())}
for c in cells:c['domain_zh']=domains[c['domain_id']]
samples=[{'cell':s['cell'],'examples':[t[:300]+('…' if len(t)>300 else '') for t in s['examples']]} for s in json.loads((ROOT/'analysis/sample_texts.json').read_text())]
def data(j):return json.dumps(j,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
nav='<a class="major" href="#workbench">理解与评审工作台</a>'+''.join('<a class="major" href="#%s">%s</a>'%(s,html.escape(l)) for s,l,_ in chapters)
nav+='<a class="major" href="#planner">自己的配比预算工具</a><a class="major" href="#explorer">200桶查询</a><a class="major" href="#notes">我的研究笔记</a>'
nav='<a class="major" href="#decision-lab">候选决策实验室</a>'+nav
nav='<a class="major" href="#transfer-lab">预算分解与交换草案</a>'+nav
nav='<a class="major" href="#order-lab">顺序整数账本</a>'+nav
nav='<a class="major" href="#engineering-lab">工程证据链与下一项检查</a><a class="major" href="#assessment-lab">理解检查与回答记录</a>'+nav
details=''
for slug,label,_ in chapters:
    links=''.join('<a class="%s" href="#%s">%s</a>'%('minor' if kind=='h3' else '',ident,html.escape(text)) for s,_,ident,text,kind in toc if s==slug)
    details+='<details%s><summary>%s · 章节</summary>%s</details>'%(' open' if slug=='learning' else '',html.escape(label),links)
explorer='''<section class="chapter" id="explorer"><h1>200桶查询：比例、库存与计划曝光</h1><p>点击桶号查看公开样本片段。库存与计划抽样量使用不同分母；Cooldown尚未完成。</p><div class="filters"><label>搜索 <input id="bucket-search" placeholder="如 c13q4 / Agent / 数学"></label><label>语义域 <select id="bucket-domain"></select></label><label>质量档 <select id="bucket-quality"><option value="">全部Q档</option><option value="0">Q0</option><option value="1">Q1</option><option value="2">Q2</option><option value="3">Q3</option><option value="4">Q4</option></select></label><label>排序 <select id="bucket-sort"><option value="id">按桶号</option><option value="epochs">按曝光轮数降序</option><option value="delta">按主阶段增幅降序</option></select></label></div><p class="status" id="bucket-count" aria-live="polite"></p><div class="table-wrap bucket-scroll"><table><thead><tr><th>桶 / 样本</th><th>语义域</th><th>库存/B</th><th>原配比</th><th>新主阶段</th><th>Cooldown计划</th><th>全程计划/B</th><th>计划轮数</th></tr></thead><tbody id="bucket-rows"></tbody></table></div><div class="sample-panel" id="sample-panel" hidden></div><p><a href="analysis/cell_weights.csv" download>下载完整200桶CSV</a> · <a href="analysis/domain_weights.csv" download>下载40域CSV</a> · <a href="analysis/summary.json">查看计算摘要</a></p></section>'''
notes='''<section class="chapter notes" id="notes"><h1>我的研究笔记</h1><p class="status">内容仅保存在当前浏览器的本地存储，可导出为文本。更换浏览器、清除网站数据或改用独立HTML时可能需要重新导入。</p><textarea id="local-notes" aria-label="我的研究笔记" placeholder="记录你想验证的假设、具体证据、对自己项目的调整……"></textarea><p id="notes-status" class="status" aria-live="polite"></p><button id="export-notes" class="action">导出笔记</button></section>'''
template='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Marin 535B预训练研究 · 中文报告</title><style>CSS</style></head><body id="top"><div class="layout"><aside class="sidebar"><div class="brand">Marin 535B<br>预训练研究</div><div class="meta">公开快照：2026-10-04<br>重点：故障机制、Loss、数据顺序</div><input id="toc-search" aria-label="筛选章节目录" placeholder="筛选章节标题"><nav>NAV</nav>DETAILS</aside><main><header class="intro"><h1>从一次大规模预训练里，学会定位问题与调整数据</h1><p>逐条读工程记录，把改动、症状、原因和验证对应起来；重新计算配比和曲线，明确哪里仍缺少因果证据。</p><div class="facts"><span>主帖 + 27条评论</span><span>58条生产事件</span><span>40域 × 5质量档</span><span>7段生产run</span><span>934条配比观测</span></div><div class="actions"><a href="#report">从三个案例开始</a><a href="#data">直接看数据分析</a><a href="#deep">第二轮新增结论</a><a href="REPORT_ZH.md">Markdown主报告</a><a href="report_standalone.html" download>下载单文件HTML</a><button class="action" id="print-report">打印 / 保存PDF</button></div></header>SECTIONS EXPLORER NOTES<div class="footer">文字、图与查询器可离线阅读。原网页与来源链接需要网络。每个数值对应归档文件；本机未执行535B训练。</div></main></div><a class="top-link" href="#top">回到目录</a><script type="application/json" id="bucket-data">CELLS</script><script type="application/json" id="sample-data">SAMPLES</script><script>JS</script></body></html>'''
template=template.replace('SECTIONS EXPLORER NOTES','SECTIONS PLANNER EXPLORER NOTES')
template=template.replace('<script>JS</script>','<script type="application/json" id="planner-example">PLANEXAMPLE</script><script>PLANNERCORE</script><script>JS</script><script>PLANNERUI</script>')
template=template.replace('<a href="#deep">第二轮新增结论</a>','<a href="#deep">第二轮新增结论</a><a href="#practical">第三轮：如何设计自己的实验</a><a href="#planner">输入自己的预算</a>')
template=template.replace('<span>934条配比观测</span>','<span>934条配比观测</span><span>129个运行配置复核</span>')
template=template.replace('<a href="#report">从三个案例开始</a>','<a href="#conclusions">先读最新结论与判断</a><a href="#report">从三个案例开始</a>')
template=template.replace('SECTIONS PLANNER','WORKBENCH DECISIONLAB TRANSFERLAB ORDERLAB SECTIONS PLANNER')
template=template.replace('<a href="#conclusions">先读最新结论与判断</a>','<a href="#workbench">从理解与评审工作台开始</a><a href="#learning">理解文档与自测</a><a href="#pipeline">决策管线</a><a href="#conclusions">最新研究结论</a>')
template=template.replace('<script>PLANNERUI</script>','<script>PLANNERUI</script><script type="application/json" id="workbench-data">WORKBENCHDATA</script><script type="application/json" id="rubric-data">RUBRICDATA</script><script>REVIEWCORE</script><script>WORKBENCHJS</script>')
template=template.replace('<script>WORKBENCHJS</script>','<script>WORKBENCHJS</script><script type="application/json" id="decision-data">DECISIONDATA</script><script>DECISIONCORE</script><script>DECISIONUI</script>')
template=template.replace('<a href="#workbench">从理解与评审工作台开始</a>','<a href="#decision-lab">比较候选与退步限制</a><a href="#workbench">从理解与评审工作台开始</a>')
template=template.replace('<a href="#decision-lab">比较候选与退步限制</a>','<a href="#transfer-lab">拆预算并设计供体实验</a><a href="#decision-lab">比较候选与退步限制</a>')
template=template.replace('<script>DECISIONUI</script>','<script>DECISIONUI</script><script type="application/json" id="transfer-data">TRANSFERDATA</script><script>TRANSFERUI</script>')
template=template.replace('<script>TRANSFERUI</script>','<script>TRANSFERUI</script><script type="application/json" id="order-data">ORDERDATA</script><script>ORDERCORE</script><script>ORDERUI</script>')
template=template.replace('<a href="#transfer-lab">拆预算并设计供体实验</a>','<a href="#order-lab">检查顺序与累计量</a><a href="#transfer-lab">拆预算并设计供体实验</a>')
template=template.replace('WORKBENCH DECISIONLAB','ENGINEERINGLAB ASSESSMENTLAB WORKBENCH DECISIONLAB')
template=template.replace('<script>ORDERUI</script>','<script>ORDERUI</script><script type="application/json" id="assessment-data">ASSESSMENTDATA</script><script>ASSESSMENTCORE</script><script>ASSESSMENTUI</script>')
template=template.replace('<a href="#order-lab">检查顺序与累计量</a>','<a href="#assessment-lab">保存自己的证据判断</a><a href="#order-lab">检查顺序与累计量</a>')
template=template.replace('<script>ASSESSMENTUI</script>','<script>ASSESSMENTUI</script><script type="application/json" id="engineering-data">ENGINEERINGDATA</script><script>ENGINEERINGUI</script>')
template=template.replace('公开快照：2026-10-04','训练快照：2026-10-04 · 工程核对：10-05')
template=template.replace('<a href="#assessment-lab">保存自己的证据判断</a>','<a href="#engineering-lab">核对工程解释与反证</a><a href="#assessment-lab">保存自己的证据判断</a>')
template=template.replace('<a href="#engineering-lab">核对工程解释与反证</a>', '<a href="#state-guide">最新：训练状态与切换时刻</a><a href="#contracts-guide">源码接口与恢复验收</a><a href="#scale-guide">配比历史与BPB聚合检查</a><a href="#engineering-lab">核对工程解释与反证</a>')
workbench=json.loads((ROOT/'analysis/workbench_data.json').read_text());rubric=json.loads((ROOT/'analysis/rubrics.json').read_text());chapter_paths={file:'#'+slug for slug,_,file in chapters}
for c in workbench['cases']:c['source']=chapter_paths.get(c['source'],c['source'])
for rule in rubric['rules']:
    for link in rule['evidence_links']:link['href']=chapter_paths.get(link['href'],link['href'])
mapping={'CSS':(ROOT/'assets/report.css').read_text()+'\n'+(ROOT/'assets/workbench.css').read_text(),'NAV':nav,'DETAILS':details,'SECTIONS':'\n'.join(sections),'EXPLORER':explorer,'NOTES':notes,'CELLS':data(cells),'SAMPLES':data(samples),'JS':(ROOT/'assets/report.js').read_text(),
         'WORKBENCH':(ROOT/'assets/workbench.html').read_text(),'WORKBENCHDATA':data(workbench),'RUBRICDATA':data(rubric),'REVIEWCORE':(ROOT/'assets/review-core.js').read_text(),'WORKBENCHJS':(ROOT/'assets/workbench.js').read_text(),
         'PLANNER':(ROOT/'assets/planner.html').read_text(),'PLANEXAMPLE':data(json.loads((ROOT/'analysis/planner_example.json').read_text())),
         'PLANNERCORE':(ROOT/'assets/planner-core.js').read_text(),'PLANNERUI':(ROOT/'assets/planner-ui.js').read_text()}
mapping['CSS']+='\n'+(ROOT/'assets/decision.css').read_text()
mapping.update({'DECISIONLAB':(ROOT/'assets/decision.html').read_text(),'DECISIONDATA':data(json.loads((ROOT/'analysis/decision_data.json').read_text())),
                'DECISIONCORE':(ROOT/'assets/decision-core.js').read_text(),'DECISIONUI':(ROOT/'assets/decision-ui.js').read_text()})
mapping['CSS']+='\n'+(ROOT/'assets/transfer.css').read_text()
mapping.update({'TRANSFERLAB':(ROOT/'assets/transfer.html').read_text(),'TRANSFERDATA':data(json.loads((ROOT/'analysis/transfer_data.json').read_text())),
                'TRANSFERUI':(ROOT/'assets/transfer-ui.js').read_text()})
mapping['CSS']+='\n'+(ROOT/'assets/order.css').read_text()
mapping.update({'ORDERLAB':(ROOT/'assets/order.html').read_text(),'ORDERDATA':data(json.loads((ROOT/'analysis/order_workbench_data.json').read_text())),
                'ORDERCORE':(ROOT/'assets/order-core.js').read_text(),'ORDERUI':(ROOT/'assets/order-ui.js').read_text()})
mapping['CSS']+='\n'+(ROOT/'assets/assessment.css').read_text()
mapping.update({'ASSESSMENTLAB':(ROOT/'assets/assessment.html').read_text(),'ASSESSMENTDATA':data(json.loads((ROOT/'analysis/assessment_data.json').read_text())),
                'ASSESSMENTCORE':(ROOT/'assets/assessment-core.js').read_text(),'ASSESSMENTUI':(ROOT/'assets/assessment-ui.js').read_text()})
mapping['CSS']+='\n'+(ROOT/'assets/engineering.css').read_text()
mapping.update({'ENGINEERINGLAB':(ROOT/'assets/engineering.html').read_text(),'ENGINEERINGDATA':data(json.loads((ROOT/'analysis/engineering_data.json').read_text())),'ENGINEERINGUI':(ROOT/'assets/engineering-ui.js').read_text()})
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
