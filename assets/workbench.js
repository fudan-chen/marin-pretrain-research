(function(){
 'use strict';
 const input=JSON.parse(document.getElementById('workbench-data').textContent),framework=JSON.parse(document.getElementById('rubric-data').textContent),C=ReviewCore;
 const q=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),pct=n=>(n>0?'+':'')+n.toFixed(3)+'%';
 const labels=framework.status_labels,key='marin-review-framework-v1.1';let selectedStage='P1',audit=C.normalizeAudit(framework.example_audit,framework.rules),storageMessage='';
 try{const saved=localStorage.getItem(key);if(saved)audit=C.normalizeAudit(JSON.parse(saved),framework.rules);}catch(e){storageMessage='本地记录未读取；可用JSON保存。';}
 function save(){
  let normalized;
  try{normalized=C.normalizeAudit(audit,framework.rules);}catch(e){storageMessage='目标尚未填写完整，当前编辑暂未保存。';return;}
  try{localStorage.setItem(key,JSON.stringify(normalized));storageMessage='已保存到本浏览器；不会上传。';}catch(e){storageMessage='本地存储不可用，请导出JSON。';}
 }
 q('#case-select').innerHTML=input.cases.map(c=>`<option value="${c.id}">${esc(c.label)}</option>`).join('');
 function renderCase(){
  const c=input.cases.find(c=>c.id===q('#case-select').value),names={observed:'公开观察',computed:'本地复算',mechanism:'机制解释',hypothesis:'待检验假设'};
  q('#case-reading').innerHTML=`<div class="mechanism-chain" role="group" aria-label="案例解释链">${c.nodes.map((n,i)=>`${i?'<span class="chain-arrow" aria-hidden="true">→</span>':''}<div class="mechanism-node ${n.kind}"><small>${names[n.kind]}</small><span>${esc(n.text)}</span></div>`).join('')}</div><p>${esc(c.reading)}</p><div class="interpretation-grid"><div><strong>当前证据支持</strong><p>${esc(c.allowed)}</p></div><div><strong>尚不能推出</strong><p>${esc(c.forbidden)}</p></div><div><strong>下一项验证</strong><p>${esc(c.next)}</p></div></div><p class="status">相关规则：${c.rules.map(id=>`<a href="#review-board" data-rule-jump="${id}">${id}</a>`).join(' · ')} · <a href="${esc(c.source)}">来源与完整解释</a></p>`;
 }
 q('#case-select').addEventListener('change',renderCase);renderCase();
 function plot(m,r){
  const v=C.compareMetric(m,r),bound=Math.max(.01,Math.ceil(Math.max(Math.abs(v.meanChangePct),...v.points.map(p=>Math.abs(p.changePct)))*1.15*100)/100),X=n=>160+140*n/bound;
  const marks=v.points.map((p,i)=>`<circle cx="${X(p.changePct)}" cy="${25+i*17}" r="4" fill="${p.changePct<0?'#286a70':'#ad662e'}"/><text x="6" y="${29+i*17}" font-size="9">s${p.seed}</text>`).join('');
  return {v,html:`<article class="metric-card"><h3>${esc(m.label)}</h3><p><strong>${pct(v.meanChangePct)}</strong> <span class="status">均值之比</span></p><svg viewBox="0 0 320 107" role="img" aria-label="${esc(m.label)}，三seed变化${esc(v.points.map(p=>pct(p.changePct)).join('、'))}。刻度范围正负${bound.toFixed(2)}%。"><line x1="20" y1="82" x2="300" y2="82" stroke="#a4afb5"/><line x1="160" y1="10" x2="160" y2="85" stroke="#a4afb5" stroke-dasharray="3 3"/>${marks}<rect x="${X(v.meanChangePct)-4}" y="72" width="8" height="8" fill="#225a83"/><text x="20" y="101" text-anchor="start" font-size="9">−${bound.toFixed(2)}%</text><text x="160" y="101" text-anchor="middle" font-size="9">0</text><text x="300" y="101" text-anchor="end" font-size="9">+${bound.toFixed(2)}%</text></svg><p class="status">${v.points.map(p=>'seed'+p.seed+' '+pct(p.changePct)).join('；')}</p></article>`};
 }
 function renderMetrics(){
  const r=q('#reference-select').value,all=input.metrics.map(m=>plot(m,r));
  q('#metric-cards').innerHTML=all.map(x=>x.html).join('');
  q('#metric-reading-table tbody').innerHTML=all.map((x,i)=>`<tr><td>${esc(input.metrics[i].label)}</td><td>${x.v.baseline.toFixed(6)}</td><td>${x.v.selected.toFixed(6)}</td><td>${pct(x.v.meanChangePct)}</td></tr>`).join('');
  q('#reference-reading').textContent=r==='proportional'?'相对库存比例基线：代码相关BPB收益更一致；通用文本额外均值改善很小，数学两组退步、一组改善。':'相对旧Harrier：通用文本与代码均值改善；GSM8K BPB三组都退步。这里的参照与模型范围必须一起写出。';
 }
 q('#reference-select').addEventListener('change',renderMetrics);renderMetrics();
 function statusText(row){return labels[C.effectiveStatus(row)];}
 function renderRules(){
  const filter=q('#rule-filter').value,term=q('#rule-search').value.trim().toLowerCase();
  const visible=framework.rules.filter(r=>{
   const s=C.effectiveStatus(audit.judgments[r.id]);return (filter==='all'||filter==='gaps'&&['partial','unassessed'].includes(s)||filter==='conflicts'&&s==='fail'||filter==='accepted'&&['pass','na'].includes(s))&&[r.id,r.title,r.area,r.question,r.why].join(' ').toLowerCase().includes(term);
  });
  q('#rule-cards').innerHTML=visible.map(r=>{const row=audit.judgments[r.id];return `<article class="rule-card" id="review-${r.id}" data-rule="${r.id}"><h3>${r.id} · ${esc(r.title)} <small>${esc(r.area)}</small></h3><p>${esc(r.question)}</p><details><summary>查看判断锚点与依据</summary><p>${esc(r.why)}</p><ul>${['pass','partial','fail'].map(s=>`<li><strong>${labels[s]}：</strong>${esc(r.anchors[s])}</li>`).join('')}</ul><p>${r.evidence_links.map(l=>`<a href="${esc(l.href)}">${esc(l.label)}</a>`).join(' · ')}</p></details><label>${r.id}证据状态 <select data-status="${r.id}" aria-label="${r.id}证据状态">${Object.entries(labels).map(([s,l])=>`<option value="${s}"${s===row.status?' selected':''}>${l}</option>`).join('')}</select></label><label>${r.id}证据或不适用理由 <textarea data-evidence="${r.id}" aria-label="${r.id}证据或不适用理由" maxlength="10000">${esc(row.evidence)}</textarea></label><p class="status rule-effective" id="effective-${r.id}">有效状态：${statusText(row)}</p></article>`;}).join('');
  q('#review-status').textContent=`显示${visible.length}/18条规则。${storageMessage||'修改后保存到本浏览器，可导出。'}`;
 }
 function renderPipeline(){
  const states=framework.pipeline.map(s=>C.evaluateStage(s,audit));
  q('#pipeline-steps').innerHTML=framework.pipeline.map((s,i)=>`<button class="pipeline-step${s.id===selectedStage?' selected':''}" data-stage="${s.id}" aria-pressed="${s.id===selectedStage}"><strong>${s.id} ${esc(s.title)}</strong><span>${esc(states[i].label)}</span></button>`).join('');
  const stage=framework.pipeline.find(s=>s.id===selectedStage),state=states.find(s=>s.id===selectedStage);
  q('#pipeline-detail').innerHTML=`<div class="pipeline-result ${state.ready?'ready':'pending'}"><h3>${stage.id} · ${esc(stage.title)}</h3><p><strong>${esc(state.label)}</strong> · ${esc(stage.question)}</p><p>所需规则：${stage.required.map(id=>`<a href="#review-board" data-rule-jump="${id}">${id}</a>`).join(' · ')}</p>${state.conflicts.length?`<p>需修正：${state.conflicts.join('、')}</p>`:''}${state.missing.length?`<p>待补证据：${state.missing.join('、')}</p>`:''}<p>产物：${stage.outputs.map(esc).join('；')}。</p><p>${esc(stage.next)}</p><p class="status">所选阶段的必备项不能用“不适用”代替。若项目不研究顺序，可有理由跳过P4，不能称该阶段已验证。“材料齐全”仅根据自评输入判定，实际内容仍需核验。</p></div>`;
 }
 function refresh(){q('#review-target').value=audit.target;renderRules();renderPipeline();}
 q('#review-target').addEventListener('input',()=>{audit.target=q('#review-target').value;save();q('#review-status').textContent=storageMessage;});
 q('#rule-filter').addEventListener('change',renderRules);q('#rule-search').addEventListener('input',renderRules);
 q('#rule-cards').addEventListener('change',e=>{const id=e.target.dataset.status;if(!id)return;audit.judgments[id].status=e.target.value;save();renderRules();renderPipeline();});
 q('#rule-cards').addEventListener('input',e=>{const id=e.target.dataset.evidence;if(!id)return;audit.judgments[id].evidence=e.target.value;save();q('#effective-'+id).textContent='有效状态：'+statusText(audit.judgments[id]);renderPipeline();});
 q('#pipeline-steps').addEventListener('click',e=>{const b=e.target.closest('[data-stage]');if(!b)return;selectedStage=b.dataset.stage;renderPipeline();});
 document.getElementById('workbench').addEventListener('click',e=>{const link=e.target.closest('[data-rule-jump]');if(!link)return;q('#rule-filter').value='all';q('#rule-search').value=link.dataset.ruleJump;renderRules();});
 q('#reset-review').addEventListener('click',()=>{audit=C.normalizeAudit(framework.example_audit,framework.rules);q('#rule-filter').value='all';q('#rule-search').value='';save();refresh();});
 q('#blank-review').addEventListener('click',()=>{audit=C.normalizeAudit({schema:'marin-review/1',target:'待填写自己的问题与结论范围',scope:'自己的计划，尚未核验。',judgments:{}},framework.rules);q('#rule-filter').value='all';q('#rule-search').value='';save();refresh();});
 function serialize(){audit.target=q('#review-target').value;return JSON.stringify(C.normalizeAudit(audit,framework.rules),null,2);}
 q('#copy-review-json').addEventListener('click',()=>{try{q('#review-json').value=serialize();q('#review-import-status').textContent='已填入当前评审，可复制。';}catch(e){q('#review-import-status').textContent=e.message;}});
 q('#import-review').addEventListener('click',()=>{try{const next=C.normalizeAudit(JSON.parse(q('#review-json').value),framework.rules);audit=next;save();refresh();q('#review-import-status').textContent='导入成功；当前评审已更新。';}catch(e){q('#review-import-status').textContent='导入未完成：'+e.message+' 当前判断保留。';}});
 q('#migrate-review').addEventListener('click',()=>{try{const next=C.migrateAudit1_0(JSON.parse(q('#review-json').value),framework.rules);audit=next;save();refresh();q('#review-import-status').textContent='1.0已显式迁移；原R16保留，当前R16未确认，须重评。旧版本缓存未改。';}catch(e){q('#review-import-status').textContent='迁移未完成：'+e.message+' 当前判断保留。';}});
 q('#export-review').addEventListener('click',()=>{try{const url=URL.createObjectURL(new Blob([serialize()],{type:'application/json;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='marin-证据评审.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){q('#review-status').textContent=e.message;}});
 refresh();
 // Opening a fragment inside a collapsed appendix should reveal its enclosing details.
 function revealHash(){const id=decodeURIComponent(location.hash.slice(1));const target=document.getElementById(id);if(!target)return;for(let p=target.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;}
 window.addEventListener('hashchange',revealHash);revealHash();
 let printState=[];
 window.addEventListener('beforeprint',()=>{printState=Array.from(document.querySelectorAll('.large-table-details')).map(d=>[d,d.open]);printState.forEach(([d])=>{d.open=true;});});
 window.addEventListener('afterprint',()=>{printState.forEach(([d,open])=>{d.open=open;});printState=[];});
})();
