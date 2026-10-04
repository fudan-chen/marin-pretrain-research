(function(){
 'use strict';
 const bank=JSON.parse(document.getElementById('assessment-data').textContent),C=AssessmentCore;
 const q=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const key='marin-reader-assessment-0.1-'+bank.bank_sha256,now=()=>new Date().toISOString();
 let record=C.blank(bank),message='',storageMessage='',selectedRevision='';
 const reviewNames={not_answered:'未作答',reviews_pending:'评审未齐',same_reviewer_label:'相同标记，不能作双评',disagreement_open:'两份评分有分歧',manually_adjudicated:'已记录人工仲裁',anchor_ratings_agree:'两份锚点评分一致'};
 const anchorNames={not_assessed:'未评',not_settled:'尚未确定',critical_overreach:'已记录关键越界',local_anchors_met:'本地锚点达到（未外部验证）',needs_revision:'按本地锚点需补理解'};
 try{const s=localStorage.getItem(key);if(s)record=C.normalize(JSON.parse(s),bank);}catch(e){storageMessage='保存记录未读取：'+e.message;}
 let storageBlocked=!!storageMessage;
 function save(){if(storageBlocked){q('#assessment-storage').textContent=storageMessage+' 原缓存保留，请导出当前新记录。';return;}
  try{localStorage.setItem(key,JSON.stringify(C.normalize(record,bank)));storageMessage='已保存到本浏览器；记录未经外部核验，不上传。';}
  catch(e){storageMessage='本地存储不可用，请导出JSON。';}q('#assessment-storage').textContent=storageMessage;
 }
 function showMessage(s){message=s;q('#assessment-message').textContent=s;}
 const currentCase=()=>bank.cases.find(c=>c.id===q('#assessment-case').value);
 const chapterLinks={'REPORT_ZH.md':'#report','CONCLUSIONS_ZH.md':'#conclusions','TRANSFER_GUIDE_ZH.md':'#transfer-guide','ORDER_GUIDE_ZH.md':'#order-guide'};
 function revision(){const row=record.cases[currentCase().id];return row.revisions.find(r=>r.revision_id===selectedRevision);}
 q('#assessment-case').innerHTML=bank.cases.map(c=>`<option value="${c.id}">${esc(c.id+' · '+c.title)}</option>`).join('');
 function renderOverview(){const s=C.summary(record,bank);q('#assessment-case-table tbody').innerHTML=s.cases.map((r,i)=>`<tr><td>${esc(bank.cases[i].id+' '+bank.cases[i].title)}</td><td>${r.revision_id||'未作答'}</td><td>${esc(reviewNames[r.review_status])}</td><td>${esc(anchorNames[r.anchor_status])}</td><td>${esc(r.critical_errors.join('、')||'无已记录项')}</td></tr>`).join('');}
 function renderReference(c){
  q('#assessment-reference-body').innerHTML=`<p><strong>允许的结论：</strong>${esc(c.allowed_claim)}</p><p>${esc(c.explanation)}</p><p><strong>追加证据：</strong>${esc(c.next_evidence)}</p><p><strong>关键越界：</strong>${c.critical_errors.map(e=>esc(e.text)).join('；')}</p><div class="table-wrap"><table><thead><tr><th>维度</th><th>2：达到本地锚点</th><th>1：部分</th><th>0：未达到/错误</th></tr></thead><tbody>${bank.dimensions.map(d=>`<tr><td>${esc(d.label)}</td>${[2,1,0].map(k=>`<td>${esc(c.anchors[d.id][k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div><p>${c.links.map(l=>`<a href="${esc(chapterLinks[l.href]||l.href)}">${esc(l.label)}</a>`).join(' · ')} · <a href="#figure-${esc(c.figure.split('/').at(-1).replace('.png',''))}">对应图</a></p>`;
 }
 function renderReviewer(){
  const c=currentCase(),r=revision(),slot=q('#assessment-review-slot').value,raw=r&&(slot==='adjudication'?r.adjudication:r.reviews[slot]);
  q('#assessment-reviewer').value=raw?.reviewer_label||'';
  q('#assessment-ratings').innerHTML=bank.dimensions.map(d=>`<div><label for="assessment-rating-${d.id}">${esc(d.label)}</label><select id="assessment-rating-${d.id}"><option value="">尚未评</option><option value="0">0：未达到 / 错误</option><option value="1">1：部分</option><option value="2">2：达到本地锚点</option></select></div>`).join('');
  for(const d of bank.dimensions)q('#assessment-rating-'+d.id).value=raw?.ratings[d.id]??'';
  q('#assessment-critical-status').value=raw?.critical_status||'not_checked';
  q('#assessment-critical-options').innerHTML=c.critical_errors.map(e=>`<label class="critical-choice"><input type="checkbox" data-critical-id="${e.id}"${raw?.critical_errors.includes(e.id)?' checked':''}>${esc(e.text)}</label>`).join('');
  q('#assessment-evidence').value=raw?.evidence_note||'';
 }
 function renderResult(){
  const r=revision(),s=C.summarizeRevision(r),rows=r?[['A',r.reviews.A],['B',r.reviews.B],['仲裁',r.adjudication]]:[];
  q('#assessment-review-result').classList.toggle('conflict',s.disagreement_open||s.critical_errors.length>0);
  q('#assessment-review-result').textContent=`${reviewNames[s.review_status]}；${anchorNames[s.anchor_status]}。${s.disagreement_open?'保留两份原判断，不平均；需要另一标记逐项说明仲裁理由。':'不同标记的真实身份、答案正确性与迁移理解均需人工核验。'} 本修订保留${r?.review_events.length||0}条评审事件，可在JSON中回查。`;
  q('#assessment-review-table tbody').innerHTML=rows.map(([name,v])=>`<tr><td>${name}</td><td>${esc(v?.reviewer_label||'未填写')}</td>${bank.dimensions.map(d=>`<td>${v?.ratings[d.id]??'未评'}</td>`).join('')}<td>${esc(!v?'未检查':v.critical_status==='not_checked'?'尚未检查':v.critical_status==='none'?'未发现列出越界':v.critical_errors.join('、'))}</td><td>${esc(v?.evidence_note||'未填写')}</td></tr>`).join('');
  const canArbitrate=!!r&&C.complete(r.reviews.A)&&C.complete(r.reviews.B)&&r.reviews.A.reviewer_label.trim()!==r.reviews.B.reviewer_label.trim()&&!C.equivalent(r.reviews.A,r.reviews.B);
  q('#assessment-review-slot option[value="adjudication"]').disabled=!canArbitrate;
 }
 function renderRevision(){const r=revision();if(!r)return;q('#assessment-saved-answer').textContent=r.answer;q('#assessment-exposure').textContent=`${r.revision_id} · ${r.reference_in_page_status==='opened_before_save'?'保存前已在本页展开参考':'保存时尚未在本页展开参考'}。这个记录不能排除其他材料使用。`;renderReviewer();renderResult();}
 function renderCase(){
  const c=currentCase(),row=record.cases[c.id];q('#assessment-title').textContent=c.title;q('#assessment-prompt').textContent=c.prompt;q('#assessment-rules').textContent='对应规则：'+c.rules.join('、');q('#assessment-answer').value=row.draft;
  renderReference(c);q('#assessment-reference').open=row.reference_first_opened_at!==null;
  q('#assessment-review-panel').hidden=row.revisions.length===0;
  if(row.revisions.length){selectedRevision=row.revisions.at(-1).revision_id;q('#assessment-revision').innerHTML=row.revisions.map(r=>`<option value="${r.revision_id}">${esc(r.revision_id)}</option>`).join('');q('#assessment-revision').value=selectedRevision;q('#assessment-review-slot').value='A';renderRevision();}
  renderOverview();
 }
 q('#assessment-person').value=record.participant_label;
 q('#assessment-person').addEventListener('input',()=>{record.participant_label=q('#assessment-person').value;save();});
 q('#assessment-answer').addEventListener('input',()=>{record.cases[currentCase().id].draft=q('#assessment-answer').value;save();});
 q('#assessment-case').addEventListener('change',()=>{showMessage('');renderCase();});
 q('#assessment-save-answer').addEventListener('click',()=>{try{record=C.addAnswer(record,bank,currentCase().id,now());save();renderCase();showMessage('已保存新修订；以前的回答与评分保留，当前修订尚未评。');}catch(e){showMessage(e.message);}});
 q('#assessment-reference').addEventListener('toggle',()=>{if(q('#assessment-reference').open&&record.cases[currentCase().id].reference_first_opened_at===null){record=C.openReference(record,bank,currentCase().id,now());save();}});
 q('#assessment-revision').addEventListener('change',()=>{selectedRevision=q('#assessment-revision').value;q('#assessment-review-slot').value='A';renderRevision();});
 q('#assessment-review-slot').addEventListener('change',renderReviewer);
 q('#assessment-save-review').addEventListener('click',()=>{
  try{const review={reviewer_label:q('#assessment-reviewer').value,ratings:Object.fromEntries(bank.dimensions.map(d=>[d.id,q('#assessment-rating-'+d.id).value===''?null:Number(q('#assessment-rating-'+d.id).value)])),critical_status:q('#assessment-critical-status').value,critical_errors:Array.from(document.querySelectorAll('#assessment-critical-options input:checked')).map(e=>e.dataset.criticalId),evidence_note:q('#assessment-evidence').value,saved_at:now()};
   record=C.putReview(record,bank,currentCase().id,selectedRevision,q('#assessment-review-slot').value,review);save();renderResult();renderOverview();showMessage(C.complete(review)?'人工评分已记录；工具未核验答案真伪或评审身份。':'已保存评审草稿；维度、越界检查、标记或依据仍未齐。');
  }catch(e){showMessage('评审未保存：'+e.message+' 原记录保留。');}
 });
 function serialize(){return JSON.stringify(C.exportRecord(record,bank),null,2);}
 q('#assessment-copy-json').addEventListener('click',()=>{q('#assessment-json').value=serialize();showMessage('当前记录已填入JSON框，可复制。');});
 q('#assessment-export').addEventListener('click',()=>{const blob=new Blob([serialize()],{type:'application/json;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='marin-理解检查-未核验记录.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
 q('#assessment-import').addEventListener('click',()=>{try{const next=C.normalize(JSON.parse(q('#assessment-json').value),bank);record=next;save();q('#assessment-person').value=record.participant_label;renderCase();showMessage('导入成功：只检查格式与版本，内容仍未核验。');}catch(e){showMessage('导入未完成：'+e.message+' 当前记录保留。');}});
 renderCase();q('#assessment-storage').textContent=storageMessage||'尚未保存。此题库使用独立缓存，不改变研究笔记和实验评审。';
})();
