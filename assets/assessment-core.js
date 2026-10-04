(function(root){
 'use strict';
 const dims=['scope','mechanism','next'];
 const copy=x=>JSON.parse(JSON.stringify(x));
 const object=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
 function text(x,limit,label){if(typeof x!=='string'||x.length>limit)throw new Error(label+'格式或长度不正确。');return x;}
 function stamp(x,label){if(x===null)return null;if(typeof x!=='string'||!/^\d{4}-\d\d-\d\dT/.test(x)||!Number.isFinite(Date.parse(x)))throw new Error(label+'不是时间记录。');return x;}
 function blank(bank){
  return {schema:'marin-reader-assessment/1',protocol_version:bank.protocol_version,bank_sha256:bank.bank_sha256,
   record_kind:'local_unverified_reader_record',participant_label:'',cases:Object.fromEntries(bank.cases.map(c=>[c.id,{draft:'',reference_first_opened_at:null,revisions:[]}]))};
 }
 function normalizeReview(raw,c){
  if(raw===null)return null;
  if(!object(raw)||!object(raw.ratings)||!['not_checked','none','issues'].includes(raw.critical_status)||!Array.isArray(raw.critical_errors))throw new Error('评审字段不完整。');
  const ratings={};
  if(Object.keys(raw.ratings).some(d=>!dims.includes(d)))throw new Error('未知评分维度。');
  for(const d of dims){const v=raw.ratings[d];if(v!==null&&![0,1,2].includes(v))throw new Error('评分只能是未评或0/1/2。');ratings[d]=v;}
  const ids=new Set(c.critical_errors.map(e=>e.id));
  if(raw.critical_errors.some(e=>!ids.has(e))||new Set(raw.critical_errors).size!==raw.critical_errors.length)throw new Error('关键越界编号不正确。');
  if((raw.critical_status==='issues')!==(raw.critical_errors.length>0))throw new Error('关键越界状态与选项矛盾。');
  return {reviewer_label:text(raw.reviewer_label,120,'评审标记'),ratings,critical_status:raw.critical_status,critical_errors:raw.critical_errors.slice().sort(),
   evidence_note:text(raw.evidence_note,4000,'答案摘句与判断理由'),saved_at:stamp(raw.saved_at,'评审时间')};
 }
 function complete(r){return !!(r&&r.reviewer_label.trim()&&r.evidence_note.trim()&&r.saved_at!==null&&dims.every(d=>r.ratings[d]!==null)&&r.critical_status!=='not_checked');}
 function equivalent(a,b){return dims.every(d=>a.ratings[d]===b.ratings[d])&&a.critical_status===b.critical_status&&JSON.stringify(a.critical_errors)===JSON.stringify(b.critical_errors);}
 function normalize(raw,bank){
  if(!object(raw)||raw.schema!=='marin-reader-assessment/1'||raw.protocol_version!==bank.protocol_version||raw.bank_sha256!==bank.bank_sha256)throw new Error('题目版本或来源摘要不同；先人工迁移，不覆盖当前记录。');
  if(raw.record_kind!=='local_unverified_reader_record'||!object(raw.cases))throw new Error('只接收未核验的本地回答记录。');
  const out=blank(bank);out.participant_label=text(raw.participant_label,120,'答题者标记');
  const ids=new Set(bank.cases.map(c=>c.id));if(Object.keys(raw.cases).some(id=>!ids.has(id)))throw new Error('存在未知题号。');
  for(const c of bank.cases){
   const row=raw.cases[c.id];if(!object(row)||!Array.isArray(row.revisions)||row.revisions.length>30)throw new Error(c.id+'回答记录缺失或修订过多。');
   const next={draft:text(row.draft,12000,'回答草稿'),reference_first_opened_at:stamp(row.reference_first_opened_at,'参考展开时间'),revisions:[]};
   for(const [i,r] of row.revisions.entries()){
    if(!object(r)||r.revision_id!==c.id+'-r'+(i+1)||!object(r.reviews)||Object.keys(r.reviews).some(k=>!['A','B'].includes(k)))throw new Error(c.id+'修订序号或评审槽不正确。');
    const answer=text(r.answer,12000,'回答');if(!answer.trim())throw new Error('空白不能作为已作答记录。');
    if(!['not_opened_in_this_page','opened_before_save'].includes(r.reference_in_page_status))throw new Error('参考展开口径不正确。');
    if(r.reference_in_page_status==='opened_before_save'&&next.reference_first_opened_at===null)throw new Error('已看参考的标记缺少本页展开记录。');
    if(!Array.isArray(r.review_events)||r.review_events.length>100)throw new Error('评审历史缺失或超过100条。');
    const n={revision_id:r.revision_id,answer,saved_at:stamp(r.saved_at,'回答时间'),reference_in_page_status:r.reference_in_page_status,
     reviews:{A:normalizeReview(r.reviews.A,c),B:normalizeReview(r.reviews.B,c)},adjudication:normalizeReview(r.adjudication,c),review_events:[]};
    const last={A:null,B:null,adjudication:null};let arbitrationSeen=false;
    for(const e of r.review_events){
     if(!object(e)||!['A','B','adjudication'].includes(e.slot)||e.revision_id!==n.revision_id)throw new Error('评审历史未对应当前原文。');
     if(arbitrationSeen&&e.slot!=='adjudication')throw new Error('仲裁后不能改写原双评依据。');
     const v=normalizeReview(e.review,c);if(v===null)throw new Error('评审历史不可为空。');
     if(e.slot==='adjudication')arbitrationSeen=true;last[e.slot]=v;
     n.review_events.push({slot:e.slot,revision_id:e.revision_id,review:v});
    }
    if(JSON.stringify(last.A)!==JSON.stringify(n.reviews.A)||JSON.stringify(last.B)!==JSON.stringify(n.reviews.B)||JSON.stringify(last.adjudication)!==JSON.stringify(n.adjudication))throw new Error('当前评分与保留的评审历史不一致。');
    if(n.adjudication!==null){
     if(!complete(n.reviews.A)||!complete(n.reviews.B)||n.reviews.A.reviewer_label.trim()===n.reviews.B.reviewer_label.trim()||equivalent(n.reviews.A,n.reviews.B)||!complete(n.adjudication)||[n.reviews.A,n.reviews.B].some(x=>x.reviewer_label.trim()===n.adjudication.reviewer_label.trim()))throw new Error('仲裁需要两份不同标记、完整且有分歧的评审，以及另一标记的完整理由。');
    }
    next.revisions.push(n);
   }
   out.cases[c.id]=next;
  }
  return out;
 }
 function addAnswer(raw,bank,id,now){
  const out=normalize(raw,bank),row=out.cases[id];if(!row)throw new Error('未知题号。');
  if(!row.draft.trim())throw new Error('先写下自己的判断，再保存回答。');
  if(row.revisions.length>=30)throw new Error('当前题目达到30份修订上限，请另存记录。');
  row.revisions.push({revision_id:id+'-r'+(row.revisions.length+1),answer:row.draft,saved_at:stamp(now,'回答时间'),
   reference_in_page_status:row.reference_first_opened_at===null?'not_opened_in_this_page':'opened_before_save',reviews:{A:null,B:null},adjudication:null,review_events:[]});
  return normalize(out,bank);
 }
 function openReference(raw,bank,id,now){const out=normalize(raw,bank);if(!out.cases[id])throw new Error('未知题号。');out.cases[id].reference_first_opened_at??=stamp(now,'参考展开时间');return normalize(out,bank);}
 function putReview(raw,bank,id,revisionId,slot,review){
  const out=normalize(raw,bank),r=out.cases[id]?.revisions.find(x=>x.revision_id===revisionId);
  if(!r||!['A','B','adjudication'].includes(slot))throw new Error('先选择已经保存的回答修订。');
  if(r.adjudication&&slot!=='adjudication')throw new Error('该修订已有仲裁；另存回答修订后再评，不覆盖仲裁依据。');
  if(r.review_events.length>=100)throw new Error('本修订的评审历史达到100条，请另存回答修订。');
  if(slot==='adjudication')r.adjudication=review;else r.reviews[slot]=review;
  r.review_events.push({slot,revision_id:revisionId,review:copy(review)});
  return normalize(out,bank);
 }
 function summarizeRevision(r){
  if(!r)return {answered:false,review_status:'not_answered',critical_errors:[],disagreement_open:false,anchor_status:'not_assessed',claimed_distinct_reviewers:false};
  const all=Object.values(r.reviews).filter(Boolean),full=all.filter(complete),distinct=full.length===2&&full[0].reviewer_label.trim()!==full[1].reviewer_label.trim();
  const disagreement=distinct&&!equivalent(full[0],full[1]),adjudicated=!!r.adjudication;
  const effective=adjudicated?[r.adjudication]:all;
  const critical=[...new Set(effective.flatMap(x=>x.critical_errors))].sort();
  let reviewStatus=full.length<2?'reviews_pending':!distinct?'same_reviewer_label':disagreement&&!adjudicated?'disagreement_open':adjudicated?'manually_adjudicated':'anchor_ratings_agree';
  const settled=adjudicated||(distinct&&!disagreement),ratings=adjudicated?r.adjudication:full[0];
  const anchorStatus=critical.length?'critical_overreach':!settled?'not_settled':dims.every(d=>ratings.ratings[d]===2)?'local_anchors_met':'needs_revision';
  return {answered:true,revision_id:r.revision_id,review_status:reviewStatus,critical_errors:critical,disagreement_open:disagreement&&!adjudicated,
   anchor_status:anchorStatus,claimed_distinct_reviewers:distinct,adjudicated,external_identity_verified:false,external_comprehension_verified:false};
 }
 function summary(raw,bank){
  const out=normalize(raw,bank);return {scope:'Structural state of manual local anchor ratings; no semantic correctness or comprehension validity verified',
   cases:bank.cases.map(c=>({case_id:c.id,...summarizeRevision(out.cases[c.id].revisions.at(-1))})),
   overall_score:null,training_gate_decision:null,external_reader_study:null,same_question_revision_is_transfer_evidence:false};
 }
 function exportRecord(raw,bank){const out=normalize(raw,bank);return {...out,summary:summary(out,bank)};}
 const api={blank,normalize,normalizeReview,complete,equivalent,addAnswer,openReference,putReview,summarizeRevision,summary,exportRecord};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;root.AssessmentCore=api;
})(typeof globalThis!=='undefined'?globalThis:this);
