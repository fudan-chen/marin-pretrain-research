(function(root){
 'use strict';
 const statuses=new Set(['pass','partial','fail','unassessed','na']);
 function normalizeAudit(raw,rules){
  if(!raw||typeof raw!=='object'||Array.isArray(raw)||raw.schema!=='marin-review/1')throw new Error('格式应为marin-review/1评审JSON。');
  if(raw.framework_version!==undefined&&raw.framework_version!=='1.1')throw new Error('规则版本不同，需要先人工核对；1.0可显式迁移并重新评R16。');
  if(raw.framework_version===undefined&&raw.judgments&&Object.keys(raw.judgments).length)throw new Error('已有判断缺规则版本，不能自动按1.1解释。');
  if(typeof raw.target!=='string'||!raw.target.trim()||raw.target.length>2000)throw new Error('请填写2000字符以内的评审目标。');
  if(!raw.judgments||typeof raw.judgments!=='object'||Array.isArray(raw.judgments))throw new Error('judgments应为规则对象。');
  const ids=new Set(rules.map(r=>r.id));
  for(const id of Object.keys(raw.judgments))if(!ids.has(id))throw new Error('未知规则编号：'+id);
  const result={schema:'marin-review/1',framework_version:'1.1',target:raw.target.trim(),scope:typeof raw.scope==='string'?raw.scope.slice(0,4000):'',judgments:{}};
  if(raw.migration!==undefined){
   if(!raw.migration||raw.migration.from_version!=='1.0'||raw.migration.to_version!=='1.1'||raw.migration.changed_rule!=='R16'||typeof raw.migration.prior_R16_json!=='string'||raw.migration.prior_R16_json.length>15000)throw new Error('迁移记录格式不正确。');
   result.migration={from_version:'1.0',to_version:'1.1',changed_rule:'R16',prior_R16_json:raw.migration.prior_R16_json};
  }
  for(const rule of rules){
   const row=raw.judgments[rule.id]||{status:'unassessed',evidence:''};
   if(!row||typeof row!=='object'||Array.isArray(row)||!statuses.has(row.status)||typeof row.evidence!=='string'||row.evidence.length>10000)throw new Error(rule.id+'的状态或证据格式不正确。');
   result.judgments[rule.id]={status:row.status,evidence:row.evidence};
  }
  return result;
 }
 function migrateAudit1_0(raw,rules){
  if(!raw||raw.schema!=='marin-review/1'||raw.framework_version!=='1.0'||!raw.judgments||typeof raw.judgments!=='object')throw new Error('显式迁移只接受1.0评审JSON。');
  // Validate the old record's field types before resetting the changed anchor.
  normalizeAudit({...raw,framework_version:'1.1'},rules);
  const next=JSON.parse(JSON.stringify(raw)),prior=next.judgments.R16||{status:'unassessed',evidence:''};
  next.framework_version='1.1';next.migration={from_version:'1.0',to_version:'1.1',changed_rule:'R16',prior_R16_json:JSON.stringify(prior)};
  next.judgments.R16={status:'unassessed',evidence:'由1.0迁移：请先声明图表准备或理解范围，再按1.1重新评R16。旧R16保留在migration.prior_R16_json。'};
  return normalizeAudit(next,rules);
 }
 function effectiveStatus(row){
  if(!row)return 'unassessed';
  if((row.status==='pass'||row.status==='na')&&!row.evidence.trim())return 'unassessed';
  return row.status;
 }
 function evaluateStage(stage,audit){
  const missing=[],conflicts=[],accepted=[];
  for(const id of stage.required){
   const row=audit.judgments[id],s=effectiveStatus(row);
   if(s==='fail')conflicts.push(id);else if(s==='pass')accepted.push(id);else missing.push(id);
  }
  return {id:stage.id,label:conflicts.length?'需修正当前主张':missing.length?'需补证据':'材料齐全（自评）',ready:missing.length===0&&conflicts.length===0,missing,conflicts,accepted};
 }
 function compareMetric(metric,reference){
  if(!['old','proportional'].includes(reference))throw new Error('参照只能是旧配比或库存比例。');
  if(!metric.means||!Array.isArray(metric.seeds)||metric.seeds.length!==3)throw new Error('指标缺少三组原值。');
  const baseline=metric.means[reference],selected=metric.means.selected;
  if(!Number.isFinite(baseline)||baseline<=0||!Number.isFinite(selected))throw new Error('BPB均值必须为有限数且参照为正。');
  const points=metric.seeds.map(r=>{
   if(!Number.isFinite(r[reference])||r[reference]<=0||!Number.isFinite(r.selected))throw new Error('BPB原值不正确。');
   return {seed:r.seed,changePct:100*(r.selected/r[reference]-1)};
  });
  return {baseline,selected,meanChangePct:100*(selected/baseline-1),points};
 }
 const api={normalizeAudit,migrateAudit1_0,effectiveStatus,evaluateStage,compareMetric};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;
 root.ReviewCore=api;
})(typeof globalThis!=='undefined'?globalThis:this);
