(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.DecisionCore=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
 'use strict';
 const finite=n=>typeof n==='number'&&Number.isFinite(n);
 function validate(input,plan){
  if(!input||!Array.isArray(input.metrics)||!Array.isArray(input.candidates))throw Error('候选数据格式无效');
  const ids=input.metrics.map(m=>m.id);
  if(!plan||!input.baselines[plan.baseline])throw Error('参照未定义');
  if(!ids.includes(plan.primary))throw Error('主目标未定义');
  if(!plan.caps||typeof plan.caps!=='object'||Array.isArray(plan.caps))throw Error('退步限制格式无效');
  for(const [k,v] of Object.entries(plan.caps)){
   if(!ids.includes(k)||!finite(v)||Math.abs(v)>1000)throw Error('限制必须是已定义指标的有限百分数');
  }
  const b=input.baselines[plan.baseline].values;
  for(const m of ids)if(!finite(b[m])||b[m]<=0)throw Error('参照BPB必须为正且有限');
  if(!input.candidates.length)throw Error('没有候选');
  const unique=new Set();
  for(const r of input.candidates){
   if(!r.id||unique.has(r.id))throw Error('候选ID重复或缺失');unique.add(r.id);
   for(const m of ids)if(!r.values||!finite(r.values[m])||r.values[m]<=0)throw Error('候选指标缺失或无效，不能当作通过');
  }
  return ids;
 }
 function analyze(input,plan){
  const ids=validate(input,plan),b=input.baselines[plan.baseline].values;
  const rows=input.candidates.map(r=>{
   const changes=Object.fromEntries(ids.map(m=>[m,100*(r.values[m]/b[m]-1)]));
   const violations=Object.entries(plan.caps).filter(([m,cap])=>changes[m]>cap+1e-10).map(([metric,cap])=>({metric,cap,change:changes[metric]}));
   return {...r,changes,violations,eligible:violations.length===0};
  }).sort((a,b)=>a.values[plan.primary]-b.values[plan.primary]||a.id.localeCompare(b.id));
  let n=0;for(const r of rows)r.eligibleRank=r.eligible?++n:null;
  return {rows,eligibleCount:n,excludedCount:rows.length-n,best:rows.find(r=>r.eligible)||null,
          guardrailCount:Object.keys(plan.caps).length,plan:{baseline:plan.baseline,primary:plan.primary,caps:{...plan.caps}}};
 }
 function exportRecord(input,result){
  return {schema:'marin-observed-decision-record/1',status:'exploratory_reanalysis_not_preregistered',
   evidence_version:input.evidence_version,dataset_sha256:input.dataset_sha256,source_sha256:input.source_sha256,
   scope:input.scope,ranking_seed:0,metric_unit:'BPB',ranking_direction:'minimize',
   settings:result.plan,counts:{eligible:result.eligibleCount,excluded:result.excludedCount},
   observed_best_id:result.best?result.best.id:null,
   shortlist:result.rows.filter(r=>r.eligible).slice(0,10).map(r=>({id:r.id,rank:r.eligibleRank,values:r.values,changes:r.changes,observed_seeds:r.observed_seeds})),
   original_selector_reconstructed:false,independent_confirmation_executed:false,capability_accuracy_verified:false,
   next_confirmation:{status:'planned_not_executed',candidate_freeze_utc:null,checkpoint_content_digest:null,token_budget:null,
                      independent_eval_manifest:null,guardrails_frozen_before_results:false},
   limitations:input.unknown};
 }
 function toCSV(result,metrics){
  const cell=v=>'"'+String(v??'').replace(/"/g,'""')+'"';
  const header=['candidate','eligible','eligible_rank','violations',...metrics.map(m=>m.id+'_bpb'),...metrics.map(m=>m.id+'_delta_pct')];
  return [header,...result.rows.map(r=>[r.id,r.eligible,r.eligibleRank,r.violations.map(v=>v.metric).join(';'),...metrics.map(m=>r.values[m.id]),...metrics.map(m=>r.changes[m.id])])].map(row=>row.map(cell).join(',')).join('\n')+'\n';
 }
 return {analyze,exportRecord,toCSV};
});
