(function(){
 'use strict';
 const section=document.getElementById('planner'),q=s=>section.querySelector(s),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const example=JSON.parse(document.getElementById('planner-example').textContent);let current;
 const fields=['name','stockB','historyB','weightPct','floorPct','capEpochs','earlyShiftPP'];
 const fieldLabels={name:'桶名',stockB:'库存/B',historyB:'历史抽样/B',weightPct:'基准比例/%',floorPct:'阶段保底/%',capEpochs:'曝光上限/轮',earlyShiftPP:'前段增幅/百分点'};
 function load(x){
  current=JSON.parse(JSON.stringify(x));q('#plan-budget').value=x.budgetB;q('#plan-fraction').value=x.earlyFraction*100;q('#plan-block').value=x.blockUnits;
  q('#plan-label').textContent=x.label||'自填数据';
  q('#plan-input-rows').innerHTML=x.buckets.map((r,i)=>'<tr>'+fields.map(k=>`<td><input data-row="${i}" data-key="${k}" aria-label="桶${i+1} ${fieldLabels[k]}" ${k==='name'?'type="text"':'type="number" step="any"'} value="${esc(r[k])}"></td>`).join('')+'</tr>').join('');
  calculate();
 }
 function read(){
  const x={label:current.label,budgetB:Number(q('#plan-budget').value),earlyFraction:Number(q('#plan-fraction').value)/100,blockUnits:Number(q('#plan-block').value),buckets:current.buckets.map(r=>({...r}))};
  for(const n of section.querySelectorAll('#plan-input-rows input')){
   if(!n.value.trim())throw Error('请填完所有字段，缺少历史统计时不要把空白当成零');
   x.buckets[Number(n.dataset.row)][n.dataset.key]=n.dataset.key==='name'?n.value:Number(n.value);
  }
  for(const s of ['#plan-budget','#plan-fraction','#plan-block'])if(!q(s).value.trim())throw Error('请填完预算、阶段占比与每块采样单位');
  return x;
 }
 function calculate(){
  try{
   const input=read(),r=MixPlanner.analyze(input);current=input;q('#plan-error').hidden=true;q('#plan-results').hidden=false;
   const warnings=r.constraints.length?r.constraints.join('；'):'总保底与剩余容量可同时满足';
   if(!r.scheduleValid){q('#plan-summary').textContent=warnings+'。'+r.scheduleError;q('#plan-output-rows').innerHTML='';q('#plan-bars').innerHTML='';q('#plan-quantization').textContent='';return;}
   const n=v=>v.toFixed(6),pct=v=>v.toFixed(3)+'%';
   q('#plan-summary').textContent=`${warnings}。补偿顺序与恒定配比的理论曝光差L1：${n(r.continuousMatchedVsConstantL1B)}B；直接交换两个阶段：${n(r.continuousSwappedVsMatchedL1B)}B。取整后，补偿顺序相对恒定配比仍差${n(r.effectiveMatchedVsConstantL1B)}B。${r.executableWithinConstraints?'当前方案通过这些预算约束。':'当前方案有未满足的约束，见下表。'}`;
   q('#plan-output-rows').innerHTML=r.rows.map(x=>`<tr><td>${esc(x.name)}</td><td>${pct(x.earlyPct)}</td><td>${pct(x.latePct)}</td><td>${x.expectedB.toFixed(4)}</td><td>${x.epochs.toFixed(4)}</td><td>${x.quantizationDeltaB.toFixed(6)}</td><td>${x.swappedMinusMatchedB.toFixed(4)}</td><td>${esc(x.violations.join('；')||'通过')}</td></tr>`).join('');
   q('#plan-bars').innerHTML=r.rows.map(x=>`<div class="plan-bar-row"><span>${esc(x.name)}</span><div class="plan-bar-track"><i class="early" style="width:${x.earlyPct}%" title="前段 ${pct(x.earlyPct)}"></i><i class="late" style="width:${x.latePct}%" title="后段 ${pct(x.latePct)}"></i></div><span>${pct(x.earlyPct)} / ${pct(x.latePct)}</span></div>`).join('');
   q('#plan-quantization').textContent=`每块${r.K}个采样单位，前段余数${r.quantization.early.remainder}分给${r.rows[r.quantization.early.recipient].name}，后段余数${r.quantization.late.remainder}分给${r.rows[r.quantization.late.recipient].name}。这里的阶段预算按连续token量计算；实际运行还需按完整混合块重算阶段边界与尾块。`;
  }catch(e){q('#plan-error').hidden=false;q('#plan-error').textContent=e.message;q('#plan-results').hidden=true;}
 }
 section.addEventListener('input',e=>{if(e.target.id!=='plan-json')calculate();});
 q('#plan-reset').addEventListener('click',()=>load(example));
 q('#plan-apply-json').addEventListener('click',()=>{try{const x=JSON.parse(q('#plan-json').value);MixPlanner.analyze(x);load(x);}catch(e){q('#plan-error').hidden=false;q('#plan-error').textContent='导入失败：'+e.message;}});
 q('#plan-show-json').addEventListener('click',()=>{try{q('#plan-json').value=JSON.stringify(read(),null,2);}catch(e){q('#plan-error').hidden=false;q('#plan-error').textContent=e.message;}});
 q('#plan-export').addEventListener('click',()=>{try{const input=read(),result=MixPlanner.analyze(input),url=URL.createObjectURL(new Blob([JSON.stringify({input,result,scope:'预算与曝光算术，不预测Loss；实际loader、token流、checkpoint需另行验证。'},null,2)],{type:'application/json;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='配比与顺序预算审计.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){q('#plan-error').hidden=false;q('#plan-error').textContent=e.message;}});
 load(example);
})();
