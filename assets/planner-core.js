/* Budget arithmetic only. No loss predictor and no optimizer of mixture quality. */
(function(root){
 'use strict';
 const sum=xs=>xs.reduce((s,x)=>s+x,0);
 function finite(x,label,min=0,max=Infinity){if(typeof x!=='number'||!Number.isFinite(x)||x<min||x>max)throw Error(label+'超出范围或不是有限数');return x;}
 function quantize(weights,K){
  const z=sum(weights);if(z<=0||weights.some(x=>!Number.isFinite(x)||x<0))throw Error('权重必须非负且总量为正');
  const counts=weights.map(x=>Math.floor(x/z*K));const recipient=counts.indexOf(Math.max(...counts));
  const remainder=K-sum(counts);counts[recipient]+=remainder;
  return {counts,effective:counts.map(x=>x/K),remainder,recipient,zero:weights.map((x,i)=>x>0&&counts[i]===0?i:-1).filter(x=>x>=0)};
 }
 function analyze(input){
  const B=finite(input.budgetB,'后续预算',Number.MIN_VALUE,1e9),f=finite(input.earlyFraction,'早期占比',.000001,.999999),K=finite(input.blockUnits,'每块采样单位',1,1e6);
  if(!Number.isInteger(K))throw Error('每块采样单位必须是整数');
  if(!Array.isArray(input.buckets)||!input.buckets.length||input.buckets.length>200)throw Error('需要1至200个桶');
  const seen=new Set();const rows=input.buckets.map((r,i)=>{
   if(typeof r.name!=='string'||!r.name.trim()||seen.has(r.name))throw Error('桶名为空或重复');seen.add(r.name);
   for(const k of ['stockB','historyB','weightPct','floorPct','capEpochs','earlyShiftPP'])if(typeof r[k]!=='number'||!Number.isFinite(r[k]))throw Error('桶 '+(i+1)+' 的 '+k+' 不是有限数');
   finite(r.stockB,'库存',Number.MIN_VALUE);finite(r.historyB,'历史曝光');finite(r.weightPct,'比例',0,100);finite(r.floorPct,'保底',0,100);finite(r.capEpochs,'曝光上限');finite(r.earlyShiftPP,'早期增幅',-100,100);
   return {...r,weight:r.weightPct/100,floor:r.floorPct/100,delta:r.earlyShiftPP/100,upper:Math.max(0,Math.min(1,(r.capEpochs*r.stockB-r.historyB)/B))};
  });
  if(Math.abs(sum(rows.map(r=>r.weight))-1)>1e-8)throw Error('基准比例必须合计100%');
  if(Math.abs(sum(rows.map(r=>r.delta)))>1e-8)throw Error('早期增幅必须合计0个百分点：加给一个桶的量要从其他桶扣回');
  const lowerSum=sum(rows.map(r=>r.floor)),upperSum=sum(rows.map(r=>r.upper));let constraints=[];
  if(lowerSum>1+1e-10)constraints.push('保底合计超过100%');
  if(upperSum<1-1e-10)constraints.push('库存与曝光上限不足以容纳全部后续预算');
  rows.forEach(r=>{if(r.floor>r.upper+1e-10)constraints.push(r.name+'的保底超过剩余曝光容量');if(r.historyB>r.capEpochs*r.stockB+1e-10)constraints.push(r.name+'的历史曝光已超过上限');});
  const baseline=rows.map(r=>r.weight),early=rows.map(r=>r.weight+r.delta),late=rows.map(r=>r.weight-f/(1-f)*r.delta);
  const scheduleValid=[...early,...late].every(x=>x>=-1e-12&&x<=1+1e-12);
  if(!scheduleValid)return {B,f,K,rows,constraints,capacityFeasible:constraints.length===0,scheduleValid:false,scheduleError:'曝光补偿后的阶段比例超出0%至100%；降低早期增幅或调整阶段长度。'};
  const clean=xs=>xs.map(x=>Math.max(0,Math.min(1,x))),e=clean(early),l=clean(late),qb=quantize(baseline,K),qe=quantize(e,K),ql=quantize(l,K);
  const scenarios={constant:baseline,matched:rows.map((_,i)=>f*e[i]+(1-f)*l[i]),swapped:rows.map((_,i)=>f*l[i]+(1-f)*e[i])};
  const effective={constant:qb.effective,matched:rows.map((_,i)=>f*qe.effective[i]+(1-f)*ql.effective[i]),swapped:rows.map((_,i)=>f*ql.effective[i]+(1-f)*qe.effective[i])};
  const results=rows.map((r,i)=>{
   const violations=[];if(e[i]<r.floor-1e-10||l[i]<r.floor-1e-10)violations.push('阶段比例低于保底');
   if(qe.effective[i]<r.floor-1e-10||ql.effective[i]<r.floor-1e-10)violations.push('取整后低于保底');
   const expectedB=r.historyB+B*scenarios.matched[i],effectiveB=r.historyB+B*effective.matched[i];
   if(expectedB>r.capEpochs*r.stockB+1e-10)violations.push('计划曝光超过上限');
   if(effectiveB>r.capEpochs*r.stockB+1e-10)violations.push('取整后的计划曝光超过上限');
   if((e[i]>0&&qe.counts[i]===0)||(l[i]>0&&ql.counts[i]===0))violations.push('至少一个阶段正权重取成零');
   return {...r,earlyPct:e[i]*100,latePct:l[i]*100,expectedB,effectiveB,epochs:expectedB/r.stockB,effectiveEpochs:effectiveB/r.stockB,quantizationDeltaB:effectiveB-expectedB,swappedMinusMatchedB:B*(scenarios.swapped[i]-scenarios.matched[i]),violations};
  });
  const l1=(a,b)=>B*sum(a.map((x,i)=>Math.abs(x-b[i])));
  return {B,f,K,rows:results,constraints,capacityFeasible:constraints.length===0,scheduleValid:true,scenarios,effective,quantization:{constant:qb,early:qe,late:ql},
          continuousMatchedVsConstantL1B:l1(scenarios.matched,scenarios.constant),continuousSwappedVsMatchedL1B:l1(scenarios.swapped,scenarios.matched),
          effectiveMatchedVsConstantL1B:l1(effective.matched,effective.constant),effectiveSwappedVsMatchedL1B:l1(effective.swapped,effective.matched),
          executableWithinConstraints:constraints.length===0&&results.every(r=>r.violations.length===0)};
 }
 const api={analyze,quantize};root.MixPlanner=api;if(typeof module==='object'&&module.exports)module.exports=api;
})(globalThis);
