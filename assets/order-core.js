/* Integer count contracts only. Encoded loader weights are compiled separately in Python. */
(function(root){'use strict';
 const copy=x=>JSON.parse(JSON.stringify(x)),sum=xs=>xs.reduce((a,b)=>a+b,0);
 function partial(counts,begin,end){let cursor=0,out={};for(const k of Object.keys(counts).sort()){out[k]=Math.max(0,Math.min(cursor+counts[k],end)-Math.max(cursor,begin));cursor+=counts[k];}return out;}
 function compile(data,k,lock){
  if(typeof k!=='number'||!Number.isInteger(k)||k<0||k>data.locked_default.capacity.symmetric_max_k)throw Error('k必须是0至19的整数；不能裁剪越界输入');
  if(typeof lock!=='boolean')throw Error('必须明确是否锁住边缘块');
  const template=copy(lock?data.locked_default:data.unlocked_default),base=data.baseline_counts,K=template.block_size_sequences,cells=Object.keys(base[0]).sort();
  template.schema='marin-count-order-review/1';template.amplitude_k=k;template.export_scope='Counts only; use Python compiler and native probe to obtain checked loader weights';template.whole_window_count_match_by_construction=lock||k===0;
  const intervals=lock?[0,393216,442368,3194880,3981312]:[0,393216,3194880];
  for(const arm of template.arms){let early=copy(base[0]),late=copy(base[1]),u0=arm.sign*k*2,u1=arm.sign*k*7;
   early[template.receiver]+=u0;early[template.donor]-=u0;late[template.receiver]-=u1;late[template.donor]+=u1;
   const stageCounts=lock?[data.initial_counts,base[0],early,late,base[1]]:[data.initial_counts,early,late];
   if(stageCounts.some(c=>sum(Object.values(c))!==K||Object.values(c).some(n=>n<0)))throw Error('整数计数不守恒或出现负数');
   arm.stages=intervals.map((s,i)=>({sequence_begin:s,counts:copy(stageCounts[i])}));arm.interior_counts=[early,late];
   arm.interior_cumulative_difference_sequences=Object.fromEntries(cells.map(c=>[c,56*(early[c]-base[0][c])+16*(late[c]-base[1][c])]));
   arm.interior_l1_difference_sequences=sum(Object.values(arm.interior_cumulative_difference_sequences).map(Math.abs));
   arm.receiver_forward_sequences=56*u0;arm.receiver_early_delta_units=u0;arm.receiver_late_delta_units=-u1;
   const lead=partial(lock?base[0]:early,9216,K),tail=partial(lock?base[1]:late,0,43008),baselineLead=partial(base[0],9216,K),baselineTail=partial(base[1],0,43008);
   arm.unpermuted_debug_difference_sequences=Object.fromEntries(cells.map(c=>[c,arm.interior_cumulative_difference_sequences[c]+lead[c]+tail[c]-baselineLead[c]-baselineTail[c]]));
   arm.unpermuted_debug_l1_sequences=sum(Object.values(arm.unpermuted_debug_difference_sequences).map(Math.abs));
  }
  template.diagnostic_scope='Unpermuted-ID debugging, not historical shuffle or actual token reads';
  return template;
 }
 const api={compile,partial};root.OrderLedger=api;if(typeof module==='object'&&module.exports)module.exports=api;
})(globalThis);
