(function(){
 'use strict';
 const input=JSON.parse(document.getElementById('decision-data').textContent),C=DecisionCore,q=s=>document.querySelector(s);
 const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),pct=n=>(n>0?'+':'')+n.toFixed(3)+'%';
 const label=id=>input.metrics.find(m=>m.id===id).label;let result=null;
 q('#decision-primary').innerHTML=input.metrics.slice(0,3).map(m=>`<option value="${m.id}">${esc(m.label)}</option>`).join('');
 q('#decision-caps').innerHTML=input.metrics.map(m=>`<div class="cap-row"><label for="cap-on-${m.id}"><input type="checkbox" id="cap-on-${m.id}">${esc(m.label)}限制</label><div class="cap-value"><label for="cap-value-${m.id}">最大相对变化（%）</label><input type="number" id="cap-value-${m.id}" value="0" min="-1000" max="1000" step="0.1" disabled></div></div>`).join('');
 function plan(){
  const caps={};for(const m of input.metrics)if(q('#cap-on-'+m.id).checked){const s=q('#cap-value-'+m.id).value.trim();if(!s)throw Error(label(m.id)+'限制为空');caps[m.id]=Number(s);}
  return {baseline:q('#decision-baseline').value,primary:q('#decision-primary').value,caps};
 }
 function drawChart(){
  const rows=result.rows,xs=rows.map(r=>r.changes.paloma),ys=rows.map(r=>r.changes.humaneval);
  function bounds(v,minPad=.01){const lo=Math.min(0,...v),hi=Math.max(0,...v),p=Math.max((hi-lo)*.07,minPad);return [lo-p,hi+p];}
  const zoom=q('#decision-chart-range').value==='focus',focusRows=rows.filter(r=>r.role!=='candidate');
  const [xl,xh]=bounds(zoom?focusRows.map(r=>r.changes.paloma):xs,zoom ? .35 : .01),[yl,yh]=bounds(zoom?focusRows.map(r=>r.changes.humaneval):ys,zoom?5:.01),X=v=>76+(v-xl)/(xh-xl)*620,Y=v=>32+(yh-v)/(yh-yl)*330;
  const visible=rows.filter(r=>r.changes.paloma>=xl&&r.changes.paloma<=xh&&r.changes.humaneval>=yl&&r.changes.humaneval<=yh);
  const marks=visible.filter(r=>r.role==='candidate').map(r=>`<circle cx="${X(r.changes.paloma)}" cy="${Y(r.changes.humaneval)}" r="2.5" fill="${r.eligible?'#246f8d':'#adb5bb'}" opacity=".65"><title>${esc(r.id)}；Paloma ${pct(r.changes.paloma)}；HumanEval ${pct(r.changes.humaneval)}；GSM8K ${pct(r.changes.gsm8k)}；${r.eligible?'符合当前限制':'排除：'+r.violations.map(v=>label(v.metric)).join('、')}</title></circle>`).join('');
  const focus=rows.filter(r=>r.role!=='candidate').map(r=>{const x=X(r.changes.paloma),y=Y(r.changes.humaneval),color=r.eligible?'#176a92':'#a4572b';return (r.role==='selected'?`<path d="M${x},${y-7} L${x+7},${y} L${x},${y+7} L${x-7},${y} Z" fill="${color}" stroke="white"/>`:`<circle cx="${x}" cy="${y}" r="6" fill="${color}" stroke="white"/>`)+`<text x="${x+9}" y="${y+4}" font-size="11">${r.id.slice(0,4)}</text>`;}).join('');
  const ticks=[0,1,2,3,4].map(i=>{const x=xl+(xh-xl)*i/4,y=yl+(yh-yl)*i/4;return `<text x="${X(x)}" y="382" text-anchor="middle" font-size="12">${x.toFixed(2)}%</text><text x="68" y="${Y(y)+4}" text-anchor="end" font-size="12">${y.toFixed(2)}%</text>`;}).join('');
  q('#decision-chart').innerHTML=`<svg viewBox="0 0 750 438" role="img" aria-label="597个seed0候选，横轴Paloma BPB变化，纵轴HumanEval BPB变化，左下更好。符合限制${result.eligibleCount}个。完整数值在下表和CSV。"><rect x="76" y="32" width="620" height="330" fill="#f7fafb" stroke="#d7dfe5"/><line x1="${X(0)}" y1="32" x2="${X(0)}" y2="362" stroke="#a2b2bf" stroke-dasharray="4 3"/><line x1="76" y1="${Y(0)}" x2="696" y2="${Y(0)}" stroke="#a2b2bf" stroke-dasharray="4 3"/>${marks}${focus}${ticks}<text x="390" y="411" text-anchor="middle" font-size="14">Paloma macro BPB相对变化（%）</text><text x="17" y="192" transform="rotate(-90 17 192)" text-anchor="middle" font-size="14">HumanEval目标文本BPB相对变化（%）</text></svg>`;
  q('#decision-chart-reading').textContent='参照：'+input.baselines[result.plan.baseline].label+'。'+(zoom?'局部显示 '+visible.length+' / 597点；范围外 '+(597-visible.length)+' 点仍参与筛选与排名，可在完整范围和CSV查看。':'完整显示597个点，极端退步没有截走。')+' 主目标或限制改变后，符合条件的集合会变化。';
  q('#decision-chart').dataset.visibleCount=visible.length;q('#decision-chart').dataset.outsideCount=597-visible.length;
  q('#decision-chart svg').setAttribute('aria-label','597个seed0候选；当前显示'+visible.length+'点，范围外'+(597-visible.length)+'点仍参与筛选。横轴Paloma、纵轴HumanEval目标文本BPB变化，左下更好；符合当前限制'+result.eligibleCount+'个。');
 }
 function focus(){
  q('#decision-focus').innerHTML=result.rows.filter(r=>r.role!=='candidate').map(r=>`<div><strong>${r.role==='selected'?'选中996f':'替代197c'} · ${r.eligible?'符合当前限制':'被当前限制排除'}</strong><p>${r.eligible?'符合条件者中的观察排名：'+r.eligibleRank:'越界项：'+r.violations.map(v=>label(v.metric)+' '+pct(v.change)+' > '+pct(v.cap)).join('；')}</p><p>已公开的同权重seed标签：${r.observed_seeds.join('、')}。排名仍只使用seed0。</p><p class="status">${input.metrics.map(m=>m.label+' '+pct(r.changes[m.id])).join('；')}</p></div>`).join('');
 }
 function table(){
  if(!result)return;const term=q('#decision-row-search').value.trim().toLowerCase(),filter=q('#decision-row-filter').value;
  const rows=result.rows.filter(r=>r.id.toLowerCase().includes(term)&&(filter==='all'||filter==='eligible'&&r.eligible||filter==='excluded'&&!r.eligible));
  q('#decision-row-status').textContent='匹配 '+rows.length+' / 597；显示前 '+Math.min(30,rows.length)+' 行，按主目标BPB排序。';
  q('#decision-table thead').innerHTML='<tr><th>候选 / seed0</th><th>符合条件排名</th>'+input.metrics.map(m=>`<th>${esc(m.label)} Δ%</th>`).join('')+'<th>排除原因</th></tr>';
  q('#decision-table tbody').innerHTML=rows.slice(0,30).map(r=>`<tr><td><a href="https://wandb.ai/marin-community/marin_moe/runs/${esc(r.run)}" target="_blank" rel="noopener">${esc(r.id)}</a>${r.role==='selected'?'（选中）':r.role==='alternative'?'（替代）':''}</td><td>${r.eligibleRank??'排除'}</td>${input.metrics.map(m=>`<td class="${r.changes[m.id]>0?'decision-fail':'decision-pass'}" title="BPB ${r.values[m.id]}">${pct(r.changes[m.id])}</td>`).join('')}<td>${r.violations.map(v=>esc(label(v.metric))).join('、')||'当前限制未排除'}</td></tr>`).join('');
 }
 function render(){
  try{result=C.analyze(input,plan());q('#decision-status').textContent='探索性重算；阈值是在已知结果后设定。未执行独立确认，不会让R07/R12/R15自动通过。';q('#decision-status').classList.remove('decision-fail');}
  catch(e){result=null;q('#decision-status').textContent='输入无效：'+e.message+'。排序与导出已停止。';q('#decision-status').classList.add('decision-fail');for(const id of ['#decision-summary','#decision-chart','#decision-focus'])q(id).innerHTML='';q('#decision-table tbody').innerHTML='';q('#decision-row-status').textContent='';q('#decision-record').value='';for(const id of ['#decision-export-json','#decision-export-csv','#decision-copy'])q(id).disabled=true;return;}
  for(const id of ['#decision-export-json','#decision-export-csv','#decision-copy'])q(id).disabled=false;
  q('#decision-summary').innerHTML=`<div><strong>${result.eligibleCount} / 597</strong>符合启用限制；${result.excludedCount}个被排除</div><div><strong>${result.guardrailCount}项限制</strong>${result.guardrailCount?'每一项都必须满足':'未过滤风险，只有单目标排序'}</div><div><strong>${result.best?esc(result.best.id.slice(0,8)):'没有符合条件者'}</strong>${result.best?'当前条件下最小的'+esc(label(result.plan.primary))+' BPB；待复验':'应重新审视预算/目标/阈值，不能静默放宽后宣称通过'}</div>`;
  drawChart();focus();table();q('#decision-record').value=JSON.stringify(C.exportRecord(input,result),null,2);
 }
 function preset(){
  const p=q('#decision-preset').value;if(p==='custom')return;
  q('#decision-primary').value=p==='balanced'?'humaneval':'paloma';
  const caps=p==='macro'?{}:p==='balanced'?{paloma:0,gsm8k:0}:{paloma:0,humaneval:0,gsm8k:0,medqa:2,belebele:1,include:1};
  for(const m of input.metrics){const enabled=Object.hasOwn?Object.hasOwn(caps,m.id):Object.prototype.hasOwnProperty.call(caps,m.id);q('#cap-on-'+m.id).checked=enabled;q('#cap-value-'+m.id).disabled=!enabled;q('#cap-value-'+m.id).value=enabled?caps[m.id]:0;}render();
 }
 for(const m of input.metrics){q('#cap-on-'+m.id).addEventListener('change',()=>{q('#cap-value-'+m.id).disabled=!q('#cap-on-'+m.id).checked;q('#decision-preset').value='custom';render();});q('#cap-value-'+m.id).addEventListener('input',()=>{q('#decision-preset').value='custom';render();});}
 q('#decision-baseline').addEventListener('change',render);q('#decision-primary').addEventListener('change',()=>{q('#decision-preset').value='custom';render();});q('#decision-preset').addEventListener('change',preset);
 q('#decision-row-search').addEventListener('input',table);q('#decision-row-filter').addEventListener('change',table);
 q('#decision-chart-range').addEventListener('change',()=>{if(result)drawChart();});
 function download(name,body,type){const a=document.createElement('a'),url=URL.createObjectURL(new Blob([body],{type}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 q('#decision-export-json').addEventListener('click',()=>{if(result)download('marin-exploratory-decision.json',JSON.stringify(C.exportRecord(input,result),null,2),'application/json');});
 q('#decision-export-csv').addEventListener('click',()=>{if(result)download('marin-observed-candidates.csv','\ufeff'+C.toCSV(result,input.metrics),'text/csv;charset=utf-8');});
 q('#decision-copy').addEventListener('click',()=>{if(result){q('#decision-record-view').open=true;q('#decision-record').focus();q('#decision-record').select();}});
 preset();
})();
