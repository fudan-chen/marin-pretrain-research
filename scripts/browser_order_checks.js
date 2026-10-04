// Tabbit nodejs program on the acquired local report. No Node imports.
const equal=(a,b)=>assert.equal(JSON.stringify(a),JSON.stringify(b));
const savedLocal=await page.evaluate(()=>({...localStorage})),errors=[];page.on('pageerror',e=>errors.push(String(e)));
await page.goto('http://127.0.0.1:8765/?view=v8#order-lab',{waitUntil:'domcontentloaded'});
await expect(page.getByLabel('历史顺序对照',{exact:true})).toBeVisible();
const data=JSON.parse(await page.locator('#order-data').textContent()),histories=[],cases=[];
const fmt=n=>n!==0&&Math.abs(n)<1e-5?n.toExponential(3):n.toLocaleString('en-US',{maximumFractionDigits:6});
for(let i=0;i<data.historical_pairs.length;i++){
 await page.getByLabel('历史顺序对照',{exact:true}).selectOption(String(i));
 const p=data.historical_pairs[i],values=await page.locator('#order-history-values tbody tr').evaluateAll(rows=>rows.map(r=>r.cells[1].textContent));
 equal(values,[...p.continuous_conventions.map(c=>fmt(Number(c.l1_exposure_difference_tokens))),fmt(p.complete_block_l1_tokens),'未确认']);
 histories.push({pair:p.pair,values});
}
for(let k=0;k<=19;k++)for(const lock of [true,false]){
 await page.getByLabel('整数步幅k（0至19）',{exact:true}).fill(String(k));
 await page.getByLabel('边缘块处理',{exact:true}).selectOption(lock?'locked':'changed');
 const output=JSON.parse(await page.getByLabel('顺序计数草案JSON',{exact:true}).inputValue());
 assert.equal(output.amplitude_k,k);assert.equal(output.lock_edge_blocks,lock);assert.equal(output.whole_window_count_match_by_construction,lock||k===0);
 const shown=await page.locator('#order-arm-table tbody tr').evaluateAll(rows=>rows.map(r=>[...r.cells].map(c=>c.textContent)));
 const expected=output.arms.map(a=>[a.id,String(data.baseline_counts[0].c39q4+a.sign*k*2),String(data.baseline_counts[1].c39q4-a.sign*k*7),'0',fmt(a.unpermuted_debug_l1_sequences)]);equal(shown,expected);
 cases.push({k,lock,wholeMatch:output.whole_window_count_match_by_construction,receiverCounts:shown.map(r=>r.slice(1,3)),debugL1:shown.map(r=>r[4])});
}
await page.getByLabel('整数步幅k（0至19）',{exact:true}).fill('10');await page.getByLabel('边缘块处理',{exact:true}).selectOption('changed');
const counter=JSON.parse(await page.getByLabel('顺序计数草案JSON',{exact:true}).inputValue());equal(counter.arms.map(a=>a.unpermuted_debug_l1_sequences),[0,140,140]);
assert((await page.locator('#order-guarantee').textContent()).includes('整体匹配没有建立'));
for(const raw of ['20','-1','1.5','']){
 await page.getByLabel('整数步幅k（0至19）',{exact:true}).fill(raw);await expect(page.locator('#order-error')).toBeVisible();await expect(page.locator('#order-result')).toBeHidden();assert.equal(await page.getByLabel('顺序计数草案JSON',{exact:true}).inputValue(),'');
}
await page.getByLabel('整数步幅k（0至19）',{exact:true}).fill('10');await page.getByLabel('边缘块处理',{exact:true}).selectOption('locked');
await page.evaluate(()=>{window.orderCaptured=[];window.orderOriginalCreate=URL.createObjectURL;URL.createObjectURL=function(blob){window.orderCaptured.push(blob);return window.orderOriginalCreate.call(this,blob)};window.orderPrevent=e=>{if(e.target.closest('a[download]'))e.preventDefault()};document.addEventListener('click',window.orderPrevent,true)});
let exported;
try{
 await page.getByRole('button',{name:'导出当前计数草案JSON',exact:true}).click();exported=await page.evaluate(async()=>JSON.parse(await window.orderCaptured.at(-1).text()));
 assert.equal(exported.schema,'marin-count-order-review/1');assert.equal(exported.launchable,false);assert.equal(exported.generation_results,null);assert.equal(exported.actual_data_cursor,null);assert(exported.arms.every(a=>a.stages.every(s=>!('weights' in s))));
}finally{
 await page.evaluate(()=>{URL.createObjectURL=window.orderOriginalCreate;document.removeEventListener('click',window.orderPrevent,true);delete window.orderCaptured;delete window.orderOriginalCreate;delete window.orderPrevent});
}
const printTables=await page.evaluate(()=>{const ds=[...document.querySelectorAll('.large-table-details')],before=ds.map(d=>d.open);window.dispatchEvent(new Event('beforeprint'));const allOpened=ds.every(d=>d.open);window.dispatchEvent(new Event('afterprint'));return {count:ds.length,allOpened,restored:ds.every((d,i)=>d.open===before[i])}});assert.equal(printTables.count,6);assert(printTables.allOpened&&printTables.restored);
equal(await page.evaluate(()=>({...localStorage})),savedLocal);assert.equal(errors.length,0);
await page.getByLabel('历史顺序对照',{exact:true}).selectOption('0');
return {functional:{eightHistoricalPairsMatch:true,fortyModeAmplitudeCasesMatch:true,unlockedCounterexampleVisible:true,invalidAmplitudeBlocksOutput:true,exportIsPlannedCountsOnly:true,existingLocalDataPreserved:true},histories,cases,printTables,errors,exportScope:'Actual button Blob checked; OS save not tested',exportedSummary:{schema:exported.schema,status:exported.status,stageCounts:exported.arms.map(a=>a.stages.length),generationResults:exported.generation_results}};
