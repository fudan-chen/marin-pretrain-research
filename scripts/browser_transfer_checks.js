// Run as Tabbit nodejs input on the acquired local report tab; no Node imports.
// Real button Blob is checked; the download default is prevented, no OS save claim.
const equal=(a,b)=>assert.equal(JSON.stringify(a),JSON.stringify(b));
const savedLocal = await page.evaluate(() => ({...localStorage}));
const errors=[];page.on('pageerror',e=>errors.push(String(e)));
await page.goto('http://127.0.0.1:8765/?view=v7#transfer-lab',{waitUntil:'domcontentloaded'});
await expect(page.getByLabel('相对996f的候选',{exact:true})).toBeVisible();
const input=JSON.parse(await page.locator('#transfer-data').textContent()),matrix=[];
for(const p of input.profiles){for(const budget of ['nominal','recorded_physical']){
 await page.getByLabel('相对996f的候选',{exact:true}).selectOption(p.id);
 await page.getByLabel('候选差值的预算口径',{exact:true}).selectOption(budget);
 const texts=await page.locator('#transfer-summary strong').allTextContents(),c=p.comparisons[budget];
 equal(texts,[p.changed_phase_cells+' / 200',...[c.moved_tokens,c.cross_domain_net_tokens,c.within_domain_cancellation_tokens].map(x=>(x/1e9).toFixed(3)+' B')]);
 await page.getByLabel('查看一个域内部的五档',{exact:true}).selectOption('28');
 const shown=await page.locator('#transfer-quality-table tbody tr').evaluateAll(rs=>rs.map(r=>[...r.cells].map(c=>c.textContent)));
 const expected=c.cells.filter(x=>x.domain===28).map(x=>[x.cell,(x.base_tokens/1e9).toFixed(3),(x.target_tokens/1e9).toFixed(3),(x.delta_tokens>0?'+':'')+(x.delta_tokens/1e9).toFixed(3)]);
 equal(shown,expected);assert.equal(await page.locator('#transfer-domain-table tbody tr').count(),40);
 matrix.push({candidate:p.id,budget,summary:texts,qualityRows:shown.length});
}}
await page.evaluate(()=>{window.v7Blobs=[];window.v7OriginalCreate=URL.createObjectURL;
 URL.createObjectURL=function(blob){window.v7Blobs.push(blob);return window.v7OriginalCreate.call(this,blob)};
 window.v7PreventDownload=e=>{if(e.target.closest('a[download]'))e.preventDefault()};document.addEventListener('click',window.v7PreventDownload,true)});
const plans=[];
try{
 for(const p of input.plans){
  await page.getByLabel('数学加量的供体',{exact:true}).selectOption(String(p.donor_domain));
  const actual=JSON.parse(await page.getByLabel('数学交换草案JSON',{exact:true}).inputValue());equal(actual,p);
  assert.equal(await page.locator('#transfer-arms tbody tr').count(),4);
  await page.getByRole('button',{name:'导出当前四臂草案JSON',exact:true}).click();
  const output=await page.evaluate(async()=>JSON.parse(await window.v7Blobs.at(-1).text()));equal(output,p);
  assert.equal(output.launchable,false);assert.equal(output.generation_results,null);
  plans.push({donor:p.donor_domain,arms:output.arms.length,status:output.status,launchable:output.launchable,resultsNull:output.generation_results===null});
 }
}finally{
 await page.evaluate(()=>{URL.createObjectURL=window.v7OriginalCreate;document.removeEventListener('click',window.v7PreventDownload,true);
  delete window.v7Blobs;delete window.v7OriginalCreate;delete window.v7PreventDownload});
}
const printTables=await page.evaluate(()=>{const ds=[...document.querySelectorAll('.large-table-details')],before=ds.map(d=>d.open);
 window.dispatchEvent(new Event('beforeprint'));const allOpened=ds.every(d=>d.open);window.dispatchEvent(new Event('afterprint'));
 return {count:ds.length,allOpened,restored:ds.every((d,i)=>d.open===before[i])}});
assert.equal(printTables.count,6);assert(printTables.allOpened&&printTables.restored);
equal(await page.evaluate(()=>({...localStorage})),savedLocal);assert.equal(errors.length,0);
await page.getByLabel('相对996f的候选',{exact:true}).selectOption(input.profiles[0].id);
await page.getByLabel('候选差值的预算口径',{exact:true}).selectOption('nominal');
await page.getByLabel('数学加量的供体',{exact:true}).selectOption('26');
return {functional:{eightProfileBudgetCasesMatch:true,fiveQualityRowsMatch:true,threePlansMatch:true,exportRetainsPlannedNullState:true,existingLocalDataPreserved:true},matrix,plans,printTables,errors,
 exportScope:'Observed Blob from real button; OS file save not tested'};
