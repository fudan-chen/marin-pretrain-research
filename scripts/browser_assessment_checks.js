// Tabbit program. Test fixtures are not reader observations; original local data is restored.
const equal=(a,b)=>assert.equal(JSON.stringify(a),JSON.stringify(b));
globalThis.v9OriginalLocal=await page.evaluate(()=>({...localStorage}));
const errors=[];page.on('pageerror',e=>errors.push(String(e)));
await page.goto('http://127.0.0.1:8765/?view=v9#assessment-lab',{waitUntil:'domcontentloaded'});
const bank=JSON.parse(await page.locator('#assessment-data').textContent());
const key='marin-reader-assessment-0.1-'+bank.bank_sha256;
const exposeJSON=async()=>{await page.locator('#assessment-json').evaluate(e=>e.closest('details').open=true)};
async function importRecord(record){await exposeJSON();await page.locator('#assessment-json').fill(JSON.stringify(record));await page.getByRole('button',{name:'导入并检查格式',exact:true}).click();}
async function serialized(){await page.getByRole('button',{name:'把当前记录放入JSON框',exact:true}).click();return JSON.parse(await page.locator('#assessment-json').inputValue());}
const blank=await page.evaluate(()=>AssessmentCore.blank(JSON.parse(document.getElementById('assessment-data').textContent)));
let exported,frame=[];
try{
 await importRecord(blank);
 for(const c of bank.cases){
  await page.getByLabel('选择题目',{exact:true}).selectOption(c.id);assert.equal(await page.locator('#assessment-prompt').textContent(),c.prompt);
  await page.locator('#assessment-reference').evaluate(e=>e.open=true);
  await expect(page.locator('#assessment-reference-body')).toContainText(c.allowed_claim);
  assert.equal(await page.locator('#assessment-reference-body tbody tr').count(),3);
  const figureHref=await page.locator('#assessment-reference-body a').last().getAttribute('href');assert.equal(await page.locator(figureHref).count(),1);
  frame.push(c.id);
 }
 await importRecord(blank);await page.getByLabel('选择题目',{exact:true}).selectOption('Q09');
 await page.getByLabel('答题者标记（可留空，用代号即可）',{exact:true}).fill('浏览器验收样例，并非读者观察');
 await page.getByLabel('你的判断：看到什么、为什么、还缺什么、怎样验证',{exact:true}).fill('整块累计为零就够了；140序列就是历史读取差。 <img src=x onerror="alert(1)">');
 await page.getByRole('button',{name:'保存为一份新回答',exact:true}).click();
 let r=await serialized();assert.equal(r.cases.Q09.revisions[0].reference_in_page_status,'not_opened_in_this_page');assert.equal(await page.locator('#assessment-saved-answer img').count(),0);
 await page.locator('#assessment-reference').evaluate(e=>e.open=true);
 await page.waitForFunction(key=>JSON.parse(localStorage.getItem(key)).cases.Q09.reference_first_opened_at!==null,key);
 async function rate(slot,label,scope,critical,note){
  await page.getByLabel('评审记录',{exact:true}).selectOption(slot);await page.getByLabel('评审者标记',{exact:true}).fill(label);
  await page.getByLabel('结论范围',{exact:true}).selectOption(String(scope));await page.getByLabel('限制原因',{exact:true}).selectOption('2');await page.getByLabel('最小追加证据',{exact:true}).selectOption('2');
  await page.getByLabel('是否检查关键越界',{exact:true}).selectOption(critical?'issues':'none');
  await page.locator('#assessment-critical-options input').setChecked(critical);
  await page.getByLabel('摘录答案原句，并说明与哪条锚点对应；仲裁还须说明为何取舍',{exact:true}).fill(note);
  await page.getByRole('button',{name:'保存这份人工评审',exact:true}).click();
 }
 await rate('A','验收A',2,true,'“140序列就是历史读取差”越过诊断范围；本评分是测试输入。');
 await rate('B','验收B',2,false,'测试另一份意见，以检查分歧是否保留；不代表真实评审。');
 await expect(page.locator('#assessment-review-result')).toContainText('两份评分有分歧');
 await expect(page.locator('#assessment-review-result')).toContainText('关键越界');
 r=await serialized();assert.equal(r.summary.cases.find(x=>x.case_id==='Q09').anchor_status,'critical_overreach');
 assert.equal(r.summary.overall_score,null);assert.equal(r.summary.training_gate_decision,null);
 await rate('adjudication','验收仲裁C',2,true,'原句将合法关闭shuffle反例当历史实测；依Q09范围锚点保留越界。此仲裁仅为验收样例。');
 await expect(page.locator('#assessment-review-result')).toContainText('人工仲裁');
 const before=await serialized();await rate('A','验收A',0,true,'试图覆盖已仲裁依据，应被阻止。');
 await expect(page.locator('#assessment-message')).toContainText('原记录保留');equal(await serialized(),before);
 await page.getByLabel('你的判断：看到什么、为什么、还缺什么、怎样验证',{exact:true}).fill('完整块相同仍须查部分块；共同边缘和游标提供条件匹配，真实key与store仍待核对。');
 await page.getByRole('button',{name:'保存为一份新回答',exact:true}).click();r=await serialized();assert.equal(r.cases.Q09.revisions.length,2);assert.equal(r.cases.Q09.revisions[1].reference_in_page_status,'opened_before_save');assert.equal(r.summary.cases.find(x=>x.case_id==='Q09').review_status,'reviews_pending');assert.equal(r.cases.Q09.revisions[0].review_events.length,3);
 await page.reload({waitUntil:'domcontentloaded'});await page.getByLabel('选择题目',{exact:true}).selectOption('Q09');equal(await serialized(),r);
 const bad=JSON.parse(JSON.stringify(r));bad.bank_sha256='other-version';await importRecord(bad);await expect(page.locator('#assessment-message')).toContainText('当前记录保留');equal(await serialized(),r);
 const fabricated=JSON.parse(JSON.stringify(r));fabricated.summary={overall_score:100,training_gate_decision:'pass',external_reader_study:'verified'};await importRecord(fabricated);equal(await serialized(),r);
 // Verify actual export button Blob, prevent an OS download from this fixture.
 await page.evaluate(()=>{window.v9Blobs=[];window.v9Create=URL.createObjectURL;URL.createObjectURL=function(b){window.v9Blobs.push(b);return window.v9Create.call(this,b)};window.v9Prevent=e=>{if(e.target.closest('a[download]'))e.preventDefault()};document.addEventListener('click',window.v9Prevent,true)});
 try{await page.getByRole('button',{name:'导出理解检查JSON',exact:true}).click();exported=await page.evaluate(async()=>JSON.parse(await window.v9Blobs.at(-1).text()));equal(exported,r);}finally{await page.evaluate(()=>{URL.createObjectURL=window.v9Create;document.removeEventListener('click',window.v9Prevent,true);delete window.v9Blobs;delete window.v9Create;delete window.v9Prevent});}
 // Current framework migration is explicit, R16 reset and original evidence kept.
 await page.locator('#review-json').evaluate(e=>e.closest('details').open=true);await page.getByRole('button',{name:'填入当前评审JSON',exact:true}).click();
 const originalReview=JSON.parse(await page.locator('#review-json').inputValue()),old=JSON.parse(JSON.stringify(originalReview));old.framework_version='1.0';
 await page.locator('#review-json').fill(JSON.stringify(old));await page.getByRole('button',{name:'导入评审JSON',exact:true}).click();await expect(page.locator('#review-import-status')).toContainText('当前判断保留');
 await page.getByRole('button',{name:'从1.0迁移并重评R16',exact:true}).click();await page.getByRole('button',{name:'填入当前评审JSON',exact:true}).click();
 const migrated=JSON.parse(await page.locator('#review-json').inputValue());assert.equal(migrated.framework_version,'1.1');assert.equal(migrated.judgments.R16.status,'unassessed');equal(JSON.parse(migrated.migration.prior_R16_json),old.judgments.R16);
 for(const id of Object.keys(old.judgments))if(id!=='R16')equal(migrated.judgments[id],old.judgments[id]);
 assert.equal(errors.length,0);
}finally{
 await page.evaluate(saved=>{for(const k of Object.keys(localStorage))if(!(k in saved))localStorage.removeItem(k);for(const [k,v] of Object.entries(saved))localStorage.setItem(k,v)},globalThis.v9OriginalLocal);
 await page.reload({waitUntil:'domcontentloaded'});
 equal(await page.evaluate(()=>({...localStorage})),globalThis.v9OriginalLocal);
}
return {bank_sha256:bank.bank_sha256,functional:{tenSourceBoundCases:true,firstAnswerExposureRecorded:true,htmlLikeTextEscaped:true,criticalOverreachCannotBeAveraged:true,disagreementAndAdjudicationPreserved:true,newRevisionDoesNotInheritRatings:true,persistedAfterReload:true,invalidVersionRetainsRecord:true,fabricatedSummaryIgnored:true,actualExportBlobChecked:true,explicitRuleMigrationResetsR16:true,existingLocalDataRestored:true},caseIds:frame,errors,exportScope:'Actual button Blob checked, OS save not tested',exportSummary:{schema:exported.schema,answerRevisions:exported.cases.Q09.revisions.length,reviewEvents:exported.cases.Q09.revisions[0].review_events.length,overallScore:exported.summary.overall_score,externalReaderStudy:exported.summary.external_reader_study,trainingGateDecision:exported.summary.training_gate_decision}};
