'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const R=path.resolve(__dirname,'..'),C=require('../assets/decision-core.js'),d=JSON.parse(fs.readFileSync(path.join(R,'analysis/decision_data.json'),'utf8'));
let groups=0;const test=(name,fn)=>{fn();groups++;console.log('PASS',name);};
const p=(caps={},baseline='selected',primary='paloma')=>({baseline,primary,caps});
test('Missing or invalid metric cannot silently pass',()=>{
 const bad=structuredClone(d);delete bad.candidates[0].values.gsm8k;assert.throws(()=>C.analyze(bad,p()),/缺失/);
 for(const plan of [p({gsm8k:NaN}),p({wrong:0}),p({gsm8k:'0'}),p({},'absent'),p({},'selected','absent')])assert.throws(()=>C.analyze(d,plan));
});
test('Every constraint must pass before primary ranking',()=>{
 const r=C.analyze(d,p({gsm8k:0,medqa:2}));for(const row of r.rows)assert.equal(row.eligible,row.changes.gsm8k<=1e-10&&row.changes.medqa<=2+1e-10);
 assert.equal(r.best?.id,r.rows.filter(x=>x.eligible).sort((a,b)=>a.values.paloma-b.values.paloma||a.id.localeCompare(b.id))[0]?.id);
});
test('197c three-metric improvement does not cancel medical regression',()=>{
 const r=C.analyze(d,p({paloma:0,humaneval:0,gsm8k:0,medqa:2})),a=r.rows.find(r=>r.role==='alternative');
 assert(a.changes.paloma<0&&a.changes.humaneval<0&&a.changes.gsm8k<0);assert(a.changes.medqa>10);assert(!a.eligible);assert(a.violations.some(v=>v.metric==='medqa'));
});
test('Stricter caps produce a subset with no hidden relaxation',()=>{
 const loose=C.analyze(d,p({gsm8k:2})),strict=C.analyze(d,p({gsm8k:0,medqa:0}));const ids=new Set(loose.rows.filter(r=>r.eligible).map(r=>r.id));
 assert(strict.rows.filter(r=>r.eligible).every(r=>ids.has(r.id)));assert(strict.eligibleCount<=loose.eligibleCount);
});
test('Infeasible input returns no winner and preserves every rejection',()=>{
 const r=C.analyze(d,p({paloma:-100}));assert.equal(r.eligibleCount,0);assert.equal(r.best,null);assert.equal(r.excludedCount,597);assert(r.rows.every(r=>r.violations.length));
});
test('No guardrails is explicitly single-objective retrospective ranking',()=>{
 const r=C.analyze(d,p());assert.equal(r.eligibleCount,597);assert.equal(r.guardrailCount,0);assert(r.rows.every(r=>r.eligibleRank));
 const record=C.exportRecord(d,r);assert.equal(record.status,'exploratory_reanalysis_not_preregistered');assert.equal(record.original_selector_reconstructed,false);assert.equal(record.independent_confirmation_executed,false);assert.equal(record.capability_accuracy_verified,false);assert.equal(record.next_confirmation.token_budget,null);
});
test('Changing baseline changes the denominator without altering raw BPB',()=>{
 const a=C.analyze(d,p({},'old')),b=C.analyze(d,p({},'proportional'));
 for(const row of a.rows){const other=b.rows.find(r=>r.id===row.id);assert.deepEqual(row.values,other.values);for(const m of d.metrics){assert.equal(row.changes[m.id],100*(row.values[m.id]/d.baselines.old.values[m.id]-1));assert.equal(other.changes[m.id],100*(row.values[m.id]/d.baselines.proportional.values[m.id]-1));}}
});
test('Seven raw metrics all survive ranking and CSV export',()=>{
 const r=C.analyze(d,p({},'proportional','humaneval')),csv=C.toCSV(r,d.metrics);assert.equal(csv.trim().split('\n').length,598);assert(csv.includes('"medqa_delta_pct"'));assert(csv.includes('"include_delta_pct"'));assert(csv.includes('197c9f5ceff6b9ee'));
 const back=JSON.parse(JSON.stringify(C.exportRecord(d,r)));assert.equal(back.dataset_sha256,d.dataset_sha256);assert.equal(back.ranking_seed,0);assert.equal(back.shortlist.length,10);
});
test('Input arrays and immutable dataset remain unchanged',()=>{
 const before=JSON.stringify(d),caps={gsm8k:0},plan=p(caps);C.analyze(d,plan);assert.equal(JSON.stringify(d),before);assert.deepEqual(caps,{gsm8k:0});
});
const scenarios=[];
for(const baseline of ['proportional','old','selected'])for(const [name,primary,caps] of [
 ['macro_only','paloma',{}],['code_with_math_guard','humaneval',{paloma:0,gsm8k:0}],
 ['three_metrics_with_medical_guard','paloma',{paloma:0,humaneval:0,gsm8k:0,medqa:2,belebele:1,include:1}]]){
 const r=C.analyze(d,p(caps,baseline,primary));scenarios.push({baseline,name,primary,caps,eligible:r.eligibleCount,best:r.best?.id??null,
 selected:r.rows.filter(r=>r.role==='selected'||r.role==='alternative').map(r=>({id:r.id,rank:r.eligibleRank,violations:r.violations,changes:r.changes}))});
}
fs.writeFileSync(path.join(R,'analysis/decision_validation.json'),JSON.stringify({test_groups_passed:groups,candidates:597,metrics:7,scenarios,
 not_verified:['historical selector','preregistered validity of retrospective thresholds','independent model confirmation','generation accuracy or target-scale performance']},null,2)+'\n');
console.log('Decision tests passed:',groups);
