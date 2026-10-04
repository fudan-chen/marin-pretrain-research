const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');const core=require('../assets/order-core.js');
const d=JSON.parse(fs.readFileSync('analysis/order_workbench_data.json')),probe=JSON.parse(fs.readFileSync('analysis/order_loader_probe_python312.json'));const digest=x=>crypto.createHash('sha256').update(JSON.stringify(Object.fromEntries(Object.keys(x).sort().map(k=>[k,x[k]])))).digest('hex');let passed=[];
for(let k=0;k<=19;k++)for(const lock of [true,false]){
 const p=core.compile(d,k,lock);for(const a of p.arms)for(let s=0;s<a.stages.length;s++){
  const n=probe.compiled_rows.find(r=>r.k===k&&r.lock_edges===lock&&r.arm===a.id&&r.stage===s);assert(n);assert.equal(digest(a.stages[s].counts),n.count_sha256);
  assert.equal(Object.values(a.stages[s].counts).reduce((x,y)=>x+y,0),49152);
 }assert(p.arms.every(a=>a.interior_l1_difference_sequences===0));
}
passed.push('All 40 mode/amplitude combinations match 480 native count digests');
const unlocked=core.compile(d,10,false);for(const a of unlocked.arms){const n=probe.unlocked_debug_k10.find(x=>x.arm===a.id);assert.deepEqual(a.unpermuted_debug_difference_sequences,n.difference_sequences);assert.equal(a.unpermuted_debug_l1_sequences,n.l1_difference_sequences)}passed.push('Unpermuted negative control matches archived-method diagnostic');
assert.equal(core.compile(d,10,true).whole_window_count_match_by_construction,true);assert.equal(unlocked.whole_window_count_match_by_construction,false);passed.push('Interior equality cannot waive changed outer blocks');
for(const bad of [20,-1,1.5,NaN,Infinity,true,null])assert.throws(()=>core.compile(d,bad,true));passed.push('Invalid amplitude fails without clipping');
assert.equal(core.compile(d,0,false).whole_window_count_match_by_construction,true);assert.equal(new Set(core.compile(d,0,true).arms.map(a=>JSON.stringify(a.interior_counts))).size,1);passed.push('Zero amplitude remains one no-op recipe');
const exported=core.compile(d,10,true);assert(exported.arms.every(a=>a.stages.every(s=>!('weights' in s))));assert.equal(exported.generation_results,null);assert.equal(exported.actual_data_cursor,null);assert.equal(exported.launchable,false);passed.push('Export is a planned count contract and preserves unknown live state');
const before=JSON.stringify(d);core.compile(d,19,false);assert.equal(JSON.stringify(d),before);passed.push('Source inputs remain immutable');
fs.writeFileSync('analysis/order_js_validation.json',JSON.stringify({test_groups_passed:passed.length,tests:passed,mode_amplitude_cases:40,scope:'Count ledger and exported states, not trainer execution'},null,2)+'\n');console.log('Order JS:',passed.length,'groups passed');
