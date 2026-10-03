const cells=JSON.parse(document.getElementById('bucket-data').textContent);
const samples=JSON.parse(document.getElementById('sample-data').textContent);
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const domains=[...new Map(cells.map(c=>[c.domain_id,[c.domain_zh,c.domain]])).entries()];
$('#bucket-domain').innerHTML='<option value="">全部40域</option>'+domains.map(([i,n])=>`<option value="${i}">c${String(i).padStart(2,'0')} ${esc(n[0])}</option>`).join('');
function updateBuckets(){
 const query=$('#bucket-search').value.toLowerCase().trim(),dom=$('#bucket-domain').value,q=$('#bucket-quality').value,sort=$('#bucket-sort').value;
 let rows=cells.filter(c=>(dom===''||c.domain_id===Number(dom))&&(q===''||c.quality===Number(q))&&`${c.cell} ${c.domain} ${c.domain_zh}`.toLowerCase().includes(query));
 if(sort==='epochs')rows.sort((a,b)=>b.planned_epochs-a.planned_epochs);else if(sort==='delta')rows.sort((a,b)=>(b.phase1_pct-b.phase0_pct)-(a.phase1_pct-a.phase0_pct));else rows.sort((a,b)=>a.cell.localeCompare(b.cell));
 const pct=v=>v.toFixed(v<.01?6:3)+'%';
 $('#bucket-rows').innerHTML=rows.map(c=>`<tr><td><button data-cell="${c.cell}" aria-label="查看${c.cell}样本">${c.cell}</button></td><td>${esc(c.domain_zh)}<br><small>${esc(c.domain)}</small></td><td>${(c.available_tokens/1e9).toFixed(4)}</td><td>${pct(c.phase0_pct)}</td><td>${pct(c.phase1_pct)}</td><td>${pct(c.phase2_pct)}</td><td>${(c.planned_tokens/1e9).toFixed(4)}</td><td>${c.planned_epochs.toFixed(3)}</td></tr>`).join('');
 $('#bucket-count').textContent=`显示 ${rows.length} / 200 桶；B = 十亿tokens。未来阶段按当前4K配置计算。`;
}
for(const s of ['#bucket-search','#bucket-domain','#bucket-quality','#bucket-sort'])$(s).addEventListener('input',updateBuckets);
$('#bucket-rows').addEventListener('click',e=>{
 const b=e.target.closest('[data-cell]');if(!b)return;
 const c=cells.find(c=>c.cell===b.dataset.cell),a=samples.find(x=>x.cell===c.cell);const panel=$('#sample-panel');
 panel.hidden=false;panel.innerHTML=`<h3>${c.cell} · ${esc(c.domain_zh)}</h3><p>以下为原组成页公开样本的前300字符，仅用于发现标签不纯，不能估计质量比例。</p>${a.examples.map((t,i)=>`<p>公开样本 ${i+1}</p><pre>${esc(t)}</pre>`).join('')}<a target="_blank" rel="noopener" href="https://storage.googleapis.com/marin-public/held/harrier-k40-cluster-overview/2026.08.18/index.html?revision=uniform-sampling#cluster-${c.domain_id}">打开原页面的这个cluster</a>`;
 panel.scrollIntoView({block:'nearest'});
});
updateBuckets();
$('#toc-search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('.sidebar details a').forEach(a=>{a.hidden=!a.textContent.toLowerCase().includes(q)});if(q)document.querySelectorAll('.sidebar details').forEach(d=>d.open=true)});
const notes=$('#local-notes'),noteKey='marin-research-2026-10-04-notes';
try{notes.value=localStorage.getItem(noteKey)||''}catch(e){$('#notes-status').textContent='此浏览器不允许本地存储，请用导出保存。'}
notes.addEventListener('input',()=>{try{localStorage.setItem(noteKey,notes.value);$('#notes-status').textContent='已保存到本浏览器；不会上传。'}catch(e){$('#notes-status').textContent='本地存储失败，请用导出保存。'}});
$('#export-notes').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([notes.value],{type:'text/plain;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='marin-研究笔记.txt';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
$('#print-report').addEventListener('click',()=>window.print());
