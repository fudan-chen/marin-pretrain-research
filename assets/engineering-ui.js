(function(){
 'use strict';
 const element=document.getElementById('engineering-data');if(!element)return;
 const bank=JSON.parse(element.textContent), $=id=>document.getElementById(id);
 const notes=Object.create(null), labels={supported:'材料支持这个范围',refuted:'原记录有反证',unknown:'这层结论尚不能确认'};
 function node(tag,text){const e=document.createElement(tag);e.textContent=text;return e;}
 function current(){return bank.episodes.find(e=>e.id===$('engineering-case').value);}
 function link(ref){const a=node('a','原记录 / 固定 diff');a.href=ref.url;a.target='_blank';a.rel='noopener';return a;}
 bank.episodes.forEach(e=>{const o=node('option',e.id+' · '+e.title);o.value=e.id;$('engineering-case').append(o);});
 function claim(){const c=current().claims[Number($('engineering-claim').value)];$('engineering-verdict').textContent=labels[c.status];$('engineering-verdict').dataset.status=c.status;$('engineering-reason').textContent=c.reason;}
 function render(){
  const e=current();$('engineering-title').textContent=e.title;$('engineering-scope').textContent='规则 '+e.rules.join(' / ')+' · GitHub 单独刷新；训练曲线仍为 10 月 4 日快照';
  $('engineering-timeline').replaceChildren();e.events.forEach(v=>{const li=node('li','');li.append(node('strong',v.label),node('p',v.text),link(e.refs[v.ref_index]));$('engineering-timeline').append(li);});
  $('engineering-evidence').replaceChildren();[['实际观察','observation'],['机制解释','mechanism'],['做了什么','action'],['解决到哪一步','outcome'],['仍然未知','unknown']].forEach(([label,key])=>{const a=node('article','');a.append(node('h3',label),node('p',e[key]));$('engineering-evidence').append(a);});
  $('engineering-claim').replaceChildren();e.claims.forEach((c,i)=>{const o=node('option',c.text);o.value=String(i);$('engineering-claim').append(o);});claim();
  $('engineering-test').replaceChildren();[['方案','plan'],['预期','prediction'],['什么会推翻','falsifier'],['执行边界','boundary']].forEach(([l,k])=>$('engineering-test').append(node('dt',l),node('dd',e.next_test[k])));
  $('engineering-notes').value=notes[e.id]||'';$('engineering-export-status').textContent='';
 }
 $('engineering-case').addEventListener('change',render);$('engineering-claim').addEventListener('change',claim);
 $('engineering-notes').addEventListener('input',()=>{notes[current().id]=$('engineering-notes').value;});
 $('engineering-export').addEventListener('click',()=>{
  const e=current(),c=e.claims[Number($('engineering-claim').value)];
  const draft={schema:'marin-engineering-draft/1',atlas_sha256:bank.atlas_sha256,case_id:e.id,claim:c,source_sha256:bank.source_sha256,evidence:e,next_test:e.next_test,user_notes:$('engineering-notes').value,execution_status:'not_executed',reader_understanding:null,training_gate_decision:null};
  const blob=new Blob([JSON.stringify(draft,null,2)],{type:'application/json;charset=utf-8'}),url=URL.createObjectURL(blob),a=node('a','');a.href=url;a.download='marin-engineering-'+e.id+'.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  $('engineering-export-status').textContent='已生成本案 JSON；实际保存位置由浏览器决定。草案未执行，不产生训练决定。';
 });
 render();
})();
