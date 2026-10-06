"""Original JAX clipping/chunk-plan/A2A parameter helpers with host transfer replay.
Synthetic identity rows; no actual collective, GPU, routing planner or model forward.
"""
import ast,hashlib,json,pathlib,types
from typing import NamedTuple
import jax,jax.numpy as jnp,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];E=R/'sources/receiver_layout_2026_10_07/ep_common.py';P=R/'sources/engineering_v77_2026_10_07/current_ep_ragged_all_to_all.py';ns={'jax':jax,'jnp':jnp,'NamedTuple':NamedTuple}
def extract(p,names):
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),str(p),'exec'),ns)
extract(E,['_prefix_cap_counts','_clip_receiver_group_sizes','ExpertA2aParams','_expert_granular_a2a_params']);extract(P,['_ChunkPlan','_chunk_plans'])
def run(counts,logical,physical,semantic_columns):
 c=jnp.array(counts,dtype=jnp.int32);nshards=2;local=4;chunks=2;chunk_experts=2;chunk_of=(jnp.arange(8)%local)//chunk_experts
 clipped=tuple(ns['_clip_receiver_group_sizes'](jnp.where(chunk_of[None,:]==k,c,0),local_expert_size=local,receiver_capacity=logical) for k in range(chunks))
 plans=[]
 for s in range(nshards):
  routing=types.SimpleNamespace(all_group_sizes=c,chunk_clipped_group_sizes=clipped,shard_id=s)
  layout=types.SimpleNamespace(local_experts=local,ep_size=nshards,chunk_experts=chunk_experts,chunk_capacity=physical)
  plans.append(ns['_chunk_plans'](routing,layout))
 rows=[];expert_rows=[]
 for s in range(nshards):
  rr=[];ee=[]
  for e,num in enumerate(counts[s]):
   rr.extend([s*10000+semantic_columns[e]*100+i for i in range(num)]);ee.extend([e]*num)
  rows.append(np.array(rr,dtype=float));expert_rows.append(np.array(ee))
 receiver=[[np.full(physical,np.nan) for k in range(chunks)] for s in range(nshards)];writes=[[set() for k in range(chunks)] for s in range(nshards)]
 for sender in range(nshards):
  for k in range(chunks):
   params=plans[sender][k].dispatch_params
   for dest in range(nshards):
    for e in range(local):
     i=dest*local+e;a=int(params.input_offsets[i]);b=int(params.output_offsets[i]);length=int(params.send_sizes[i])
     assert length==int(plans[dest][k].dispatch_params.recv_sizes[sender*local+e])
     if length:
      assert not (set(range(b,b+length))&writes[dest][k]);assert b+length<=physical
      receiver[dest][k][b:b+length]=rows[sender][a:a+length];writes[dest][k].update(range(b,b+length))
 restored=[np.full(len(x),np.nan) for x in rows];return_writes=[set() for x in rows];records=[]
 for dest in range(nshards):
  for k in range(chunks):
   plan=plans[dest][k];a=np.asarray(plan.active_group_sizes);p=np.asarray(plan.physical_group_sizes);n=int(a.sum())
   assert writes[dest][k]==set(range(n));assert np.array_equal(p[:-1],a[:-1]);assert int(p.sum())==physical and int(p[-1]-a[-1])==physical-n
   expected=[]
   for e in range(k*chunk_experts,(k+1)*chunk_experts):
    global_e=dest*local+e
    for sender in range(nshards):
     length=int(clipped[k][sender,global_e]);expected.extend([sender*10000+semantic_columns[global_e]*100+i for i in range(length)])
   assert np.array_equal(receiver[dest][k][:n],expected)
   ret=plan.return_params
   for sender in range(nshards):
    for e in range(local):
     i=sender*local+e;start=int(ret.input_offsets[i]);out=int(ret.output_offsets[i]);length=int(ret.send_sizes[i])
     assert length==int(plans[sender][k].return_params.recv_sizes[dest*local+e])
     if length:
      assert not (set(range(out,out+length))&return_writes[sender])
      restored[sender][out:out+length]=receiver[dest][k][start:start+length];return_writes[sender].update(range(out,out+length))
   records.append({'receiver':dest,'chunk':k,'raw_demand':int(c[:,dest*local+k*chunk_experts:dest*local+(k+1)*chunk_experts].sum()),'active_group_sizes':a.tolist(),'physical_group_sizes':p.tolist(),'receiver_ids_active':receiver[dest][k][:n].astype(int).tolist(),'tail_unwritten_rows':physical-n,'dispatch_params':{k:np.asarray(v).tolist() for k,v in plan.dispatch_params._asdict().items()},'return_params':{k:np.asarray(v).tolist() for k,v in plan.return_params._asdict().items()}})
 accepted=np.asarray(sum(clipped));semantic_accepted=np.zeros(8,dtype=int)
 for e in range(8):semantic_accepted[semantic_columns[e]]=accepted[:,e].sum()
 for s in range(nshards):
  mask=np.zeros(len(rows[s]),dtype=bool);start=0
  for e,num in enumerate(counts[s]):mask[start:start+int(accepted[s,e])]=True;start+=num
  assert np.array_equal(restored[s][mask],rows[s][mask]);assert np.isnan(restored[s][~mask]).all()
 return {'logical_chunk_capacity':logical,'physical_chunk_capacity':physical,'demand':int(c.sum()),'total_logical_capacity':logical*chunks*nshards,'accepted':int(accepted.sum()),'dropped':int(c.sum()-accepted.sum()),'accepted_by_sender':accepted.sum(axis=1).tolist(),'accepted_by_semantic_expert':semantic_accepted.tolist(),'chunks':records,'accepted_identity_roundtrip':True,'actual_collective':None}
def main():
 checks=[]
 def check(n,c):assert c,n;checks.append(n)
 counts=np.array([[4,1,3,0,2,4,0,1],[2,3,0,2,4,1,3,2]])
 base=run(counts,3,5,list(range(8)));room=run(counts,8,10,list(range(8)));perm=[1,0,3,2,5,4,7,6];swap=run(counts[:,perm],3,5,perm)
 for name,o in [('low_capacity',base),('aggregate_room',room),('expert_relabeling',swap)]:
  check(name+' original plans pack active prefix and put padding only last',all(c['physical_group_sizes'][:-1]==c['active_group_sizes'][:-1] for c in o['chunks']))
  check(name+' host replay roundtrips exactly accepted original identity prefixes',o['accepted_identity_roundtrip'])
 check('Earlier experts/senders bias visible at low capacity',base['accepted_by_semantic_expert']==[3,0,3,0,3,0,3,0] and base['accepted_by_sender']==[8,4])
 check('Aggregate logical capacity equal to demand does not prevent local drops',room['total_logical_capacity']==room['demand']==32 and room['dropped']==5)
 check('Relabeling preserves demand/capacity/count but changes accepted identities',swap['demand']==base['demand'] and swap['accepted']==base['accepted'] and swap['accepted_by_semantic_expert']!=base['accepted_by_semantic_expert'])
 check('Relabeling changes sender acceptance without changing global count',swap['accepted_by_sender']==[6,6])
 o={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'numpy':np.__version__},'observations':{'low_capacity':base,'aggregate_room':room,'expert_relabeling':swap},'explicit_substitutions':['two synthetic shards, eight expert demand counts and integer identity rows','original plan functions called via structural namespace rather than full routing/layout constructors','host copies execute original offset/size metadata; no ragged_all_to_all','fixed logical/physical capacities, no production capacity-factor planner'],'actual_Hero_acceptance':None,'actual_GPU_execution':None,'actual_domain_drop_bias':None,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [E,P]}}
 (R/'analysis/receiver_layout_cpu.json').write_text(json.dumps(o,indent=2)+'\n');print('Original plan / host replay checks:',len(checks))
 for k,v in o['observations'].items():print(k,{x:v[x] for x in ['demand','accepted','dropped','accepted_by_sender','accepted_by_semantic_expert']})
if __name__=='__main__':main()
