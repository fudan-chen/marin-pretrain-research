"""Original block quota and real JAX permutation on same-head Harrier weights.
Counts logical sequence assignments, not real document/token reads or effective loss mass.
"""
import ast,csv,hashlib,json,pathlib,types,warnings
import jax,numpy as np
import probe_phase_budget as base
R=base.R;P=R/'sources/integer_exposure_2026_10_07/mixture.py';ns={'jax':jax,'np':np,'warnings':warnings}
tree=ast.parse(P.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MixtureDataset');names=['_normalize_weights','_compute_expected_counts_per_block','_compute_unpermuted_ids'];nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]+[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_compute_block_assignment'];assert len(nodes)==4
for n in nodes:n.decorator_list=[]
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(P),'exec'),ns)
def main():
 checks=[]
 def ck(n,c):assert c,n;checks.append(n)
 ck('Fixed CPU permutation backend',jax.default_backend()=='cpu' and jax.__version__=='0.7.2')
 cfg=base.ns['harrier_mix_2026_08_18_data_config'](ctx=base.ctx,total_steps=390251,batch_size=11264,max_seq_len=4096,experiment_flops=2.7e24,validation=[])
 keys=list(base.raw['available_tokens']);dummy=types.SimpleNamespace(dataset_index=keys,datasets={k:None for k in keys});block=49152;total=390251*11264;quota=[];stage=[];complete=np.zeros(200,np.int64);continuous=np.zeros(200,np.float64);last_ids=None
 for i,(start,weights) in enumerate(cfg.train_weights):
  w=ns['_normalize_weights'](weights);offset=start*11264;end=cfg.train_weights[i+1][0]*11264 if i+1<len(cfg.train_weights) else total;duration=end-offset;full,tail=divmod(duration,block)
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always');q=ns['_compute_expected_counts_per_block'](dummy,w,block)
  ids=ns['_compute_unpermuted_ids'](dummy,q);ck('Stage '+str(i)+' block quota conserved',int(q.sum())==block and len(ids)==block)
  floor=np.array([int(w.get(k,0)*block) for k in keys]);winner=int(np.argmax(floor));remainder=int(block-floor.sum());ck('Stage '+str(i)+' original remainder concentrated at largest count',np.array_equal(q-floor,np.eye(1,200,winner,dtype=np.int64)[0]*remainder))
  ck('Stage '+str(i)+' encoded IDs have exact quota',np.array_equal(np.bincount(ids>>16,minlength=200),q))
  quota.append(q.tolist());complete+=q.astype(np.int64)*full;continuous+=np.array([w.get(k,0)*duration for k in keys]);stage.append({'phase_index':i,'start_step':start,'full_blocks':full,'tail_sequences':tail,'positive_weight_zero_quota':[k for k,n in zip(keys,q) if w.get(k,0)>0 and n==0],'warning_count':len(ws),'remainder':remainder,'remainder_recipient':keys[winner]});last_ids=ids
 ck('Only final phase has partial block',all(s['tail_sequences']==0 for s in stage[:-1]) and stage[-1]['tail_sequences']>0)
 tail=stage[-1]['tail_sequences'];lastblock=total//block;tail_rows=[];totals=[]
 for seed in [7,8]:
  perm=np.asarray(ns['_compute_block_assignment'](last_ids,lastblock,jax.random.PRNGKey(seed)));counts=np.bincount((perm[:tail]>>16).astype(np.int64),minlength=200);fullcounts=np.bincount((perm>>16).astype(np.int64),minlength=200)
  ck('Seed '+str(seed)+' full permutation preserves quota',np.array_equal(fullcounts,quota[-1]))
  ck('Seed '+str(seed)+' tail length and bounds exact',int(counts.sum())==tail and np.all(counts<=quota[-1]))
  totals.append(complete+counts);tail_rows.append({'artificial_seed':seed,'last_block_id':lastblock,'tail_sequences':tail,'tail_counts':dict(zip(keys,map(int,counts))),'packed_id_sha256':hashlib.sha256(perm.tobytes()).hexdigest()})
 ck('Final integer assignment total exact for both seeds',all(int(t.sum())==total for t in totals))
 ck('Partial tail quotas differ across artificial keys',not np.array_equal(totals[0],totals[1]))
 inv=np.array([base.raw['available_tokens'][k] for k in keys]);epochs=totals[0]*4096/inv;nominal=continuous*4096/inv;rows=[]
 for j,k in enumerate(keys):rows.append({'cell':k,'continuous_sequence_budget':float(continuous[j]),'full_block_sequences':int(complete[j]),'exact_sequence_budget_seed7':int(totals[0][j]),'exact_sequence_budget_seed8':int(totals[1][j]),'continuous_full_token_epochs':float(nominal[j]),'integer_full_token_epochs_seed7':float(epochs[j]),'block_counts':[q[j] for q in quota]})
 ck('Positive continuous budget can have zero integer assignment',any(x['continuous_sequence_budget']>0 and x['exact_sequence_budget_seed7']==0 for x in rows))
 out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'numpy':np.__version__,'backend':jax.default_backend()},'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,base.H,base.L,base.S,R/'scripts/probe_phase_budget.py']},'stages':stage,'tail_controls':tail_rows,'cells':rows,'largest_integer_full_token_epoch_cell':keys[int(np.argmax(epochs))],'largest_integer_full_token_epochs_seed7':float(epochs.max()),'largest_continuous_epoch_cell':keys[int(np.argmax(nominal))],'largest_continuous_epochs':float(nominal.max()),'total_logical_sequences':total,'explicit_substitutions':['Original quota/ID/permutation functions, not full MixtureDataset or DataLoader','Declared 200-key insertion order; actual Hero dataset order unknown','Two artificial keys, no production data seed attribution','Python/NumPy int64 external accumulation avoids source int32 overflow; original per-block quota dtype kept','Full sequence length multiplication is nominal token mass; no documents, masks, cached contents or effective loss targets'], 'actual_Hero_dataset_key_order':None,'actual_Hero_logical_assignment_replay':None,'actual_Hero_token_exposure':None,'actual_Hero_loss_effect':None,'actual_GPU_execution':None}
 (R/'analysis/integer_exposure_cpu.json').write_text(json.dumps(out,indent=2)+'\n')
 with (R/'analysis/integer_exposure_cells.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 print(json.dumps({'checks_passed':len(checks),'stages':stage,'selected':[x for x in rows if x['cell'] in ['c27q0','c22q0','c30q4']]},indent=2))
if __name__=='__main__':main()
