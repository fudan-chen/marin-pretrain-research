"""Original 49,152-sequence block permutation and logical IDs at recorded d512 resume boundary.
Conditional replay with supplied keys and finite identity children, not historical token replay.
"""
import asyncio,collections,csv,dataclasses,hashlib,json,pathlib,warnings
import jax,numpy as np
import probe_mixture_identity_cpu as mix
import probe_loader_resume as loader
R=pathlib.Path(__file__).resolve().parents[1];P=R/'sources/resume_boundary_2026_10_08/producer_meta.json';producer=json.loads(P.read_text())['data']['project']['run'];pc=json.loads(producer['config']);ps=json.loads(producer['summaryMetrics']);pd=pc['data']['value'];B=1024;L=4096;K=49152;RESUME=393;START=384;END=432;offset=RESUME*B;block=offset//K;cut=offset%K;checks=[];files={pathlib.Path(__file__),P,mix.P,loader.SP,R/'scripts/probe_mixture_identity_cpu.py',R/'scripts/probe_loader_resume.py',R/'analysis/domain_ablation_audit.csv'}
def check(n,v):assert v,n;checks.append({'name':n,'passed':True})
# Execute original schedule-to-example rescaling function.
env={'BatchSchedule':loader.Schedule};mix_nodes=__import__('ast').parse(mix.P.read_text()).body;ast=__import__('ast');node=next(n for n in mix_nodes if isinstance(n,ast.FunctionDef) and n.name=='rescale_mixture_schedule_for_batch_schedule');exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])),str(mix.P),'exec'),env)
schedule=loader.Schedule(B)
def build(data,key=7,weights=None):
 order=[n for n in data['components'] if n.startswith('c')];stages=env['rescale_mixture_schedule_for_batch_schedule'](weights if weights is not None else data['train_weights'],schedule)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',UserWarning);return mix.Mix({n:mix.Identity(n,10**12) for n in order},stages,K,key=key)
def load(name):
 p=R/'sources/practical_2026_10_04'/('config_'+name+'.json');files.add(p);return json.loads(json.loads(p.read_text())['data']['project']['run']['config'])['data']['value']
def domain_counts(ids):return np.bincount(np.asarray(ids,dtype=np.int64)>>16,minlength=200)
def compare(m,old,group,domain,key=7):
 ob=np.asarray(old._get_block(block));nb=np.asarray(m._get_block(block));a=ob[:cut];b=nb[:cut];suffix=nb[cut:];ac=domain_counts(a);bc=domain_counts(b);common=np.intersect1d(a,b);repeated=np.intersect1d(a,suffix);next_repeat=np.intersect1d(a,nb[cut:cut+B]);base=np.asarray(m._counts_after_stage[0]);oldbase=8*np.asarray(old._counts_per_block_per_stage[0]);assert np.array_equal(base,oldbase)
 row={'group':group,'domain':domain,'supplied_mixture_key_seed':key,'producer_prefix_sequences':len(a),'declared_prefix_sequences':len(b),'named_count_l1':int(abs(bc-ac).sum()),'named_count_half_l1':int(abs(bc-ac).sum()//2),'prefix_identity_intersection':len(common),'prefix_unmatched_identities_each_side':cut-len(common),'same_position_identity_count':int(np.count_nonzero(a==b)),'hybrid_block_repeated_logical_ids':len(repeated),'next_batch_repeated_logical_ids':len(next_repeat),'block_start_base_offsets_equal':True,'block_original_prefix_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'block_declared_prefix_sha256':hashlib.sha256(b.tobytes()).hexdigest(),'prefix_count_sum_delta':int((bc-ac).sum())}
 cells=[{'group':group,'domain':domain,'cell':n,'producer_block_prefix_count':int(ac[i]),'declared_block_prefix_count':int(bc[i]),'declared_minus_producer':int(bc[i]-ac[i]),'base_logical_offset':int(base[i])} for i,n in enumerate(m.dataset_index)]
 return row,cells
async def main():
 rows=list(csv.DictReader((R/'analysis/domain_ablation_audit.csv').open()));old=build(pd);results=[];cells=[];examples={};declared_match=[]
 for r in rows:
  data=load(r['run_name']);declared_match.append(data['components']==pd['components'] and data['train_weights'][0][1]==pd['train_weights'][0][1] and data['shuffle']==pd['shuffle'] and data['experiment_budget']==pd['experiment_budget'] and data['target_budget']==pd['target_budget'])
  m=build(data);row,cr=compare(m,old,r['group'],int(r['domain']));row['run_name']=r['run_name'];results.append(row);cells.extend(cr)
  if int(r['domain']) in [0,39]:
   ids=await m.get_batch(list(range(offset,offset+B)));packed=m._get_block(block)[cut:cut+B];expected=[]
   for p in packed:
    d,i=m._index_into_dataset_for_id(int(p),block);expected.append(m.dataset_index[d]+':'+str(i))
   assert ids==expected;examples[r['group']+':c'+r['domain']]={'next_batch_identity_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),'first8':ids[:8],'repeated_ids_in_next_batch':row['next_batch_repeated_logical_ids']}
 check('Producer public final log and nominal token total align with 393 completed updates',ps['_step']==392 and ps['throughput/total_tokens']==RESUME*B*L and producer['state']=='finished')
 check('All eighty records preserve producer components initial weights and simulated inventory settings',len(rows)==80 and all(declared_match))
 check('Producer and continuation schedules differ before declared restore boundary',[s for s,w in pd['train_weights']]==[0,3120] and all([s for s,w in load(r['run_name'])['train_weights']]==[0,384,3120] for r in rows))
 check('Original scheduler places restore nine batches into block eight',schedule.global_data_offset_by_step(RESUME)==offset and block==8 and cut==9216 and K//B==48 and (RESUME-START)*B*L==37748736)
 check('Both source paths share per-domain logical base before block eight',all(r['block_start_base_offsets_equal'] for r in results))
 check('Prefix count changes conserve total sequences independently of identity mismatch',all(r['prefix_count_sum_delta']==0 and r['named_count_l1']%2==0 and 0<=r['prefix_unmatched_identities_each_side']<=cut for r in results))
 check('Four representative original identity-store reads match original logical decoder',len(examples)==4)
 # A no-change schedule has zero count and logical identity mismatch.
 same,_=compare(build(pd,weights=[(0,pd['train_weights'][0][1]),(384,pd['train_weights'][0][1]),(3120,pd['train_weights'][1][1])]),old,'unchanged',-1)
 check('Unchanged-weight staged control preserves prefix and introduces no within-block repeated identities',same['named_count_half_l1']==0 and same['prefix_unmatched_identities_each_side']==0 and same['hybrid_block_repeated_logical_ids']==0)
 representative=load(rows[0]['run_name']);bad_weights=[(0,representative['train_weights'][0][1]),(393,representative['train_weights'][1][1]),(3120,representative['train_weights'][2][1])]
 try:build(representative,weights=bad_weights);unaligned=None
 except AssertionError as exc:unaligned=str(exc)
 check('Moving phase boundary directly to step 393 violates original block-alignment assertion',unaligned is not None and 'multiple of block_size' in unaligned)
 bridge=build(representative,weights=[(0,representative['train_weights'][0][1]),(432,representative['train_weights'][1][1]),(3120,representative['train_weights'][2][1])]);bridge_block=np.asarray(bridge._get_block(8));old_block=np.asarray(old._get_block(8))
 check('Aligned step432 bridge preserves entire old block8 but delays 39 resumed updates',np.array_equal(bridge_block,old_block) and END-RESUME==39 and bridge._get_stage_for_block(8)==0 and bridge._get_stage_for_block(9)==1)
 seed_controls=[]
 for key in [0,11]:
  oo=build(pd,key);mm=build(representative,key);rr,_=compare(mm,oo,'proportional_domain_ablation',0,key);seed_controls.append(rr)
 check('Supplied-key controls preserve geometric boundary without claiming actual historical key',len(seed_controls)==2 and all(r['block_start_base_offsets_equal'] and r['producer_prefix_sequences']==9216 for r in seed_controls))
 with (R/'analysis/ablation_resume_boundary_cells.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(cells[0]),lineterminator='\n');w.writeheader();w.writerows(cells)
 with (R/'analysis/ablation_resume_boundary_runs.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]),lineterminator='\n');w.writeheader();w.writerows(results)
 prefix=np.asarray(old._get_block(block)[:cut],dtype=np.int64);di=int(np.argmax(domain_counts(prefix)));local=np.sort(prefix[(prefix>>16)==di]&0xFFFF)
 partial_population={'cell':old.dataset_index[di],'observed_count':len(local),'minimum_local_id':int(local.min()),'maximum_local_id':int(local.max()),'holes_below_maximum':int(local.max()+1-len(local)),'first20_sorted_local_ids':local[:20].tolist(),'block_quota':int(old._counts_per_block_per_stage[0][di])}
 check('Partial shuffled block is not a dense per-cell prefix recoverable from count alone',partial_population['holes_below_maximum']>0 and not np.array_equal(local,np.arange(len(local))))
 summary={k:{'min':min(r[k] for r in results),'max':max(r[k] for r in results),'nonzero_runs':sum(r[k]!=0 for r in results)} for k in ['named_count_half_l1','prefix_unmatched_identities_each_side','hybrid_block_repeated_logical_ids','next_batch_repeated_logical_ids']}
 out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'run_count':80,'cell_rows':len(cells),'geometry':{'batch_size':B,'sequence_length':L,'block_sequences':K,'step384_global_sequence':START*B,'step393_global_sequence':offset,'block_index':block,'cut_sequences':cut,'nominal_positions_in_nine_steps':37748736,'next_aligned_step':432,'bridge_resumed_updates':39},'producer':{'name':producer['name'],'state':producer['state'],'summary_step':ps['_step'],'summary_nominal_tokens':ps['throughput/total_tokens'],'declared_phase_starts':[s for s,w in pd['train_weights']]},'partial_block_population':partial_population,'summary':summary,'cases':results,'identity_examples':examples,'unchanged_weights_control':same,'unaligned_step393_error':unaligned,'alternate_supplied_key_controls':seed_controls,'runtime':{'jax':jax.__version__,'numpy':np.__version__,'backend':jax.default_backend()},'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},'substitutions':['AST original mixture class and block permutation plus original BatchSchedule and stage rescaling; full package not imported','Finite identity children length1e12 prevent modulo wrap; no tokenizer, child shuffle, simulation slice or actual token read','Supplied mixture keys 7,0,11; historical launch key and code SHA not recovered','Producer-declared and continuation-declared schedules replayed; actual selected checkpoint metadata/state and actual loader execution not read'],'actual_checkpoint_contents_verified':None,'actual_historical_mixture_key':None,'actual_token_repetition_count':None,'actual_training_loss_effect':None,'actual_production_incident':None,'actual_bridge_training_result':None}
 (R/'analysis/ablation_resume_boundary.json').write_text(json.dumps(out,indent=2)+'\n');print('Resume-boundary source controls',len(checks));print(json.dumps(summary))
asyncio.run(main())
