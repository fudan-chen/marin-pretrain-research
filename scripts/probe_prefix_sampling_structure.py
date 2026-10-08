"""Original historical block-shuffle prefix structure and explicit uniform-without-replacement reference.
Synthetic sequence IDs only; IO blocks are index groups, not assumed documents or source groups.
"""
import asyncio,collections,hashlib,itertools,json,math,pathlib,sys
import jax,numpy as np
R=pathlib.Path(__file__).resolve().parents[1];A=R/'analysis';P=R/'scripts/probe_simulated_inventory_cpu.py';marker="jax.config.update('jax_threefry_partitionable',True);cases={};objects={}"
s=P.read_text();assert s.count(marker)==1;env={'__file__':str(P),'__name__':'sampling_dependency_adapters'};exec(compile(s.split(marker)[0],str(P)+'[definitions-only]','exec'),env)
Block=env['inner'].ns['BlockShufflingDataset'];Identity=env['inner'].Identity;checks=[]
def check(n,v):assert v,n;checks.append({'name':n,'passed':True})
def reference(n,k,b):
 full,tail=divmod(n,b);sizes=[(b,full)]+([(tail,1)] if tail else [])
 pair_p=sum(mult*s*(s-1) for s,mult in sizes)/(n*(n-1)) if n>1 else 0
 def nonempty_prob(size):
  if k>n-size:return 1.
  return -math.expm1(sum(math.log1p(-size/(n-i)) for i in range(k))) if k else 0.
 return {'full_sequences':n,'selected_sequences':k,'io_block_size':b,'same_io_block_pair_probability':pair_p,'expected_same_io_block_pairs':k*(k-1)/2*pair_p,'expected_distinct_io_blocks':sum(mult*nonempty_prob(s) for s,mult in sizes),'expected_tail_sequences':k*tail/n}
def stats(ids,io,n):
 h=collections.Counter(int(x)//io for x in ids);tail=n%io;f=n-tail
 return {'selected_sequences':len(ids),'unique_sequences':len(set(ids)),'distinct_io_blocks':len(h),'max_sequences_per_io_block':max(h.values(),default=0),'same_io_block_pairs':sum(v*(v-1)//2 for v in h.values()),'tail_sequence_hits':sum(int(x)>=f for x in ids) if tail else 0,'io_block_histogram':{str(k):v for k,v in sorted(h.items())}}
def run(coro):return asyncio.run(coro)
jax.config.update('jax_threefry_partitionable',True);old=json.loads((A/'simulated_inventory_cpu.json').read_text());n=1000000;k=879;io=256;w=512;results=[]
for case in ['realratio_default','realratio_zero']:
 for cell,strings in old['cases'][case]['pools'].items():
  key=np.asarray(old['cases'][case]['child_keys'][cell],dtype=np.uint32);b=Block(Identity(n),io,window_blocks=w,key=key,perm_type='feistel');ids=run(b.get_batch(list(range(k))));expected=[int(x.split(':')[1]) for x in strings];state=b._state_or_error();layout=b._window_layout(0);row=stats(ids,io,n);row.update({'case':case,'cell':cell,'window_eligible_full_blocks':len(layout.full_blocks),'window_eligible_sequences':layout.full_region_size,'all_selected_io_blocks_in_first_window':set(map(int,row['io_block_histogram'])).issubset(set(layout.full_blocks)),'prefix_reproduces_previous_ids':ids==expected,'layout_full_blocks':list(layout.full_blocks),'selected_identity_sha256':hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()});results.append(row)
check('Four historical-source prefixes reproduce all prior879synthetic IDs',len(results)==4 and all(r['prefix_reproduces_previous_ids'] and r['unique_sequences']==879 for r in results))
check('Each879-prefix stays within first512physical full blocks and131072eligible sequences',all(r['window_eligible_full_blocks']==512 and r['window_eligible_sequences']==131072 and r['all_selected_io_blocks_in_first_window'] and r['distinct_io_blocks']<=512 and r['tail_sequence_hits']==0 for r in results))
uniform_support_counterexample=[b*io for b in range(k)];check('Uniform879-subset support includes explicit879-block set impossible for this first512-block window',len(set(uniform_support_counterexample))==k and max(uniform_support_counterexample)<n and len({i//io for i in uniform_support_counterexample})==879>512)
ref=reference(n,k,io);check('Observed synthetic block clustering exceeds explicit uniform-sample reference without significance claim',all(r['same_io_block_pairs']>ref['expected_same_io_block_pairs'] and r['distinct_io_blocks']<ref['expected_distinct_io_blocks'] for r in results))
# Exact enumeration validates both reference expectations on a different, small domain.
combinations=list(itertools.combinations(range(10),3));actual=[stats(c,4,10) for c in combinations];tiny=reference(10,3,4)
check('Analytic uniform-subset pair and distinct-block references match all120tiny combinations',len(combinations)==120 and abs(sum(r['same_io_block_pairs'] for r in actual)/120-tiny['expected_same_io_block_pairs'])<1e-12 and abs(sum(r['distinct_io_blocks'] for r in actual)/120-tiny['expected_distinct_io_blocks'])<1e-12)
# Tail at output end, not a finite-key estimate of sampling probability.
tails=[]
for length in [1,255,256,257,1137,1138,1279,1280,131071,131072,131073,1000000]:
 for seed in [0,7]:
  b=Block(Identity(length),256,window_blocks=512,key=seed);tail=length%256;full=length-tail;positions=sorted(set(i for i in [0,full-1,full,length-1] if 0<=i<length));mapped=run(b.get_batch(positions));cut=int(length*old['declared_simulation_ratio']);pref=run(b.get_batch(list(range(cut))));tails.append({'length':length,'supplied_child_seed':seed,'full_region':full,'tail_size':tail,'prefix_length':cut,'tail_start_output_position':full,'boundary_output_positions':positions,'boundary_mapped_ids':mapped,'tail_region_membership_preserved':all((p>=full)==(i>=full) for p,i in zip(positions,mapped)),'prefix_tail_hits':sum(i>=full for i in pref) if tail else 0})
check('Twenty-four boundary controls preserve final tiny tail location with no sampled tail at recorded ratio',len(tails)==24 and all(t['tail_region_membership_preserved'] and t['prefix_tail_hits']==0 for t in tails))
small=Block(Identity(22),4,window_blocks=3,key=7);all22=run(small.get_batch(list(range(22))));cut_controls=[{'cut':cut,'tail_hits':sum(i>=20 for i in all22[:cut]),'predicted_tail_hits':max(0,cut-20)} for cut in [0,5,20,21,22]]
check('Full small inventory and five prefix endpoints obey exact tail-count formula',sorted(all22)==list(range(22)) and all(c['tail_hits']==c['predicted_tail_hits'] for c in cut_controls))
# Window-size interventions: same child key and inventory, no performance measurements.
key=np.asarray(old['cases']['realratio_default']['child_keys']['c00q0'],dtype=np.uint32);windows=[]
for width in [1,64,512,4096]:
 b=Block(Identity(n),io,window_blocks=width,key=key);ids=run(b.get_batch(list(range(k))));row=stats(ids,io,n);row.update({'window_blocks':width,'identity_sha256':hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()});windows.append(row)
check('Four window-size controls retain unique879IDs but change selected membership digest',len({r['identity_sha256'] for r in windows})==4 and all(r['unique_sequences']==879 and r['tail_sequence_hits']==0 for r in windows) and windows[0]['distinct_io_blocks']==4 and windows[1]['distinct_io_blocks']<=64)
# Bound tail exclusion: fraction of raw inventory, not model quality or token loss.
threshold=old['empty_inventory_control']['first_nonempty_full_sequences'];candidates=range(threshold,threshold+io);worst=max(candidates,key=lambda x:(x%io)/x);tail_bound={'first_nonempty_sequences':threshold,'io_block_size':io,'max_tail_fraction_over_nonempty_lengths':(worst%io)/worst,'maximizing_length':worst,'maximizing_tail':worst%io,'million_identity_tail_fraction':64/1000000,'reason':'For each fixed remainder, fraction decreases as length increases by256; first256nonempty lengths cover every remainder.'}
check('Tail-fraction arithmetic bound distinguishes rare synthetic small bucket from million-sized inventory',worst==1279 and worst%io==255 and tail_bound['million_identity_tail_fraction']==0.000064)
files=[pathlib.Path(__file__),P,A/'simulated_inventory_cpu.json',R/'sources/simulated_inventory_2026_10_08/dataset.py',R/'sources/simulated_inventory_2026_10_08/_prp.py']+[R/p for p in old['source_sha256']]
out={'scope':__doc__,'checks_passed':len(checks),'checks':checks,'runtime':{'jax':jax.__version__,'jaxlib':__import__('jaxlib').__version__,'numpy':np.__version__,'backend':jax.default_backend()},'four_prefixes':results,'uniform_without_replacement_reference':ref,'structural_support_counterexample':{'selected_sequences':879,'distinct_io_blocks':879,'block_prefix_cap':512,'identity_formula':'i=256*b for b=0..878','valid_subset_of_million':True,'uniform_subset_probability':'1 / binomial(1000000,879), strictly positive','loader_prefix_probability_under_contract':0},'tiny_exact_enumeration':{'combinations':120,'reference':tiny},'tail_boundary_controls':tails,'small_tail_cut_controls':cut_controls,'window_interventions':windows,'tail_fraction_bound':tail_bound,'source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))},'interpretation_limits':['Four fixed-key synthetic prefixes, not random-seed confidence intervals or uniformity hypothesis test','Uniform-without-replacement reference is explicit mathematical comparator, not claim that global Feistel yields all subsets equally often','IO block index groups not identified documents, source labels or quality groups','Window changes are unexecuted training/performance candidates; changed sample set is an intervention','Historical token inventories, actual tail composition and actual sampling/loss effects unverified'],'actual_Hero_prefix_representativeness':None,'actual_tail_sequence_contents':None,'actual_domain_quality_bias':None,'actual_performance_effect':None,'actual_training_loss_effect':None}
(A/'prefix_sampling_structure.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Prefix structure controls',len(checks));print('Reference',ref);print('Observed',[(r['case'],r['cell'],r['distinct_io_blocks'],r['same_io_block_pairs']) for r in results]);print('Windows',[(r['window_blocks'],r['distinct_io_blocks'],r['same_io_block_pairs']) for r in windows]);print('Tail bound',tail_bound)
