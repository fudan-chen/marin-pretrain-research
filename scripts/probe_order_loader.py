"""Execute six archived methods; no full trainer, JAX shuffle or token-store reads."""
import ast,json,pathlib,hashlib,sys,warnings
import numpy as np
from order_core import compile_order
R=pathlib.Path(__file__).resolve().parents[1];source=R/'sources/deepening_2026_10_04/mixture_production.py'
tree=ast.parse(source.read_text());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MixtureDataset')
names=['_normalize_weights','_compute_expected_counts_per_block','_compute_unpermuted_ids','_initialize_stage_counts','_get_stage_for_block','_index_into_dataset_for_id']
methods=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(methods)==6
module=ast.Module(body=[ast.ClassDef(name='Probe',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
ns={'np':np,'warnings':warnings};exec(compile(ast.fix_missing_locations(module),str(source),'exec'),ns)
data=json.loads((R/'analysis/order_workbench_data.json').read_text());K=49152;cells=sorted(data['initial_counts'])
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def init(stages):
    p=ns['Probe']();p.datasets={k:None for k in cells};p.dataset_index=cells;p.block_size=K
    p.weight_stages=[(s['sequence_begin'],p._normalize_weights(s['weights'])) for s in stages]
    p._counts_per_block_per_stage,p._counts_after_stage,p._unpermuted_ids_per_stage=p._initialize_stage_counts()
    p._stage_start_blocks=np.array([s//K for s,_ in p.weight_stages],dtype=np.int64)
    return p
historical=[];compiled=[];proofs=[];debug=[];inputs=[]
with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 for run,r in data['historical_runs'].items():
    for phase,w in enumerate(r['phase_weights']):
        p=ns['Probe']();p.dataset_index=cells;p.datasets={k:None for k in cells}
        actual=dict(zip(cells,map(int,p._compute_expected_counts_per_block(p._normalize_weights(w),K))))
        assert actual==r['expected_counts'][phase],(run,phase)
        historical.append({'run':run,'phase':phase,'counts':actual,'matches_local_arithmetic':True})
 for k in range(data['locked_default']['capacity']['symmetric_max_k']+1):
  for lock in [True,False]:
    plan=compile_order(data['baseline_counts'],data['initial_counts'],K,k,lock_edges=lock);ps=[]
    inputs.append({'k':k,'lock_edges':lock,'arms':[{'id':a['id'],'stages':a['stages']} for a in plan['arms']]})
    for arm in plan['arms']:
      p=init(arm['stages']);ps.append(p)
      for phase,stage in enumerate(arm['stages']):
        actual=dict(zip(cells,map(int,p._counts_per_block_per_stage[phase])))
        assert actual==stage['counts'],(k,lock,arm['id'],phase)
        compiled.append({'k':k,'lock_edges':lock,'arm':arm['id'],'stage':phase,'count_sha256':digest(actual),'matches_target':True})
    if lock:
      common=[]
      for block in [8,81]:
        arrays=[p._unpermuted_ids_per_stage[p._get_stage_for_block(block)] for p in ps]
        assert all(np.array_equal(arrays[0],x) for x in arrays)
        cursors=[[p._index_into_dataset_for_id(i<<16,block)[1] for i in range(len(cells))] for p in ps]
        assert cursors[0]==cursors[1]==cursors[2],(k,block)
        common.append({'block':block,'unpermuted_ids_sha256':hashlib.sha256(arrays[0].tobytes()).hexdigest(),'cursor_sha256':digest(cursors[0]),'arrays_and_cursors_identical':True})
      assert all(all(v==0 for v in a['interior_cumulative_difference_sequences'].values()) for a in plan['arms'])
      proofs.append({'k':k,'edge_blocks':common,'interior_counts_exact':True,'scope':'Conditional logical-index multiset proof; equal keys/mapping/history required; actual token stream unverified'})
    elif k==10:
      totals=[]
      for p,arm in zip(ps,plan['arms']):
        n=np.array([sum((56*arm['interior_counts'][0][cell],16*arm['interior_counts'][1][cell])) for cell in cells],dtype=np.int64)
        for block,a,b in [(8,9216,K),(81,0,43008)]:
          ids=p._unpermuted_ids_per_stage[p._get_stage_for_block(block)][a:b]
          n+=np.bincount(ids>>16,minlength=len(cells))
        assert int(n.sum())==4024320-402432;totals.append(n)
      for i,arm in enumerate(plan['arms']):
        delta=dict(zip(cells,map(int,totals[i]-totals[0])))
        debug.append({'arm':arm['id'],'k':k,'l1_difference_sequences':sum(abs(v) for v in delta.values()),'difference_sequences':delta,
                      'scope':'Archived unpermuted IDs, equivalent to randomize_blocks=False diagnostic; not historical shuffled read counts'})
result={'python_version':sys.version,'numpy_version':np.__version__,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'input_recipes_sha256':digest(inputs),'source_methods':['MixtureDataset.'+n for n in names],
        'historical_rows':historical,'compiled_rows':compiled,'edge_proofs':proofs,'unlocked_debug_k10':debug,
        'scope':'48 historical and 480 compiled stage-count evaluations plus 20 conditional edge/index proofs. No JAX permutation, checkpoint, actual token store or training.'}
(R/'analysis/order_loader_probe_python312.json').write_text(json.dumps(result,indent=2)+'\n')
print('Order native methods:',len(historical),'historical counts,',len(compiled),'compiled counts,',len(proofs),'conditional edge proofs.')
