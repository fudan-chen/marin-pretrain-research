"""Execute only the archived two loader methods against prospective weight arms."""
import ast,json,pathlib,sys,hashlib,warnings
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];source=R/'sources/deepening_2026_10_04/mixture_production.py'
tree=ast.parse(source.read_text());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MixtureDataset')
methods=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in ['_normalize_weights','_compute_expected_counts_per_block']]
assert len(methods)==2
module=ast.Module(body=[ast.ClassDef(name='Probe',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
ns={'np':np,'warnings':warnings};exec(compile(ast.fix_missing_locations(module),str(source),'exec'),ns)
probe=ns['Probe']();data=json.loads((R/'analysis/transfer_data.json').read_text());probe.dataset_index=sorted(data['base_weights'][0]);probe.datasets={k:None for k in probe.dataset_index}
results=[];originals=[]
with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 for plan in data['plans']:
  for arm in plan['arms']:
   for variant,weights,targets,output in [('original_continuous',arm['weights'],arm['full_block_audit']['phase_counts'],originals),
                                         ('integer_controlled',arm['integer_controlled_weights'],arm['integer_repair']['target_phase_counts'],results)]:
    for phase,w in enumerate(weights):
     counts=probe._compute_expected_counts_per_block(probe._normalize_weights(w),49152)
     actual=dict(zip(probe.dataset_index,map(int,counts)));expected=targets[phase]
     assert actual==expected,(plan['donor_domain'],arm['id'],phase,variant)
     output.append({'donor':plan['donor_domain'],'arm':arm['id'],'phase':phase,'counts':actual,'matches_target':True})
inputs=[{'donor':p['donor_domain'],'arms':[{'id':a['id'],'original':a['weights'],'integer_controlled':a['integer_controlled_weights']} for a in p['arms']]} for p in data['plans']]
result={'python_version':sys.version,'numpy_version':np.__version__,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'input_weights_sha256':hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'source_methods':['MixtureDataset._normalize_weights','MixtureDataset._compute_expected_counts_per_block'],
        'scope':'24 repaired and 24 original full-block count evaluations; no trainer, token store, cursor or inner shuffle execution', 'rows':results,'original_rows':originals}
(R/'analysis/transfer_loader_probe_python312.json').write_text(json.dumps(result,indent=2)+'\n');print('Transfer native-loader counts:',len(results),'repaired and',len(originals),'original matched under Python',sys.version.split()[0])
