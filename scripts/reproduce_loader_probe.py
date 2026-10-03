# -*- coding: utf-8 -*-
"""Run only the two inspected arithmetic methods from the archived loader; no training/JAX."""
import ast,json,pathlib,sys,warnings,hashlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
source=ROOT/'sources/deepening_2026_10_04/mixture_production.py'
tree=ast.parse(source.read_text());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MixtureDataset')
methods=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in ['_normalize_weights','_compute_expected_counts_per_block']]
assert len(methods)==2
module=ast.Module(body=[ast.ClassDef(name='CountProbe',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
ns={'np':np,'warnings':warnings};exec(compile(ast.fix_missing_locations(module),str(source),'exec'),ns)
probe=ns['CountProbe']();data=json.loads((ROOT/'sources/wandb/hero-fa4sm100-nomask-step146k_meta.json').read_text())['config']['data']['value']
probe.dataset_index=[k for k in data['components'] if any(w.get(k,0)>0 for _,w in data['train_weights'])];probe.datasets={k:None for k in probe.dataset_index}
rows=[]
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for K in [49152,12288]:
        for phase,(_,w) in enumerate(data['train_weights']):
            normalized=probe._normalize_weights(w);counts=probe._compute_expected_counts_per_block(normalized,K)
            remainder=K-sum(int(normalized.get(k,0)*K) for k in probe.dataset_index)
            rows.append({'phase':phase,'block_size':K,'normalization_total':sum(w.values()),'remainder':remainder,'counts':dict(zip(probe.dataset_index,map(int,counts)))})
result={'python_version':sys.version,'numpy_version':np.__version__,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'methods':['MixtureDataset._normalize_weights','MixtureDataset._compute_expected_counts_per_block'],'rows':rows}
out=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'analysis/loader_probe_python312.json';out.write_text(json.dumps(result,indent=2))
print('Python',sys.version.split()[0],[(r['block_size'],r['phase'],r['remainder']) for r in rows])
