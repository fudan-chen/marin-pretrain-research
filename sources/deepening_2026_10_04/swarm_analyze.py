import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent
rows=pq.read_table(ROOT/'observations.parquet').to_pylist()
old=json.loads((ROOT/'old_weights.json').read_text())
new=json.loads((ROOT/'new_weights.json').read_text())
def matches(row,weights):
 return max(abs(row[p][k]-weights[p][k]) for p in ['phase0_weights','phase1_weights'] for k in weights[p])<2e-8
selected={label:[r for r in rows if matches(r,w)] for label,w in [('old',old),('new',new)]}
assert all(len(v)==3 for v in selected.values())
(ROOT/'selected_runs.json').write_text(json.dumps(selected,indent=2))
fits=pd.read_csv(ROOT/'fits_all_fractions.csv').query('cohort == "baseline" and percentile == 100')
alpha_macro=json.loads((ROOT/'macro_speedups.json').read_text())['alpha']
results=[]
for key in sorted(selected['old'][0]['training_eval_metrics']):
 if not key.startswith('eval_dropless/') or not key.endswith('/bpb') and not key.endswith('macro_bpb'):continue
 values={label:np.array([r['training_eval_metrics'][key] for r in group],dtype=float) for label,group in selected.items()}
 if not all(np.isfinite(v).all() for v in values.values()):continue
 a,b=(values[k].mean() for k in ['old','new'])
 subset=key.split('/')[2].removesuffix('-llama3')
 if key=='eval_dropless/paloma/macro_bpb':alpha=alpha_macro
 elif key.startswith('eval_dropless/paloma/') and subset not in ['bpb','macro_bpb']:
  alpha=float(fits.query('subset == @subset and metric == "bpb"').iloc[0].alpha)
 else:alpha=np.nan
 results.append(dict(metric=key,old_mean=a,new_mean=b,change_pct=100*(b/a-1),alpha=alpha,compute_speedup=(a/b)**(1/alpha) if np.isfinite(alpha) else np.nan,old_std=values['old'].std(ddof=1),new_std=values['new'].std(ddof=1)))
df=pd.DataFrame(results)
df.to_csv(ROOT/'comparison.csv',index=False)
for r in results:
 if any(x in r['metric'] for x in ['macro_bpb','programing','s2orc','refinedweb','dolma-v1_5','github_','arxiv_']):print(r)
