# -*- coding: utf-8 -*-
"""Freeze observed endpoint inputs and a bounded selector provenance audit."""
import pathlib, json, re, hashlib, csv, math
import pyarrow.parquet as pq
R = pathlib.Path(__file__).resolve().parents[1]
S = R / 'sources'; A = R / 'analysis'; D = S / 'decision_2026_10_04'
HFREV = '75c25718e2a7cf7a3b1b498b00479351f80345e0'
rows = pq.read_table(S / 'deepening_2026_10_04/hf_observations.parquet').to_pylist()
selected = json.loads((S / 'mix_study_swarm-2026.09.14.1_selected_runs.json').read_text())
metrics = [
    ('paloma', 'Paloma macro', 'eval_dropless/paloma/macro_bpb'),
    ('humaneval', 'HumanEval目标文本', 'logprob_humaneval_10shot'),
    ('gsm8k', 'GSM8K目标文本', 'logprob_gsm8k_5shot'),
    ('medqa', 'MedQA目标文本', 'medqa_0shot'),
    ('belebele', 'Belebele语言平均', 'belebele_mean'),
    ('include', 'Include语言平均', 'include_mean'),
    ('programming', '100种编程语言', 'eval_dropless/paloma/dolma_100_programing_languages-llama3/bpb'),
]
def seed(r): return int(re.search(r'seed(\d+)', r['run_name'])[1])
def vals(r):
    return {name: r['training_eval_metrics' if key.startswith('eval') else 'grouped_bpb'][key] for name, _, key in metrics}
def hash_weights(r):
    return hashlib.sha256(json.dumps([r['phase0_weights'], r['phase1_weights']], sort_keys=True).encode()).hexdigest()
seeds_by_weights = {}
for r in rows:
    seeds_by_weights.setdefault(hash_weights(r), set()).add(seed(r))
candidates = []
selected0 = next(r for r in selected['new'] if seed(r) == 0)
for r in sorted(rows, key=lambda r: r['run_name']):
    if r['group'] != 'mixprior_candidate' or seed(r) != 0: continue
    hash_ = re.search(r'mixprior-([0-9a-f]+)-', r['run_name'])[1]
    v = vals(r)
    assert all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in v.values())
    candidates.append({'id': hash_, 'run': r['run_name'], 'values': v,
                       'observed_seeds': sorted(seeds_by_weights[hash_weights(r)]),
                       'role': 'selected' if r['run_name'] == selected0['run_name'] else 'alternative' if hash_ == '197c9f5ceff6b9ee' else 'candidate'})
assert len(candidates) == 597 and len({r['id'] for r in candidates}) == 597
baselines = {}
for name, label, rs in [
    ('proportional', '库存比例基线 seed0', [r for r in rows if r['group'] == 'proportional_baseline']),
    ('old', '旧Harrier配比 seed0', selected['old']),
    ('selected', '选中996f配比 seed0', selected['new']),
]:
    r = next(r for r in rs if seed(r) == 0)
    baselines[name] = {'label': label, 'run': r['run_name'], 'values': vals(r)}
payload = {'schema': 'marin-observed-decision-input/1', 'evidence_version': HFREV,
           'source': 'sources/deepening_2026_10_04/hf_observations.parquet',
           'source_sha256': hashlib.sha256((S / 'deepening_2026_10_04/hf_observations.parquet').read_bytes()).hexdigest(),
           'metrics': [{'id': name, 'label': label, 'source_key': key, 'unit': 'BPB', 'lower_is_better': True} for name, label, key in metrics],
           'baselines': baselines, 'candidates': candidates,
           'scope': 'registered d512 continuation recipe; restored checkpoint contents not replayed; all ranking uses seed0 only; search observations, not independent confirmation',
           'unknown': ['original selector target and constraints', 'independent candidate confirmation', 'actual generation accuracy', 'target-scale benefit']}
canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
payload['dataset_sha256'] = hashlib.sha256(canonical).hexdigest()
(A / 'decision_data.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
with (A / 'decision_candidates.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['candidate', 'run', 'role', 'observed_seed_labels'] + [x[0] for x in metrics], lineterminator='\n')
    writer.writeheader()
    for r in candidates: writer.writerow(dict(candidate=r['id'], run=r['run'], role=r['role'], observed_seed_labels=';'.join(map(str, r['observed_seeds'])), **r['values']))
tree = json.loads((D / 'marin_tree.json').read_text())
terms = ['selector', 'regmix', 'swarm', 'mixprior', 'mixture_search', 'mixture_opt', 'surrogate']
matches = [r['path'] for r in tree['tree'] if any(t in r['path'].lower() for t in terms)]
queries = []
for name in ['search_mixture_hash.json', 'search_swarm_id.json']:
    d = json.loads((D / name).read_text())
    queries.append({'file': 'sources/decision_2026_10_04/' + name, 'total_count': d['total_count'],
                    'incomplete_results': d['incomplete_results'], 'items_returned': len(d['items']),
                    'issue_numbers': [r['number'] for r in d['items']]})
audit = {'status': 'not_recovered_in_checked_public_entries', 'pinned_marin_revision': tree['sha'],
         'tree_truncated': tree['truncated'], 'filename_search_terms': terms, 'filename_matches': matches,
         'issue_queries': queries, 'issue_9126_body_unchanged_from_prior_archive':
         json.loads((D / 'issue_9126.json').read_text())['body'] == json.loads((S / 'issue_9126.json').read_text())['body'],
         'limitations': ['Filename matching is not a full text search of every blob or branch.',
                        'GitHub issue queries cover matching issue/PR indexed text, not a complete comment search.',
                        'Pinned HF registry fields give observations and budgets, not a selector execution contract.',
                        'This establishes the checked access boundary, not universal absence of public artifacts.'],
         'rubric_R12': 'unassessed; endpoints cannot establish the historical decision criteria'}
(A / 'selector_provenance_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
print('Decision input:', len(candidates), 'seed0 candidates;', len(metrics), 'BPB metrics; SHA256', payload['dataset_sha256'])
print('Selector trace:', audit['status'], tree['sha'])
