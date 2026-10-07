"""Archive complete files from a verified detached upstream checkout."""
import argparse, datetime, hashlib, json, pathlib, subprocess
R=pathlib.Path(__file__).resolve().parents[1]; S=R/'sources'
a=argparse.ArgumentParser(); a.add_argument('checkout',type=pathlib.Path); args=a.parse_args()
rev='eee467718515b2383fc3a433014afce4ab075b05'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.checkout,text=True).strip()==rev
paths=['lib/levanter/src/levanter/tensorstore_serialization.py', 'lib/levanter/tests/test_tensorstore_serialization.py', 'lib/levanter/src/levanter/checkpoint.py']
prior=json.loads((S/'archive_manifest.json').read_text()); d=S/'donation_snapshot_2026_10_07'; assert not d.exists(); rows=[]
for p in paths:
 b=subprocess.check_output(['git','show',rev+':'+p],cwd=args.checkout)
 assert b==(args.checkout/p).read_bytes(),p
 target=d/p; target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
 rows.append({'file':str(target.relative_to(S)),'upstream_path':p,'url':f'https://raw.githubusercontent.com/marin-community/marin/{rev}/{p}','git_revision':rev,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'retrieval_time_basis':'Local git-blob archive completion; full checkout previously fetched from public GitHub'})
m=json.loads((S/'source_manifest.json').read_text());m.extend(rows);(S/'source_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
old=[x for x in prior['files'] if x['file']!='source_manifest.json']; assert all(hashlib.sha256((S/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in old)
(R/'analysis/donation_snapshot_acquisition.json').write_text(json.dumps({'records':rows,'prior_integrity_manifest':prior,'prior_bytes_preserved':len(old)},indent=2)+'\n')
print('Archived',len(rows),'complete blobs; preserved',len(old),'non-bookkeeping files')
