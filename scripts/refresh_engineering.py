"""Acquire a bounded, separately dated public engineering snapshot.

This command uses the network. `make report` only consumes the saved payloads.
It never replaces the frozen W&B or earlier issue snapshots.
"""
import concurrent.futures, datetime, hashlib, json, pathlib, urllib.request

R = pathlib.Path(__file__).resolve().parents[1]
S = R / 'sources'
DEST = 'engineering_2026_10_05'
BASE = 'https://api.github.com/repos/marin-community/marin/'
ENDPOINTS = {
    'issue_8435.json': 'issues/8435',
    'issue_8435_comments.json': 'issues/8435/comments?per_page=100&page=1',
    'issue_8506.json': 'issues/8506',
    'issue_8506_comments.json': 'issues/8506/comments?per_page=100&page=1',
    'issue_8870.json': 'issues/8870',
    'issue_8870_comments.json': 'issues/8870/comments?per_page=100&page=1',
    'pull_9062.json': 'pulls/9062',
    'pull_9062_files.json': 'pulls/9062/files?per_page=100&page=1',
    'pull_9183.json': 'pulls/9183',
    'pull_9183_files.json': 'pulls/9183/files?per_page=100&page=1',
    'pull_9333.json': 'pulls/9333',
    'pull_9333_files.json': 'pulls/9333/files?per_page=100&page=1',
}

def fetch(item):
    name, endpoint = item
    url = BASE + endpoint
    request = urllib.request.Request(url, headers={'Accept':'application/vnd.github+json','User-Agent':'marin-pretrain-public-research'})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
        if 'rel="next"' in response.headers.get('Link', ''):
            raise RuntimeError('Unfetched next page: ' + url)
        metadata = {'file': DEST+'/'+name, 'url':url, 'final_url':response.url,
                    'bytes':len(body), 'sha256':hashlib.sha256(body).hexdigest(),
                    'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    return name, body, json.loads(body), metadata

def main():
    if (S/DEST).exists():
        raise RuntimeError('Dated acquisition already exists; do not overwrite it.')
    prior = json.loads((S/'archive_manifest.json').read_text())
    manifest = json.loads((S/'source_manifest.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, ENDPOINTS.items()))
    payloads = {name: obj for name, _, obj, _ in results}
    for number in (8435,8506,8870):
        assert payloads['issue_%d.json'%number]['comments'] == len(payloads['issue_%d_comments.json'%number])
    for number in (9062,9183,9333):
        assert payloads['pull_%d.json'%number]['changed_files'] == len(payloads['pull_%d_files.json'%number])
    comparisons = []
    for number, old_name in ((8435,'comments.json'),(8506,'issue_8506_comments.json'),(8870,'issue_8870_comments.json')):
        old = {c['id']:c for c in json.loads((S/old_name).read_text())}
        new = {c['id']:c for c in payloads['issue_%d_comments.json'%number]}
        comparisons.append({'issue':number,'old_count':len(old),'new_count':len(new),
            'added_ids':sorted(new.keys()-old.keys()),'removed_ids':sorted(old.keys()-new.keys()),
            'changed_bodies':[i for i in sorted(old.keys()&new.keys()) if old[i]['body']!=new[i]['body']],
            'old_file':old_name,'new_file':DEST+'/issue_%d_comments.json'%number})
    (S/DEST).mkdir()
    for name, body, _, metadata in results:
        (S/DEST/name).write_bytes(body)
        manifest.append(metadata)
    (S/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    audit = {'scope':'Bounded public GitHub acquisition; production curves and old content files not refreshed',
        'prior_integrity_manifest':prior,'new_acquisition_files':len(results),
        'comment_comparisons':comparisons,'pagination_complete':True,
        'issue_body_changes':[],'acquisition_records':[r[3] for r in results]}
    for number, old_name in ((8435,'issue.json'),(8506,'issue_8506.json'),(8870,'issue_8870.json')):
        old_body = json.loads((S/old_name).read_text())['body']
        new_body = payloads['issue_%d.json'%number]['body']
        audit['issue_body_changes'].append({'issue':number,'changed':old_body!=new_body})
    (R/'analysis/engineering_refresh_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'files':len(results),'comment_comparisons':comparisons,'issue_body_changes':audit['issue_body_changes']},ensure_ascii=False))

if __name__ == '__main__':
    main()
