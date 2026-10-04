"""Archive a bounded selector provenance search without refreshing training evidence."""
import pathlib, json, urllib.request, datetime, hashlib, concurrent.futures
R = pathlib.Path(__file__).resolve().parents[1]
S = R / 'sources'
D = S / 'decision_2026_10_04'
D.mkdir(exist_ok=True)

def fetch(item):
    name, url = item
    request = urllib.request.Request(url, headers={'User-Agent': 'Marin-Chinese-Research/6.0'})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            b = response.read()
            final = response.url
        (D / name).write_bytes(b)
        return {'file': str((D / name).relative_to(S)), 'url': url, 'final_url': final,
                'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest(),
                'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    except Exception as e:
        return {'file': str((D / name).relative_to(S)), 'url': url, 'error': str(e)}

def batch(items):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, items))
    manifest = json.loads((S / 'source_manifest.json').read_text())
    (S / 'source_manifest.json').write_text(json.dumps(manifest + records, ensure_ascii=False, indent=2))
    for r in records:
        print(r['file'], r.get('bytes', r.get('error')), flush=True)
    return records

if __name__ == '__main__':
    batch([
        ('marin_head.json', 'https://api.github.com/repos/marin-community/marin/commits/main'),
        ('issue_9126.json', 'https://api.github.com/repos/marin-community/marin/issues/9126'),
        ('search_mixture_hash.json', 'https://api.github.com/search/issues?q=repo%3Amarin-community%2Fmarin+996f489106c7b922&per_page=100'),
        ('search_swarm_id.json', 'https://api.github.com/search/issues?q=repo%3Amarin-community%2Fmarin+h100-d512-from10pct-store-4d2e363d&per_page=100'),
    ])
    sha = json.loads((D / 'marin_head.json').read_text())['sha']
    batch([('marin_tree.json', 'https://api.github.com/repos/marin-community/marin/git/trees/' + sha + '?recursive=1')])
    print('Selector search pinned to:', sha, flush=True)
