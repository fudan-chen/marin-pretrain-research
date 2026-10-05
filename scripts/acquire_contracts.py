"""Archive implementation contracts at the existing pinned revision; no training execution."""
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import urllib.request

R = pathlib.Path(__file__).resolve().parents[1]
S = R / 'sources'
D = S / 'contracts_2026_10_05'
SHA = json.loads((S / 'decision_2026_10_04/marin_tree.json').read_text())['sha']
PATHS = {
    'model.py': 'experiments/grug/moe_hero_ep/model.py',
    'checkpointing.py': 'experiments/grug/checkpointing.py',
    'api.py': 'lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/api.py',
    'xla.py': 'lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/xla.py',
    'reference.py': 'lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/reference.py',
    'test_fused.py': 'lib/levanter/tests/kernels/test_pallas_fused_cross_entropy_loss.py',
}

def fetch(item):
    name, path = item
    url = 'https://raw.githubusercontent.com/marin-community/marin/' + SHA + '/' + path
    with urllib.request.urlopen(url, timeout=45) as response:
        body = response.read()
    compile(body, name, 'exec')
    return name, body, {'file': str((D / name).relative_to(S)), 'url': url,
                        'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest(),
                        'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}

if __name__ == '__main__':
    if D.exists():
        raise SystemExit('Refusing to overwrite frozen sources')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, PATHS.items()))
    D.mkdir()
    for name, body, _ in rows:
        (D / name).write_bytes(body)
    p = S / 'source_manifest.json'
    manifest = json.loads(p.read_text())
    manifest.extend(row[2] for row in rows)
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('Archived', len(rows), 'implementation files at', SHA)
