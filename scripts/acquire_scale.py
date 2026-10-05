"""Explicit network acquisition for V11. Frozen files are never overwritten."""
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import urllib.request

R = pathlib.Path(__file__).resolve().parents[1]
S = R / 'sources'
D = S / 'scale_2026_10_05'
QUERY = 'query($e:String!,$p:String!,$r:String!){project(name:$p,entityName:$e){run(name:$r){name displayName state config summaryMetrics}}}'


def fetch(item):
    name, url, payload = item
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json', 'User-Agent': 'Marin-Chinese-Research/11'})
    with urllib.request.urlopen(req, timeout=45) as res:
        body = res.read()
    if payload:
        parsed = json.loads(body)
        run = parsed.get('data', {}).get('project', {}).get('run')
        if parsed.get('errors') or not run or run['name'] != payload['variables']['r']:
            raise RuntimeError('Missing run or GraphQL errors: ' + name)
        json.loads(run['config'])
    elif name.endswith('.py'):
        compile(body, name, 'exec')
    elif name.endswith('.html'):
        if b'<html' not in body.lower():
            raise RuntimeError('Expected HTML: ' + name)
    else:
        parsed = json.loads(body)
        if parsed.get('number') != 9126:
            raise RuntimeError('Wrong issue')
    return name, body, {'file': str((D / name).relative_to(S)), 'url': url,
                        'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest(),
                        'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        **({'request_body': payload} if payload else {})}


if __name__ == '__main__':
    if D.exists():
        raise SystemExit('Refusing to overwrite dated source directory: ' + str(D))
    items = [('data_mixing_laws_v2.html', 'https://arxiv.org/html/2403.16952v2', None),
             ('regmix_v2.html', 'https://arxiv.org/html/2407.01492v2', None),
             ('regmix_d_v1.html', 'https://arxiv.org/html/2606.18663v1', None),
             ('issue_9126.json', 'https://api.github.com/repos/marin-community/marin/issues/9126', None)]
    for size, old in [('d768', 'rav-ladder-d768-v2'), ('d1024', 'rav-ladder-d1024'), ('d1536', 'rav-ladder-d1536')]:
        for name in [old, 'h100-mix25-20260912-' + size]:
            items.append(('config_' + name + '.json', 'https://api.wandb.ai/graphql',
                          {'query': QUERY, 'variables': {'e': 'marin-community', 'p': 'marin_moe', 'r': name}}))
    sha = json.loads((S / 'decision_2026_10_04/marin_tree.json').read_text())['sha']
    for name, path in [('train_hero_ep.py','experiments/grug/moe_hero_ep/train.py'), ('labeled_eval.py','lib/levanter/src/levanter/callbacks/labeled_eval.py'), ('grug_loss.py','lib/levanter/src/levanter/grug/loss.py'), ('eval.py','lib/levanter/src/levanter/eval.py')]:
        items.append((name, 'https://raw.githubusercontent.com/marin-community/marin/' + sha + '/' + path, None))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, items))
    # All requests and payload checks succeed before any snapshot is written.
    before = json.loads((S / 'archive_manifest.json').read_text())['files']
    D.mkdir()
    for name, body, _ in results:
        (D / name).write_bytes(body)
    manifest_path = S / 'source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.extend(r[2] for r in results)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    audit = {'snapshot': 'scale_2026_10_05', 'prior_archive_files': before,
             'new_files': [r[2] for r in results], 'all_requests_validated': True,
             'old_files_overwritten': [], 'live_training_history_refreshed': False}
    (R / 'analysis/scale_refresh_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    for name, body, _ in results:
        print(name, len(body))
