"""Fetch only pinned, ungated public data; raw files remain git-ignored."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'configs'))
from fetch_locked import fetch

if __name__ == '__main__':
    sources = json.loads((Path(__file__).parent / 'data-sources.json').read_text())
    files = []
    for source in sources['sources']:
        repo, revision = source['repo'], source['revision']
        info = json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/' + repo, timeout=30))
        if info.get('gated'):
            raise RuntimeError('Access-gated source rejected: ' + repo)
        tree = json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true', timeout=30))
        lookup = {x['path']: x for x in tree}
        for filename in source['files']:
            item = lookup[filename]
            digest = item.get('lfs', {}).get('oid')
            if not digest:
                # Small regular Git blobs are checked by Git SHA1 after fetching.
                digest = None
            url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/' + urllib.parse.quote(filename)
            target = 'data/signal-selection/raw/' + repo.replace('/', '__') + '/' + filename
            files.append({'target': target, 'bytes': item['size'], 'sha256': digest,
                          'git_blob_sha': item.get('oid'), 'url': url, 'repo': repo,
                          'revision': revision, 'license': source['license']})
    for item in files:
        if item['sha256'] is None:
            data = urllib.request.urlopen(item['url'], timeout=60).read()
            actual_git = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            assert actual_git == item['git_blob_sha'], 'Git blob verification failed'
            item['sha256'] = hashlib.sha256(data).hexdigest()
            target = ROOT / item['target']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    with ThreadPoolExecutor(max_workers=3) as pool:
        verified = list(pool.map(fetch, files))
    (ROOT / 'logs/signal-selection-download.json').write_text(json.dumps({'files': files, 'verified': verified}, indent=2) + '\n')
    print('PUBLIC_DATA_VERIFIED', len(files), sum(x['bytes'] for x in files), flush=True)
