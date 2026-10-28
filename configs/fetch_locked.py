"""Fetch pinned public mirrors; verify bytes before promoting .part files."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

def verify(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _sha256(stream)
    return actual == digest

def _sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()

def fetch(item):
    target = (ROOT / item['target']).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError('Artifact target escapes project')
    target.parent.mkdir(parents=True, exist_ok=True)
    if verify(target, item['bytes'], item['sha256']):
        return {'target': item['target'], 'status': 'already_verified'}
    partial = target.with_name(target.name + '.part')
    print('FETCH', item['target'], item['bytes'], flush=True)
    subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--retry-delay', '2',
                    '--connect-timeout', '20', '--max-time', '3600', '--continue-at', '-',
                    '--silent', '--show-error', '--output', str(partial), item['url']], check=True)
    if not verify(partial, item['bytes'], item['sha256']):
        raise ValueError('Size/SHA256 mismatch: ' + item['target'])
    partial.replace(target)
    print('VERIFIED', item['target'], flush=True)
    return {'target': item['target'], 'status': 'verified', 'sha256': item['sha256'], 'bytes': item['bytes']}

if __name__ == '__main__':
    started = time.monotonic()
    lock = json.loads((ROOT / 'configs/artifacts-lock.json').read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, lock['files']))
    report = {'files': results, 'wall_seconds': time.monotonic() - started}
    (ROOT / 'logs/artifacts-verified.json').write_text(json.dumps(report, indent=2))
    print('ALL_ARTIFACTS_VERIFIED', flush=True)
