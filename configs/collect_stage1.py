"""Read-only local inventory; never imports the model or downloads weights."""
import hashlib
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
repo = ROOT / 'source/HiddenDetect'
def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return {'command': args, 'exit_code': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}

packages = {}
for name in ['torch', 'torchvision', 'transformers', 'accelerate', 'llava',
             'numpy', 'pandas', 'scikit-learn', 'pyarrow', 'fastparquet']:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
files = {}
for path in repo.rglob('*'):
    if path.is_file() and '.git' not in path.relative_to(repo).parts:
        files[path.relative_to(repo).as_posix()] = {
            'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
with zipfile.ZipFile(repo / 'data/few_shot.zip') as archive:
    inventory = [{'path': x.filename, 'bytes': x.file_size} for x in archive.infolist()]
    json_files = [x.filename for x in archive.infolist() if x.filename.endswith('.json')]
    few_shot = {}
    for name in json_files:
        rows = json.loads(archive.read(name).decode('utf-8-sig'))
        if isinstance(rows, list):
            few_shot[name] = {'count': len(rows),
                'unsafe': sum(x.get('toxicity') == 1 for x in rows),
                'safe': sum(x.get('toxicity', 0) != 1 for x in rows),
                'text_only': sum(x.get('img') in [None, 'null'] for x in rows),
                'image_paths': sorted({str(x.get('img')) for x in rows if x.get('img') not in [None, 'null']})}
report = {'date': '2026-10-27', 'python': sys.version, 'executable': sys.executable,
          'platform': platform.platform(),
          'disk': {x: dict(zip(['total', 'used', 'free'], shutil.disk_usage(x))) for x in ['C:/', 'E:/']},
          'packages': packages, 'gpu': command(['nvidia-smi']),
          'commit': command(['git', '-C', str(repo), 'rev-parse', 'HEAD']),
          'git_status': command(['git', '-C', str(repo), 'status', '--porcelain']),
          'files': files, 'few_shot_archive': inventory, 'few_shot_summary': few_shot,
          'school_gpu': {'source': 'user confirmation', 'access': 'SSH', 'os': 'Linux',
                         'reported_free_storage_gb': 60, 'reported_gpu': 'RTX 4090 24GB',
                         'remote_inspected': False}}
target = ROOT / 'logs/stage1-inventory-2026-10-27.json'
target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(target)
print(json.dumps(few_shot, ensure_ascii=False, indent=2))
