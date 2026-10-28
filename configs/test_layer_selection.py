from pathlib import Path
import subprocess
import tempfile
from run_selection import select_range, require_author_commit

assert select_range([0.0, 2.0, 1.0, 3.0, 1.0]) == (1, 3, [1, 3])
assert select_range([2.0, 0.0, 1.0]) == (0, 0, [0])
for vector in [[1.0, 1.0], [0.0, 1.0], [], [float('nan'), 0.0]]:
    try:
        select_range(vector)
    except ValueError:
        pass
    else:
        raise AssertionError('Undefined/nonfinite range must fail')
print('PASS: strict last-layer baseline, gap retention, zero-based boundary, no invented fallback')

with tempfile.TemporaryDirectory() as folder:
    subprocess.run(['git', 'init', '-q', folder], check=True)
    subprocess.run(['git', '-C', folder, '-c', 'user.name=Test', '-c', 'user.email=test@localhost',
                    'commit', '-q', '--allow-empty', '-m', 'first'], check=True)
    first = subprocess.check_output(['git', '-C', folder, 'rev-parse', 'HEAD'], text=True).strip()
    assert require_author_commit(Path(folder), first) == first
    subprocess.run(['git', '-C', folder, '-c', 'user.name=Test', '-c', 'user.email=test@localhost',
                    'commit', '-q', '--allow-empty', '-m', 'second'], check=True)
    try:
        require_author_commit(Path(folder), first)
    except ValueError:
        pass
    else:
        raise AssertionError('Clean repository on another commit must be rejected')
print('PASS: clean repository revision drift is rejected')
