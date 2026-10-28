import tempfile
from pathlib import Path
from download_public import save_acquisition_log

with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / 'lock.json'
    report = {'files': [{'sha256': 'fixed'}], 'verified': [{'status': 'verified'}]}
    save_acquisition_log(path, report)
    before = path.read_bytes()
    save_acquisition_log(path, {'files': report['files'], 'verified': [{'status': 'already_verified'}]})
    assert path.read_bytes() == before
    try:
        save_acquisition_log(path, {'files': [{'sha256': 'changed'}], 'verified': []})
    except ValueError:
        pass
    else:
        raise AssertionError('Immutable acquisition lock changed')
print('PASS: acquisition revalidation preserves original bytes/status history and rejects file drift')
