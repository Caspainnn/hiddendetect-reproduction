import json
import tempfile
from pathlib import Path
from seal import stream_hash,verify_seal
from features import digest
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder)
    for name in ('metadata.json','checkpoint.json'): (root/name).write_text('{}')
    (root/'X.npy').write_bytes(b'original')
    seal={'metadata_sha256':digest(root/'metadata.json'),'checkpoint_sha256':digest(root/'checkpoint.json'),
        'arrays':{'X.npy':{'bytes':8,'sha256':stream_hash(root/'X.npy')}}}
    verify_seal(root,seal)
    (root/'X.npy').write_bytes(b'modified')
    try: verify_seal(root,seal)
    except AssertionError: pass
    else: raise AssertionError('Same-size cache corruption accepted')
print('PASS: cache seal rejects same-size array corruption')
