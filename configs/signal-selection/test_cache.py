from pathlib import Path
import tempfile
import numpy as np
from extract_features import open_cache, checkpoint_cache

temporary_root = Path(__file__).resolve().parents[2] / 'cache'
temporary_root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
    assert Path(folder).resolve().is_relative_to(temporary_root.resolve())
    cache, start = open_cache(Path(folder), 'locked-manifest', 3)
    assert start == 0 and cache['F'].shape == (3, 32)
    cache['F'][0] = np.arange(32)
    checkpoint_cache(cache, 1)
    resumed, start = open_cache(Path(folder), 'locked-manifest', 3)
    assert start == 1 and np.array_equal(resumed['F'][0], np.arange(32))
    for wrong_key, wrong_count in [('changed-manifest', 3), ('locked-manifest', 4)]:
        try:
            open_cache(Path(folder), wrong_key, wrong_count)
        except ValueError:
            pass
        else:
            raise AssertionError('Manifest drift reused cache')
    try:
        checkpoint_cache(cache, 4)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid completed prefix accepted')
    for collection in (cache, resumed):
        for key in ('H', 'F', 'L'):
            collection[key]._mmap.close()
print('PASS: cache shape, durable-prefix resume and manifest/count drift rejection')
