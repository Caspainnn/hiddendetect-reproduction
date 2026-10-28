from pathlib import Path
import tempfile
import numpy as np
from features import regions_from_offsets, pool_hidden, projection_matrix, open_cache, checkpoint

regions = regions_from_offsets([(0,0), (0,2), (2,4), (4,6), (6,8), (8,9)], (2,8))
assert [r.tolist() for r in regions] == [[2], [3], [4], []]
hidden = np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
pooled = pool_hidden(hidden, [np.array([0,1]),np.array([2]),np.array([3]),np.array([],int)])
assert pooled.shape == (5,2) and np.allclose(pooled[0],[0,-1])
assert np.allclose(pooled[1],[.5,.5]) and np.all(pooled[4] == 0)
assert np.array_equal(projection_matrix(8,4),projection_matrix(8,4))
root = Path(__file__).resolve().parents[2] / 'cache'
root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=root) as directory:
    arrays, start = open_cache(Path(directory), 'a', 2, {'version':1})
    assert start == 0
    arrays['F'][0] = 1
    checkpoint(arrays, 1)
    for key in arrays:
        if isinstance(arrays[key], np.memmap): arrays[key]._mmap.close()
    arrays, start = open_cache(Path(directory), 'a', 2, {'version':1})
    assert start == 1 and np.all(arrays['F'][0] == 1)
    try: open_cache(Path(directory), 'a', 2, {'version':2})
    except ValueError: pass
    else: raise AssertionError('Mixed provenance accepted')
    for key in arrays:
        if isinstance(arrays[key], np.memmap): arrays[key]._mmap.close()
print('PASS: user masks, empty regions, unit-vector pooling, deterministic projection and provenance/prefix cache')
