import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from core import single_layer_scores,aggregate,threshold
z=np.array([1.,2.,3.,4.]);v=single_layer_scores(z,[0,2])
assert abs(v[0]-2.)<1e-12
assert abs(v[1]-(np.log(np.exp(1)+np.exp(3))-np.log(2)))<1e-12
assert abs(v[2]-(np.exp(1)+np.exp(3))/np.exp(z).sum())<1e-12
assert np.allclose(single_layer_scores(z+1000,[0,2])[:2],v[:2]+1000)
assert abs(single_layer_scores(z+1000,[0,2])[2]-v[2])<1e-12
f=np.arange(32,dtype=float);layer=np.tile([2.,3.,4.],(32,1));s=aggregate(f,layer)
assert s.shape==(9,)
assert np.array_equal(s,[292.5,26.,39.,52.,22.5,27.5,22.5,27.5,7.])
assert threshold(np.arange(400))==380
print('OC23_CONTRACT_PASS')
