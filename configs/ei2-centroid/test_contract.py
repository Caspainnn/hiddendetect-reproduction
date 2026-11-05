import numpy as np
from core import fit_centroids,features,pack
h=np.zeros((8,14,4),np.float16);h[:4,:,0]=1;h[4:,:,1]=1;y=np.repeat([0,1],4)
refs=fit_centroids(h,y);c=features(h,refs)
np.testing.assert_allclose(c[:4],-1);np.testing.assert_allclose(c[4:],1)
np.testing.assert_array_equal(fit_centroids(h,y),refs)
held=np.ones((3,14,4),np.float16);saved=refs.copy();features(held,refs);np.testing.assert_array_equal(refs,saved)
x,m,s=pack(np.ones((8,14)),c);assert x.shape==(8,512) and len(m)==len(s)==28 and np.isfinite(x).all()
np.testing.assert_array_equal(x[:,28:],0)
print('CENTROID_TRAIN_ONLY_ALIGNMENT_CONTRACT_PASS')
