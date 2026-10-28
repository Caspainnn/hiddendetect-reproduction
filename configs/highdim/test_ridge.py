import numpy as np
from ridge import fit_ridge,ridge_scores
rng=np.random.default_rng(1); x=rng.normal(size=(40,12)).astype(np.float32); y=(x[:,0]>0).astype(int)
head=fit_ridge(x,y)
assert np.allclose(head['mean'],x.mean(0),atol=1e-6)
assert np.corrcoef(ridge_scores(head,x),y)[0,1]>.8
saved=head['mean'].copy(); other=x*100
ridge_scores(head,other)
assert np.array_equal(saved,head['mean'])
print('PASS: full-vector ridge is training-only and score orientation is unsafe-positive')
