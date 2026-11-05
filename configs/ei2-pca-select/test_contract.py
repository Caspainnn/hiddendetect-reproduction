import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).with_name('common.py')
assert path.exists(), 'Arbitrary-direction implementation is missing'
spec=importlib.util.spec_from_file_location('select_common',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rng=np.random.default_rng(7);f=rng.normal(size=(120,14));a=rng.normal(size=(120,14,32))
mask=np.zeros((14,32),bool);mask[0,[1,6]]=True;mask[4,31]=True
x,mean,scale=m.pack(f,a,mask)
np.testing.assert_allclose(x[:,14:17],(a.reshape(120,448)[:,np.flatnonzero(mask)]-mean[14:])/scale[14:]/np.sqrt(56),atol=1e-7)
empty=m.pack(f,a,np.zeros((14,32),bool))[0];np.testing.assert_array_equal(empty[:,:14],x[:,:14]);assert not empty[:,14:].any()
prefix=m.prefix(mask);assert prefix.sum()==3 and prefix[0,:2].all() and prefix[4,0] and not prefix[0,6]
old=m.parent.pack(f,a,[2]*14)[0];new=m.pack(f,a,np.tile(np.arange(32)<2,(14,1)))[0];np.testing.assert_array_equal(old,new)
toy=rng.normal(size=(120,20));y=(toy[:,16]+rng.normal(size=120)*.3>0).astype(int)
for rho in [1.,.5]:
 state=m.fit_sparse(toy,y,.01,rho);assert state['kkt']<=2e-6 and state['mask'].shape==(6,)
 assert state['mask'][2] and abs(state['weight'][16])>.1
state=m.fit_ard(toy,y);assert state['map_gradient']<=2e-5 and np.isfinite(state['alpha']).all()
assert np.all(state['covariance_diagonal']>=0) and state['weight'][16]>0
print('PCA_DIRECTION_CONTRACT_PASS')
