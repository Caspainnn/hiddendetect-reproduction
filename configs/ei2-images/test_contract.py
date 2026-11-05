import numpy as np
from core import pack,assert_group_roles,weighted_auc,bootstrap,unit
f=np.arange(56,dtype=np.float32).reshape(4,14);z=np.ones((4,14),np.float32);x,m,s=pack(f,z);assert x.shape==(4,512) and len(m)==28 and np.isfinite(x).all();np.testing.assert_array_equal(x[:,28:],0)
assert_group_roles([{'group_id':'a','role':'train'},{'group_id':'b','role':'test'}])
try:assert_group_roles([{'group_id':'a','role':'train'},{'group_id':'a','role':'test'}])
except AssertionError:pass
else:raise AssertionError('group leakage undetected')
y=np.array([0,1,0,1]);score=np.array([0.,1.,.5,.5]);w=np.ones((2,4));np.testing.assert_allclose(weighted_auc(y,score,w),.875)
out=bootstrap(y,np.column_stack([score,score]),['a','a','b','b'],draws=20);np.testing.assert_array_equal(out[:,0],out[:,1]);np.testing.assert_allclose(unit(np.array([[3.,4.]])),[[.6,.8]])
print('IMAGE_GROUP_PACK_AND_CLUSTER_BOOTSTRAP_CONTRACT_PASS')
