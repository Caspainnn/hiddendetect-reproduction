import numpy as np
from run import pack_fixed

rng=np.random.default_rng(7);f=rng.normal(size=(40,14));a=rng.normal(size=(40,14,32)).astype(np.float32)
packed={}
for k in [4,8,16,32]:
 extra=a[:,:,:k].reshape(40,-1);x,mean,scale=pack_fixed(f,extra);packed[k]=x[:,14:14+14*k].reshape(40,14,k)
 assert x.shape==(40,512) and not x[:,14+14*k:].any()
 np.testing.assert_allclose(x[:,:14].std(0),1/np.sqrt(14),atol=1e-7)
 np.testing.assert_allclose(packed[k].std(0),1/np.sqrt(56),atol=1e-7)
 before=mean.copy();pack_fixed(f[:2],extra[:2]+100,mean,scale);np.testing.assert_array_equal(mean,before)
for k in [4,8,16]:np.testing.assert_array_equal(packed[k],packed[32][:,:,:k])
from verify import weighted_auc
from scipy.stats import rankdata
y=np.array([0,1,0,1,1]);s=np.array([.2,.2,.5,.9,.9]);counts=np.array([[1,2,3,1,2],[2,1,1,3,1]])
for row,actual in zip(counts,weighted_auc(y,s,counts)):
 ey=np.repeat(y,row);es=np.repeat(s,row);n=ey.sum();expected=(rankdata(es)[ey==1].sum()-n*(n+1)/2)/(n*(len(ey)-n))
 assert abs(actual-expected)<1e-12
print('PASS: nested information, identical coordinate scaling, fixed F block, train-only scaling and 512 padding')
