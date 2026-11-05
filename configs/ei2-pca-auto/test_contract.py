import numpy as np
from common import pack,variance_counts,choose_cv,prefix_gates

rng=np.random.default_rng(7);f=rng.normal(size=(20,14));a=rng.normal(size=(20,14,32)).astype(np.float32)
full,_,_=pack(f,a,[32]*14)
for k in [0,1,8,32]:
 x,mean,scale=pack(f,a,[k]*14);assert x.shape==(20,512) and not x[:,14+14*k:].any()
 if k:np.testing.assert_array_equal(x[:,14:14+14*k].reshape(20,14,k),full[:,14:462].reshape(20,14,32)[:,:,:k])
 before=mean.copy();pack(f[:2],a[:2]+100,[k]*14,mean,scale);np.testing.assert_array_equal(mean,before)
g=prefix_gates(rng.normal(size=(14,32)));assert np.all(np.diff(g,axis=1)<=0);mask=g>=.5;assert np.all(np.diff(mask.astype(int),axis=1)<=0)
class Basis:
 explained_variance_ratio_=np.array([.4,.3,.1]+[0.]*29)
counts,coverage,capped=variance_counts([Basis()]*14,.9);assert counts==[32]*14 and all(capped) and np.allclose(coverage,.8)
counts,_,capped=variance_counts([Basis()]*14,.7);assert counts==[2]*14 and not any(capped)
scores=np.zeros((3,32,3));scores[:,1,:]=[[.79]*3,[.80]*3,[.81]*3];scores[:,31,:]=[[.79]*3,[.81]*3,[.83]*3]
selection=choose_cv(scores);assert selection['k']==2 and selection['C']==.1
from verify import weighted_auc
from scipy.stats import rankdata
y=np.array([0,1,0,1]);s=np.array([.1,.1,.5,.7]);counts=np.array([[2,1,1,2]])
ey=np.repeat(y,counts[0]);es=np.repeat(s,counts[0]);n=ey.sum();expected=(rankdata(es)[ey==1].sum()-n*(n+1)/2)/(n*(len(ey)-n));assert abs(weighted_auc(y,s,counts)[0]-expected)<1e-12
print('PASS: variable/zero prefixes, fixed scaling, train-only standardization, monotone gates, variance cap, CV one-SE and tied AUC')
