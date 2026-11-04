"""Small synthetic checks for the information interface and train-only scaling."""
import numpy as np
from core import unit,pack

def main():
 z=np.array([[3.,4.,0.],[0.,0.,0.]])
 u,n=unit(z)
 np.testing.assert_allclose(u[0],[.6,.8,0],atol=1e-7)
 np.testing.assert_equal(u[1],0)
 assert np.isfinite(n).all()
 rng=np.random.default_rng(22);f=rng.normal(size=(30,14));extra=rng.normal(size=(30,448))
 x,mean,scale=pack(f,extra)
 assert x.shape==(30,512) and not x[:,462:].any()
 np.testing.assert_allclose(x[:,:14].mean(0),0,atol=1e-7)
 np.testing.assert_allclose(x[:,:14].std(0),1/np.sqrt(14),atol=1e-7)
 held=extra[:4]+1000
 before=mean.copy();test,_,_=pack(f[:4],held,mean,scale)
 np.testing.assert_equal(mean,before)
 np.testing.assert_allclose(test[:,14:462],(held-mean[14:])/scale[14:]/np.sqrt(448),rtol=2e-7)
 b,_,_=pack(f,np.empty((30,0)));assert b.shape==(30,512) and not b[:,14:].any()
 try:pack(f,np.zeros((30,600)))
 except AssertionError:pass
 else:raise AssertionError('Silent compression of full vocabulary')
 full,_,_=pack(f,np.zeros((30,600)),pad=False);assert full.shape==(30,614)
 print('PASS: normalization, fixed F block, train-only scaling, constant columns, padding and full interface')

if __name__=='__main__':main()
