import unittest
import tempfile
from pathlib import Path
import numpy as np
from processing_core import weights,summarize,feature_matrix,relation_transform,joint_kernel,fit_joint,predict_joint,open_array

class Tests(unittest.TestCase):
    def test_json_shape_can_create_memmap(self):
        with tempfile.TemporaryDirectory() as d:
            a=open_array(Path(d)/'a.npy',[2,3],False,False);a[:]=1;a.flush()
            self.assertEqual(a.shape,(2,3));self.assertEqual(str(a.dtype),'float16');del a
            a=open_array(Path(d)/'a.npy',[2,3],False,True);np.testing.assert_array_equal(a,np.ones((2,3)));del a
    def test_weights_and_constant_case(self):
        np.testing.assert_allclose(weights([2,2,2]),np.ones(3)/3)
        w=weights([0,1,9]);self.assertAlmostEqual(w.sum(),1);self.assertGreater(w[2],w[1])
        with self.assertRaises(AssertionError):weights([])
    def test_summaries_and_relations(self):
        r=np.random.default_rng(4);h=r.normal(size=(5,7));pre=h-.1;att=r.normal(size=h.shape);mlp=r.normal(size=h.shape);g=r.normal(size=7)
        s,rel=summarize(h,pre,att,mlp,g,np.array([0,0,1,0,0]),'sample',7)
        self.assertEqual(s.shape,(7,7));self.assertEqual(rel.shape,(29,));self.assertTrue(np.isfinite(rel).all())
        u=h/np.linalg.norm(h,axis=1)[:,None];np.testing.assert_allclose(s[0],u.mean(0));np.testing.assert_allclose(s[1],u[2])
        a,b=summarize(h,pre,att,mlp,g,np.zeros(5),'sample',7);np.testing.assert_allclose(a[1],u.mean(0))
        c,_=summarize(h,pre,att,mlp,g,np.zeros(5),'sample',7);np.testing.assert_array_equal(a,c)
        z,rz=summarize(np.zeros_like(h),np.zeros_like(h),np.zeros_like(h),np.zeros_like(h),np.zeros_like(g),np.zeros(5),'z',7)
        self.assertTrue(np.isfinite(z).all() and np.isfinite(rz).all())
    def test_duplicate_mean_and_train_only_scaling(self):
        r=np.random.default_rng(2);a=r.normal(size=(11,4,7,8))
        x=feature_matrix(a,'mean');d=feature_matrix(a,'duplicate_mean')
        np.testing.assert_allclose(x@x.T,d@d.T,atol=1e-12)
        raw=r.normal(size=(11,116));raw[:,0]=1
        tr,z,mu,sd=relation_transform(raw,np.arange(7))
        np.testing.assert_allclose(tr.mean(0),0,atol=1e-14)
        raw[7:]+=1000;tr2,_,mu2,sd2=relation_transform(raw,np.arange(7))
        np.testing.assert_array_equal(tr,tr2);np.testing.assert_array_equal(mu,mu2);np.testing.assert_array_equal(sd,sd2)
    def test_additive_kernel_primal_and_relations(self):
        r=np.random.default_rng(9);b=r.normal(size=(19,5));e=r.normal(size=(19,8));z=r.normal(size=(19,116));y=r.integers(0,2,19)
        fit=np.arange(13);valid=np.arange(13,19)
        from routing_core import centered_predict
        for gamma in (.25,1.):
            g=joint_kernel(b,e,z,gamma,fit,True)
            pred=centered_predict(g,fit,valid,y*2-1)
            _,zs,mu,sd=relation_transform(z,fit);x=np.column_stack([b,np.sqrt(gamma)*e,np.sqrt(.1*gamma)*zs]);xf=x[fit]-x[fit].mean(0)
            coef=xf.T@np.linalg.solve(xf@xf.T+np.eye(len(fit)),(y[fit]*2-1)-(y[fit]*2-1).mean())
            expected=(x[valid]-x[fit].mean(0))@coef+(y[fit]*2-1).mean()
            np.testing.assert_allclose(pred,expected,atol=1e-11)
            h=fit_joint(b[fit],e[fit],z[fit],y[fit],gamma,True,device='cpu')
            np.testing.assert_allclose(predict_joint(h,b[valid],e[valid],z[valid]),expected,atol=1e-11)

if __name__=='__main__':unittest.main()
