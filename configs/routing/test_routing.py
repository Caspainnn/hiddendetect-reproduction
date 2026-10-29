import unittest
import numpy as np
from routing_core import centered_predict, route, warm_masks, search, pad_ids, readouts

class Tests(unittest.TestCase):
    def test_kernel_matches_explicit_fold_centering(self):
        r=np.random.default_rng(8);a=r.normal(size=(19,4,2,7));y=r.integers(0,2,19)*2-1
        fit=np.arange(13);valid=np.arange(13,19);mask=np.array([1,0,1,0],bool)
        x=route(a,mask);g=x@x.T
        pred=centered_predict(g,fit,valid,y)
        xf=x[fit]-x[fit].mean(0);xv=x[valid]-x[fit].mean(0)
        coef=np.linalg.solve(xf.T@xf+np.eye(x.shape[1]),xf.T@(y[fit]-y[fit].mean()))
        np.testing.assert_allclose(pred,xv@coef+y[fit].mean(),atol=1e-10)
        gs=sum(a[:,l,int(mask[l]),:]@a[:,l,int(mask[l]),:].T for l in range(4))/4
        np.testing.assert_allclose(g,gs,atol=1e-12)
        changed=y.copy();changed[valid]=999
        np.testing.assert_allclose(pred,centered_predict(g,fit,valid,changed))
    def test_mapping_and_padding(self):
        a=np.arange(3*4*2*2).reshape(3,4,2,2)
        np.testing.assert_array_equal(route(a,[0,1,0,1])*2,a[:,np.arange(4),[0,1,0,1],:].reshape(3,-1))
        self.assertEqual(pad_ids([1,2],4,0),([1,2,0,0],[1,1,0,0]))
        self.assertEqual(readouts({'input_length':12,'readout_indices':[0,1,2,3,4,5,7,11,6]}),[7,9,10,11])
        with self.assertRaises(AssertionError):pad_ids([1]*5,4,0)
    def test_equal_budget_and_warm_start(self):
        expected=[''.join(map(str,m.astype(int))) for m in warm_masks(8)]
        for method in ['random','ga','bpso','de','scan']:
            def objective(masks):return [-float(np.sum(m)) for m in masks]
            a=search(method,8,objective,42,budget=24,population=6)
            b=search(method,8,objective,42,budget=24,population=6)
            self.assertEqual(a,b);self.assertEqual(a['unique_evaluations'],24)
            self.assertEqual(a['evaluated_masks'][:10],expected)
            self.assertEqual(len(set(a['evaluated_masks'])),24)
            self.assertEqual(sum(a['best_mask']),0)

if __name__=='__main__':unittest.main()
