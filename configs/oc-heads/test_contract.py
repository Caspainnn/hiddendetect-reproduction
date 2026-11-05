import unittest
import numpy as np
from core import baseline,threshold,scaler,make_neural,IDS
class Contracts(unittest.TestCase):
    def test_trapezoid_weights(self):
        x=np.zeros((2,14));x[0,0]=2;x[1,7]=2
        np.testing.assert_array_equal(baseline(x),[1.,2.])
        with self.assertRaises(ValueError):baseline(np.zeros((2,32)))
    def test_calibration_strict_ordinal(self):
        self.assertEqual(threshold(np.arange(400)),380)
        self.assertEqual(int((np.arange(400)>threshold(np.arange(400))).sum()),19)
    def test_train_only_scaling(self):
        mean,scale=scaler(np.array([[1.,2.],[3.,2.]]));np.testing.assert_array_equal(mean,[2.,2.]);self.assertGreater(scale[1],0)
    def test_neural_input_output_and_layer_identity(self):
        import torch
        for name in [IDS[1],IDS[3],IDS[7]]:
            torch.manual_seed(1);m=make_neural(name).eval();x=torch.randn(7,14)
            self.assertEqual(tuple(m(x).shape),(7,1));self.assertTrue(torch.isfinite(m(x)).all())
            saved={k:v.clone() for k,v in m.state_dict().items()};clone=make_neural(name);clone.load_state_dict(saved);clone.eval()
            torch.testing.assert_close(m(x),clone(x),rtol=0,atol=0)
if __name__=='__main__':unittest.main()
