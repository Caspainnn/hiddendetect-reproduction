import unittest
import numpy as np
from stability_core import draws,variants,contrasts,summary_features

class Tests(unittest.TestCase):
    def test_draws_balanced_reproducible_and_isolated(self):
        rows=[{'id':str(i),'group_id':str(i),'label':i%2,'role':'pool'} for i in range(3200)]
        rows.append({'id':'test','group_id':'test','label':0,'role':'confirmation_aegis'})
        a=draws(rows);self.assertEqual(len(a),5)
        for ix in a:
            self.assertEqual(len(set(ix)),1600)
            self.assertEqual(sum(rows[i]['label'] for i in ix),800)
            self.assertTrue(all(rows[i]['role']=='pool' for i in ix))
        self.assertEqual([x.tolist() for x in a],[x.tolist() for x in draws(rows)])
        bad=rows.copy();bad[1]={**bad[1],'group_id':'0'}
        with self.assertRaises(AssertionError):draws(bad)

    def test_controls_and_contrasts(self):
        x=np.arange(2*4*5*8).reshape(2,4,5,8)
        self.assertEqual(summary_features(x,'mean').shape,(2,32))
        self.assertEqual(summary_features(x,'dynamic').shape,summary_features(x,'random').shape)
        self.assertEqual(len(variants()),8);self.assertEqual(len(contrasts()),6)
        np.testing.assert_allclose(summary_features(x,'dynamic'),x[:,:,[1,2],:].reshape(2,-1)/np.sqrt(8))

if __name__=='__main__':unittest.main()
