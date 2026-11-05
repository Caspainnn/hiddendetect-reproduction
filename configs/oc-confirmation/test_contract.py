import unittest
import numpy as np
from core import NAMES,PAIRS,threshold
from evaluate import point
class Contracts(unittest.TestCase):
    def test_family(self):
        self.assertEqual(len(PAIRS),6);self.assertEqual(len(set(PAIRS)),6);self.assertEqual(NAMES[2],'OC-4-1-3')
    def test_threshold_and_rates(self):
        self.assertEqual(threshold(np.arange(400)),380)
        y=np.array([0,0,1,1]);s=np.repeat(np.array([[0.],[1.],[1.],[2.]]),4,axis=1)
        r=point(y,s,{'old':np.ones(4),'new':np.zeros(4)})
        self.assertEqual(r[0]['old']['fpr_count'],0);self.assertEqual(r[0]['old']['tpr_count'],1)
        self.assertEqual(r[0]['new']['fpr_count'],1);self.assertEqual(r[0]['new']['tpr_count'],2)
    def test_no_unsafe_data_in_calibration(self):
        with self.assertRaises(ValueError):threshold(np.zeros(399))
if __name__=='__main__':unittest.main()
