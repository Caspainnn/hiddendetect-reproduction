import unittest
import numpy as np
from core import NAMES,PAIRS,threshold

class Contract(unittest.TestCase):
    def test_strict_calibration(self):
        x=np.arange(400.);cut=threshold(x);self.assertEqual(cut,380);self.assertEqual(int((x>cut).sum()),19)
        self.assertEqual(int((np.ones(400)>threshold(np.ones(400))).sum()),0)
    def test_calibration_size(self):
        for x in [np.arange(399.),np.full(400,np.nan)]:
            with self.assertRaises(ValueError):threshold(x)
    def test_comparison_identity(self):
        self.assertEqual(len(PAIRS),9);self.assertEqual(len(set(PAIRS)),9)
        for j,k in PAIRS:self.assertEqual(NAMES[j].split(':')[1],NAMES[k].split(':')[1]);self.assertGreater(j,k)
if __name__=='__main__':unittest.main()
