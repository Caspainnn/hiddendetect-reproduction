import unittest
import numpy as np
from anchored_core import candidates,fit_alignment,fuse,contrasts,select

class Tests(unittest.TestCase):
    def test_normal_only_alignment_and_boundary_support(self):
        b=np.array([-1.,0.,1.,2.,30.,40.]);e=np.array([0.,2.,4.,6.,50.,90.]);y=np.array([0,0,0,0,1,1])
        h=fit_alignment(b,e,y);h2=fit_alignment(b,np.r_[e[:4],999.,-999.],y)
        self.assertEqual(h,h2)
        np.testing.assert_allclose(fuse(b,e,{'mode':'global','alpha':1.},h)[:4],b[:4],atol=1e-12)
        far=np.array([-1000.,1000.]);extra=np.array([4000.,-4000.])
        np.testing.assert_array_equal(fuse(far,extra,{'mode':'gated','alpha':1.,'width':.5},h),far)
        np.testing.assert_array_equal(fuse(b,e,{'mode':'global','alpha':0.},h),b)

    def test_budget_and_control_selection(self):
        cs=candidates();self.assertEqual(len(cs),55);self.assertEqual(len(contrasts()),7)
        for mode in ('global','gated'):
            for family in ('mean','dynamic','random'):
                self.assertEqual(sum(c['mode']==mode and c['family']==family for c in cs),9)
        rows=[{**c,'cv_mean':.7,'cv_std':.03} for c in cs]
        for r in rows:
            if r['family']=='random':r['cv_mean']=.99
        s=select(rows);self.assertEqual(s['selected'],'B');self.assertNotIn('random',s['raw_best'])

if __name__=='__main__':unittest.main()
