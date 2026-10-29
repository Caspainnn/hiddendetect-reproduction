import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from head_core import grid,center_kernel,center_cross,span_features,span_cross,fit_kernel,predict_kernel,squared_distances,validate_cache_metadata,selection

class Tests(unittest.TestCase):
    def test_cache_source_and_completion_rejected(self):
        source={'extract':'a'};meta={'manifest_sha256':'m','shape':[2,32,4096],'dtype':'float16','fixed_length':512,'artifact_lock_sha256':'l','extraction_sources':source}
        state={'next_index':2,'prefix_difference':0.,'peak_allocated_bytes':1};pilot={'rows':32,'maximum_old_B_difference':0.,'prefix_difference':0.,'peak_allocated_bytes':1}
        validate_cache_metadata(meta,state,pilot,2,'m','l',source)
        with self.assertRaises(AssertionError):validate_cache_metadata(meta,{**state,'next_index':1},pilot,2,'m','l',source)
        with self.assertRaises(AssertionError):validate_cache_metadata(meta,state,pilot,2,'m','wrong',source)
        with self.assertRaises(AssertionError):validate_cache_metadata(meta,state,pilot,2,'m','l',{'extract':'wrong'})
    def test_unique_analysis_and_selection(self):
        from head_analysis import contrasts
        self.assertEqual(len(contrasts({'selected_head':'ridge_tuned'})),5)
        rows=[{**c,'cv_mean':.7,'cv_std':.03} for c in grid()]
        rows[9]['cv_mean']=.701
        s=selection(rows);self.assertEqual(s['raw_best'],'logistic_0');self.assertEqual(s['selected'],'ridge_0')
    def test_nine_settings_per_family(self):
        rows=grid();self.assertEqual(len(rows),36)
        for method in ('ridge','logistic','svm','rbf'):self.assertEqual(sum(c['method']==method for c in rows),9)
        self.assertTrue(any(c['method']=='ridge' and c['value']==1 for c in rows))
    def test_span_logistic_and_svm_equal_raw(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(47,12));v=rng.normal(size=(9,12));y=np.arange(47)%2
        mu=x.mean(0);xc=x-mu;vc=v-mu;k=x@x.T;kv=v@x.T
        kc,col,grand=center_kernel(k);kvc=center_cross(kv,col,grand)
        z,basis,rank,error=span_features(kc);zv=span_cross(kvc,basis)
        self.assertEqual(rank,12);self.assertLess(error,1e-10)
        np.testing.assert_allclose(z@z.T,xc@xc.T,atol=1e-9)
        for c in (.1,10):
            a=LogisticRegression(C=c,tol=1e-10,max_iter=5000).fit(xc,y);b=LogisticRegression(C=c,tol=1e-10,max_iter=5000).fit(z,y)
            np.testing.assert_allclose(a.decision_function(vc),b.decision_function(zv),atol=2e-5)
            a=SVC(kernel='linear',C=c,tol=1e-9).fit(xc,y);b=SVC(kernel='precomputed',C=c,tol=1e-9).fit(kc,y)
            np.testing.assert_allclose(a.decision_function(vc),b.decision_function(kvc),atol=2e-5)
    def test_ridge_and_rbf_prediction(self):
        rng=np.random.default_rng(6);x=rng.normal(size=(19,7));v=rng.normal(size=(6,7));y=np.arange(19)%2
        k=x@x.T;kv=v@x.T;d=squared_distances(k,np.diag(k),np.diag(k));dv=squared_distances(kv,(v*v).sum(1),np.diag(k))
        for method in ('ridge','rbf'):
            h=fit_kernel(k,y,{'method':method,'value':1.,'factor':1.},d)
            pred=predict_kernel(h,kv,dv)
            raw=np.exp(-d/h['median']) if method=='rbf' else k;rawv=np.exp(-dv/h['median']) if method=='rbf' else kv
            kc,mu,grand=center_kernel(raw);expected=center_cross(rawv,mu,grand)@np.linalg.solve(kc+np.eye(len(x)),y*2-1-(y*2-1).mean())+(y*2-1).mean()
            np.testing.assert_allclose(pred,expected,atol=1e-10)

if __name__=='__main__':unittest.main()
