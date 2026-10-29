import unittest
from unittest.mock import patch
import numpy as np
from routing_core import centered_predict, route, warm_masks, search, pad_ids, readouts,block,validate_metadata,replay_selection

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
        rows=[{'previous_index':0},{'fresh_index':0}]
        fixed=np.arange(2*4*4*3).reshape(2,4,4,3)
        fresh=np.arange(4*2*3).reshape(1,4,2,3)
        old=np.arange(4*9*3).reshape(1,4,9,3)
        arrays=(rows,fixed,fresh,old);x=block(arrays,[0,1],'variable')
        np.testing.assert_array_equal(x[0,:,0],old[0,:,7])
        np.testing.assert_array_equal(x[0,:,1],old[0,:,6])
        np.testing.assert_array_equal(x[1,:,0],fresh[0,:,1])
        np.testing.assert_array_equal(block(arrays,[0,1],'fixed')[:,:,0],fixed[:,:,3])
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
    def test_de_restart_updates_population(self):
        initial={''.join(map(str,m.astype(int))) for m in warm_masks(8)}
        with patch('routing_core.de_trial',side_effect=lambda pop,i,rng:(pop[i].copy(),[])):
            result=search('de',8,lambda ms:[float(np.sum(m)) for m in ms],42,budget=24,population=6)
        self.assertGreater(result['diversity_restarts'],0)
        self.assertTrue(set(result['final_population'])-initial,'restart evaluations must feed the population')
    def test_incomplete_and_wrong_source_cache_rejected(self):
        m={'samples':[{},{}],'fresh_class_count':1}
        meta={'manifest_sha256':'m','extractor_sha256':'e','core_sha256':'c','dtype':'float16','shapes':{'fixed.npy':[2,32,4,4096],'fresh_variable.npy':[2,32,2,4096]}}
        state={'next_index':2,'prefix_max_difference':0.}
        validate_metadata(meta,state,m,'m',{'extractor':'e','core':'c'})
        with self.assertRaises(AssertionError):validate_metadata(meta,{**state,'next_index':1},m,'m',{'extractor':'e','core':'c'})
        with self.assertRaises(AssertionError):validate_metadata(meta,state,m,'m',{'extractor':'wrong','core':'c'})
    def test_selection_prefers_baseline_within_se(self):
        base='0'*32;single='1'+'0'*31;alluser='1'*32
        d=lambda v:{'fitness':v,'stats':{m:{'std':.04} for m in ('variable','fixed')}}
        cache={base:d(.70),single:d(.705),alluser:d(.69)}
        for i in range(32):cache.setdefault('0'*i+'1'+'0'*(31-i),d(.68))
        runs=[{'method':'ga','seed':1,'best_mask':[x=='1' for x in single]}]
        result=replay_selection(cache,runs)
        self.assertEqual(result['selected_mask'],base);self.assertEqual(result['raw_best_mask'],single)

if __name__=='__main__':unittest.main()
