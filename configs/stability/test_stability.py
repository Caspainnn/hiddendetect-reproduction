import unittest
import numpy as np
from stability_core import draws,variants,contrasts,summary_features

class Tests(unittest.TestCase):
    def test_extractor_environment_module_not_shadowed(self):
        import symtable
        from pathlib import Path
        source=Path(__file__).with_name('extract.py').read_text(encoding='utf-8')
        main=next(t for t in symtable.symtable(source,'extract.py','exec').get_children() if t.get_name()=='main')
        self.assertTrue(main.lookup('os').is_global())

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

    def test_bootstrap_pairs_same_test_ids_across_all_draws(self):
        from stability_analysis import analyze
        names=[str(i)+'_'+v['name'] for i in range(5) for v in variants()]
        rows=[]
        for role,n in [('calibration_aegis',400),('calibration_toxic',400),('test_aegis_holdout',8),('test_toxic_holdout',8),('confirmation_aegis',8)]:
            for j in range(n):
                label=0 if role.startswith('calibration') else j%2
                rows.append({'role':role,'label':label,'scores':{name:float((j%7)/10+label+int(name[0])*.01) for name in names}})
        e=analyze(rows)
        self.assertEqual(sum(len(v) for v in e['tests'].values()),160)
        for c in e['contrasts']['confirmation_aegis:global'].values():
            for metric in ('auroc','partial_auroc','tpr','fpr'):
                self.assertEqual(c[metric]['delta'],0.)
                self.assertEqual(c[metric]['family6_ci'],[0.,0.])

if __name__=='__main__':unittest.main()
