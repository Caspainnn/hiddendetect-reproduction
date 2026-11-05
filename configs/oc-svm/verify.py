import importlib.util,json,joblib
import numpy as np
from core import ROOT,OUT,CACHE,H2,SEEDS,digest,verify_lock,score,write

def main():
    verify_lock();r=json.loads((OUT/'evaluation.json').read_text());training=json.loads((OUT/'training.json').read_text())
    assert digest(OUT/'training.json')==r['training_sha256']
    assert digest(CACHE/'frozen-scores.npz')==r['scores_sha256']
    assert digest(CACHE/'bootstrap-distributions.npz')==r['bootstrap_sha256']
    frozen=np.load(CACHE/'frozen-scores.npz');data=np.load(H2/'cache/oc-confirmation/features.npz');y=frozen['y'];names=r['names']
    for seed in SEEDS:
        for j,name in enumerate(names[1:],1):
            item=next(t for t in training['runs'] if t['id']==name and t['seed']==(seed if name not in ['H3-C10','cost-fixed-C-unweighted'] else SEEDS[0]))
            path=ROOT/item['path'];assert digest(path)==item['sha256'];model=joblib.load(path)
            for role in ['cal','test']:np.testing.assert_allclose(score(model,data[role+'_X']),frozen[f'{seed}_{role}'][:,j],atol=1e-12,rtol=0)
        cuts=np.sort(frozen[f'{seed}_cal'],axis=0)[380]
        np.testing.assert_allclose(cuts,r['thresholds'][str(seed)],atol=1e-12,rtol=0)
    spec=importlib.util.spec_from_file_location('independent_roc',ROOT/'configs/oc-heads/verify_statistics_independent.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    maximum=0.;count=0
    for seed in SEEDS:
        s=frozen[f'{seed}_test'];cuts=np.array(r['thresholds'][str(seed)])
        for j,row in enumerate(r['seed_metrics'][str(seed)]):
            a,p=m.metric(y,s[:,j]);maximum=max(maximum,abs(a-row['auroc']),abs(p-row['pauc']))
            assert int((s[y==0,j]>cuts[j]).sum())==row['fp']
            assert int((s[y==1,j]>cuts[j]).sum())==row['tp'];count+=1
    assert maximum<=1e-12
    groups=[np.flatnonzero(y==k) for k in [0,1]];rng=np.random.default_rng(SEEDS[0]);draw=[rng.choice(g,(10000,len(g))) for g in groups]
    stored=np.load(CACHE/'bootstrap-distributions.npz');s=frozen[str(SEEDS[0])+'_test'];distribution=np.empty((10000,len(names),2))
    for j,name in enumerate(names):
        unique,bins=np.unique(s[:,j],return_inverse=True)
        for b in range(10000):distribution[b,j]=m.histogram_roc(bins[draw[0][b]],bins[draw[1][b]],len(unique))
        np.testing.assert_allclose(distribution[:,j,0],stored['auc'][:,j],atol=1e-12,rtol=0)
        np.testing.assert_allclose(distribution[:,j,1],stored['pauc'][:,j],atol=1e-12,rtol=0)
        print('OC_S1_INDEPENDENT',name,flush=True)
    for row in r['comparisons']:
        j,k=names.index(row['left']),names.index(row['right']);delta=distribution[:,j]-distribution[:,k]
        np.testing.assert_allclose(np.quantile(delta[:,1],[.025/4,1-.025/4]),row['pauc_delta_ci9875'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.quantile(delta[:,0],[.025,.975]),row['auroc_delta_ci95'],atol=1e-12,rtol=0)
        pred=s>np.array(r['thresholds'][str(SEEDS[0])]);difference=pred[:,j].astype(float)-pred[:,k].astype(float)
        for label,key in [(0,'fpr'),(1,'tpr')]:
            np.testing.assert_allclose(np.quantile(difference[draw[label]].mean(1),[.025,.975]),row[key+'_delta_ci95'],atol=1e-12,rtol=0)
    write(OUT/'verification.json',{'passed':True,'evaluation_sha256':digest(OUT/'evaluation.json'),'model_reloads':len(training['runs']),
        'point_metric_sets':count,'independent_roc_replicates':10000*len(names),'maximum_point_difference':maximum,
        'bootstrap_distributions_and_all_intervals_recomputed':True,'no_sample_data_exported':True})
    print('OC_S1_VERIFIED',flush=True)
if __name__=='__main__':main()
