import json,joblib
import numpy as np
from core import *

def main():
    verify_lock();r=json.loads((OUT/'evaluation.json').read_text());training=json.loads((OUT/'training.json').read_text())
    assert digest(OUT/'training.json')==r['training_sha256'];assert digest(CACHE/'frozen-scores.npz')==r['scores_sha256']
    assert digest(CACHE/'bootstrap-distributions.npz')==r['bootstrap_sha256']
    frozen=np.load(CACHE/'frozen-scores.npz');data=np.load(H2/'cache/oc-confirmation/features.npz');y=frozen['y'];names=r['names']
    for item in training['runs']:
        path=ROOT/item['path'];assert digest(path)==item['sha256'];art=joblib.load(path);j=names.index(item['id']);seed=item['seed']
        for role in ['cal','test']:np.testing.assert_array_equal(score(art,data[role+'_X']),frozen[f'{seed}_{role}'][:,j])
        if item['id']==PAUC:
            assert art['auxiliary_state'].shape==(300,1) and torch.isfinite(art['auxiliary_state']).all() and torch.any(art['auxiliary_state']!=0)
    m=module('a2_independent_roc',ROOT/'configs/oc-heads/verify_statistics_independent.py');maximum=0.;count=0
    for seed in SEEDS:
        s=frozen[f'{seed}_test'];cut=np.sort(frozen[f'{seed}_cal'],axis=0)[380]
        np.testing.assert_array_equal(cut,np.array(r['thresholds'][str(seed)]))
        for j,row in enumerate(r['seed_metrics'][str(seed)]):
            a,p=m.metric(y,s[:,j]);maximum=max(maximum,abs(a-row['auroc']),abs(p-row['pauc']))
            assert int((s[y==0,j]>cut[j]).sum())==row['fp'] and int((s[y==1,j]>cut[j]).sum())==row['tp'];count+=1
    assert maximum<=1e-12
    groups=[np.flatnonzero(y==k) for k in [0,1]];rng=np.random.default_rng(SEEDS[0]);draw=[rng.choice(g,(10000,len(g))) for g in groups]
    stored=np.load(CACHE/'bootstrap-distributions.npz');s=frozen[f'{SEEDS[0]}_test'];dist=np.empty((10000,len(names),2))
    for j,name in enumerate(names):
        unique,bins=np.unique(s[:,j],return_inverse=True)
        for b in range(10000):dist[b,j]=m.histogram_roc(bins[draw[0][b]],bins[draw[1][b]],len(unique))
        np.testing.assert_allclose(dist[:,j,0],stored['auc'][:,j],atol=1e-12,rtol=0);np.testing.assert_allclose(dist[:,j,1],stored['pauc'][:,j],atol=1e-12,rtol=0)
        print('OC_A2_INDEPENDENT',name,flush=True)
    family=r['family'];prediction=s>np.array(r['thresholds'][str(SEEDS[0])])
    for row in r['comparisons']:
        weights=np.array([sum(w for n,w in row['terms'] if n==name) for name in names]);delta=np.einsum('bmk,m->bk',dist,weights)
        np.testing.assert_allclose(np.quantile(delta[:,1],[.025/family,1-.025/family]),row['pauc_delta_ci'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(np.quantile(delta[:,0],[.025,.975]),row['auroc_delta_ci95'],atol=1e-12,rtol=0)
        d=prediction.astype(float)@weights
        for k,key in [(0,'fpr'),(1,'tpr')]:np.testing.assert_allclose(np.quantile(d[draw[k]].mean(1),[.025,.975]),row[key+'_delta_ci95'],atol=1e-12,rtol=0)
    write(OUT/'verification.json',{'passed':True,'evaluation_sha256':digest(OUT/'evaluation.json'),'model_reloads':len(training['runs']),
        'point_metric_sets':count,'independent_roc_replicates':10000*len(names),'maximum_point_difference':maximum,
        'bootstrap_distributions_and_all_intervals_recomputed':True,'libauc_auxiliary_states_retained':True,'no_sample_data_exported':True})
    print('OC_A2_VERIFIED',flush=True)
if __name__=='__main__':main()
