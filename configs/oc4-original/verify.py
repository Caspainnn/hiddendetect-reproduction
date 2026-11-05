import joblib
from core import *
def main():
    check_lock();r=load(OUT/'evaluation.json');t=load(OUT/'training.json')
    assert digest(OUT/'training.json')==r['training_sha256'];assert digest(CACHE/'scores.npz')==r['scores_sha256'];assert digest(CACHE/'bootstrap.npz')==r['bootstrap_sha256']
    f=np.load(CACHE/'scores.npz');data=np.load(H1/'cache/oc-heads/features.npz');stored=np.load(CACHE/'bootstrap.npz');reloaded=0;maximum=0.;point_count=0
    for item in t['runs']:
        assert digest(ROOT/item['path'])==item['sha256'];art=joblib.load(ROOT/item['path']);j=NAMES.index(item['id'])
        for key in ['cal','RV','C']:np.testing.assert_array_equal(score(art,data[key+'_X']),f[f"{item['seed']}_{key}"][:,j])
        if item['id']==attention.PAUC:assert art['auxiliary_state'].shape==(1600,1) and attention.torch.isfinite(art['auxiliary_state']).all() and attention.torch.any(art['auxiliary_state']!=0)
        reloaded+=1
    independent=module('o1_independent',ROOT/'configs/oc-heads/verify_statistics_independent.py')
    for seed in SEEDS:
        cut=np.sort(f[f'{seed}_cal'],axis=0)[380];np.testing.assert_array_equal(cut,np.array(r['thresholds'][str(seed)]))
        for key in ['RV','C']:
            y=data[key+'_y'];s=f[f'{seed}_{key}']
            for j,row in enumerate(r['seed_metrics'][str(seed)][key]):
                a,p=independent.metric(y,s[:,j]);maximum=max(maximum,abs(a-row['auroc']),abs(p-row['pauc']));point_count+=1
                assert int((s[y==0,j]>cut[j]).sum())==row['fp'] and int((s[y==1,j]>cut[j]).sum())==row['tp']
    assert maximum<1e-12
    for key in ['RV','C']:
        y=data[key+'_y'];s=f[f'{SEEDS[0]}_{key}'];draws=bootstrap_draws(y);dist=np.empty((REPEATS,len(NAMES),2))
        for j,name in enumerate(NAMES):
            unique,bins=np.unique(s[:,j],return_inverse=True)
            for b in range(REPEATS):dist[b,j]=independent.histogram_roc(bins[draws[0][b]],bins[draws[1][b]],len(unique))
            for k,field in [(0,'auc'),(1,'pauc')]:np.testing.assert_allclose(dist[:,j,k],stored[key+'_'+field][:,j],rtol=0,atol=1e-12)
            print('OC_O1_INDEPENDENT',key,name,flush=True)
        pred=s>np.array(r['thresholds'][str(SEEDS[0])]);tail=.025/27 if key=='RV' else .025
        for row in r['cohorts'][key]['comparisons']:
            weights=np.array([sum(w for n,w in row['terms'] if n==name) for name in NAMES]);delta=np.einsum('bmk,m->bk',dist,weights)
            np.testing.assert_allclose(np.quantile(delta[:,1],[tail,1-tail]),row['pauc_delta_ci'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(np.quantile(delta[:,0],[.025,.975]),row['auroc_delta_ci95'],rtol=0,atol=1e-12)
            d=pred.astype(float)@weights
            for k,field in [(0,'fpr'),(1,'tpr')]:np.testing.assert_allclose(np.quantile(d[draws[k]].mean(1),[.025,.975]),row[field+'_delta_ci95'],rtol=0,atol=1e-12)
    write(OUT/'verification.json',{'passed':True,'evaluation_sha256':digest(OUT/'evaluation.json'),'model_reloads':reloaded,'point_metric_sets':point_count,'independent_roc_replicates':2*REPEATS*len(NAMES),'maximum_point_difference':maximum,'all_distributions_and_54_intervals_recomputed':True,'h1_scores_reproduced':r['h1_scores_reproduced']})
    print('OC_O1_VERIFIED',flush=True)
if __name__=='__main__':main()
