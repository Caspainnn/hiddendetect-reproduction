import importlib.util,json,joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from core import ROOT,H1,H2,OUT,NAMES,HEADS,SEEDS,PAIRS,digest,write,verify_source,predict,threshold,h1_core

def main():
    verify_source();primary=json.loads((OUT/'retraining.json').read_text());training=json.loads((OUT/'training.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text())
    assert digest(OUT/'training.json')==primary['training_sha256'] and len(training['runs'])==18
    data=np.load(H2/'cache/oc-confirmation/features.npz');old=np.load(H1/'cache/oc-heads/features.npz');new=np.load(ROOT/'cache/oc-retraining/features.npz');frozen=np.load(ROOT/'cache/oc-retraining/frozen-scores.npz');h2f=np.load(H2/'cache/oc-confirmation/frozen-scores.npz')
    assert digest(ROOT/'cache/oc-retraining/features.npz')==training['features_sha256'];assert digest(ROOT/'cache/oc-retraining/frozen-scores.npz')==primary['frozen_scores_sha256']
    selection=json.loads((H1/'results/oc-heads/selection.json').read_text());maximum=0.;checked=0
    for seed in SEEDS:
        for j,name in enumerate(NAMES):
            if j==0:artifact=None
            else:
                group,head=name.split(':')
                if group=='old_full':entry=next(m for m in selection['selected'][head]['models'] if m['seed']==seed);path=H1/entry['path']
                else:
                    entry=next(m for m in training['runs'] if m['seed']==seed and m['group']==group and m['id']==head);path=ROOT/entry['path'];x=old['train_X'][manifest['old_indices']] if group=='old_small' else new['train_X']
                assert digest(path)==entry['sha256'];artifact=joblib.load(path)
                if group!='old_full':
                    mean,scale=h1_core().scaler(x);np.testing.assert_array_equal(mean,artifact['mean']);np.testing.assert_array_equal(scale,artifact['scale']);assert entry['param']=={HEADS[0]:.1,HEADS[1]:10.,HEADS[2]:.003}[head]
                    if head!=HEADS[2]:assert artifact['model'].C==entry['param']
                    else:assert entry['epochs']==120 and entry['optimizer_steps']==120*int(np.ceil(len(x)/256))
            for role in ['cal','test']:
                fresh=h1_core().baseline(data[role+'_X']) if j==0 else predict(artifact,data[role+'_X']);saved=frozen[f'{seed}_{role}'][:,j];delta=float(abs(fresh-saved).max());assert delta<=1e-12;maximum=max(maximum,delta);checked+=len(fresh)
                if j<4:np.testing.assert_array_equal(fresh,h2f[f'{seed}_{role}'][:,j])
                if role=='cal':assert threshold(fresh)==primary['thresholds'][str(seed)][j]
                else:
                    y=data['test_y'];m=primary['seed_metrics'][str(seed)][j]
                    assert abs(roc_auc_score(y,fresh)-m['auroc'])<=1e-12 and abs(roc_auc_score(y,fresh,max_fpr=.1)-m['pauc'])<=1e-12
                    for label,key in [(0,'fpr'),(1,'tpr')]:assert int((fresh[y==label]>m['threshold']).sum())==m[key+'_count']
    spec=importlib.util.spec_from_file_location('h3_independent',ROOT/'configs/oc-heads/verify_statistics_independent.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    y=data['test_y'];s=frozen['20261104_test'];groups=[np.flatnonzero(y==k) for k in (0,1)];rng=np.random.default_rng(20261104);draws=[rng.choice(g,(10000,len(g))) for g in groups];dist=np.empty((10000,len(NAMES),2))
    for j in range(len(NAMES)):
        unique,bins=np.unique(s[:,j],return_inverse=True)
        for b in range(10000):dist[b,j]=module.histogram_roc(bins[draws[0][b]],bins[draws[1][b]],len(unique))
    maximum_ci=0.
    for (j,k),comparison in zip(PAIRS,primary['comparisons']):
        assert comparison['left']==NAMES[j] and comparison['right']==NAMES[k]
        for metric,key,probs in [(0,'auroc_delta_ci95',[.025,.975]),(1,'pauc_delta_ci994444',[.025/9,1-.025/9])]:
            delta=float(abs(np.quantile(dist[:,j,metric]-dist[:,k,metric],probs)-comparison[key]).max());assert delta<=1e-12;maximum_ci=max(maximum_ci,delta)
    write(OUT/'verification.json',{'status':'passed','models_replayed':27,'new_models':18,'frozen_score_values_replayed':checked,'maximum_score_difference':maximum,'independent_histogram_bootstrap_repeats':10000,'independent_intervals_checked':18,'maximum_interval_difference':maximum_ci,'train_only_scalers':True,'old_H2_outputs_exact':True,'confusion_counts_exact':True,'source_lock_sha256':digest(OUT/'source-lock.json')})
    print('OC_H3_VERIFIED',checked,maximum,maximum_ci,flush=True)
if __name__=='__main__':main()
