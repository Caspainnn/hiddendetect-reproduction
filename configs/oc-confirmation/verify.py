import importlib.util,json,joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from core import ROOT,H1,OUT,NAMES,SEEDS,PAIRS,digest,read,write,verify_source,predict,threshold,h1_core
def main():
    verify_source();manifest=json.loads((OUT/'manifest.json').read_text());primary=json.loads((OUT/'confirmation.json').read_text())
    data=np.load(ROOT/'cache/oc-confirmation/features.npz');frozen=np.load(ROOT/'cache/oc-confirmation/frozen-scores.npz')
    assert digest(ROOT/'cache/oc-confirmation/features.npz')==primary['features_sha256']
    assert digest(ROOT/'cache/oc-confirmation/frozen-scores.npz')==primary['frozen_scores_sha256']
    selection=json.loads((H1/'results/oc-heads/selection.json').read_text());maximum=0.;checked=0
    for seed in SEEDS:
        for j,name in enumerate(NAMES):
            artifact=None if j==0 else joblib.load(H1/next(m for m in selection['selected'][name]['models'] if m['seed']==seed)['path'])
            for role in ['cal','test']:
                fresh=h1_core().baseline(data[role+'_X']) if j==0 else predict(artifact,data[role+'_X'])
                old=frozen[f'{seed}_{role}'][:,j];delta=float(abs(fresh-old).max());assert delta<=1e-12,(seed,name,role,delta)
                maximum=max(maximum,delta);checked+=len(fresh)
                if role=='cal':assert abs(threshold(fresh)-primary['thresholds'][str(seed)]['new'][j])<=1e-12
                else:
                    y=data['test_y'];m=primary['seed_metrics'][str(seed)][j]
                    assert abs(roc_auc_score(y,fresh)-m['auroc'])<=1e-12 and abs(roc_auc_score(y,fresh,max_fpr=.1)-m['pauc'])<=1e-12
                    for kind in ['old','new']:
                        cut=primary['thresholds'][str(seed)][kind][j]
                        for label,key in [(0,'fpr'),(1,'tpr')]:assert int((fresh[y==label]>cut).sum())==m[kind][key+'_count']
    # Independently reconstruct tied-score ROC curves with histograms, not the production helper.
    spec=importlib.util.spec_from_file_location('independent_h1_histogram',ROOT/'configs/oc-heads/verify_statistics_independent.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    y=data['test_y'];s=frozen['20261104_test'];groups=[np.flatnonzero(y==k) for k in (0,1)];rng=np.random.default_rng(20261104)
    draws=[rng.choice(g,(10000,len(g))) for g in groups];dist=np.empty((10000,4,2))
    for j in range(4):
        unique,bins=np.unique(s[:,j],return_inverse=True)
        for b in range(10000):dist[b,j]=module.histogram_roc(bins[draws[0][b]],bins[draws[1][b]],len(unique))
    interval_max=0.
    for (j,k),comparison in zip(PAIRS,primary['comparisons']):
        assert comparison['left']==NAMES[j] and comparison['right']==NAMES[k]
        for metric,key,probs in [(0,'auroc_delta_ci95',[.025,.975]),(1,'pauc_delta_ci991667',[.025/6,1-.025/6])]:
            delta=float(abs(np.quantile(dist[:,j,metric]-dist[:,k,metric],probs)-comparison[key]).max());assert delta<=1e-12
            interval_max=max(interval_max,delta)
    stability=json.loads((OUT/'stability.json').read_text());pre=json.loads((OUT/'pre-stability-seal.json').read_text())
    assert pre['confirmation_sha256']==digest(OUT/'confirmation.json')==stability['phase_one_confirmation_sha256']
    assert len(stability['runs'])==15
    source=np.load(H1/'cache/oc-heads/features.npz');trainx,trainy=source['train_X'],source['train_y'];stability_max=0.
    assert digest(H1/'cache/oc-heads/features.npz')==json.loads((H1/'results/oc-heads/manifest.json').read_text())['features_sha256']
    for seed in range(20261114,20261119):
        group=[r for r in stability['runs'] if r['seed']==seed];assert len(group)==3 and len({r['train_index_sha256'] for r in group})==1
        rng=np.random.default_rng(seed);ix=np.concatenate([rng.choice(np.flatnonzero(trainy==k),800) for k in (0,1)])
        for row in group:
            assert __import__('hashlib').sha256(ix.tobytes()).hexdigest()==row['train_index_sha256']
            path=ROOT/'cache/oc-confirmation/stability-models'/f"{row['id']}-{seed}.joblib";assert digest(path)==row['model_sha256']
            artifact=joblib.load(path);mean,scale=h1_core().scaler(trainx[ix])
            np.testing.assert_array_equal(artifact['mean'],mean);np.testing.assert_array_equal(artifact['scale'],scale)
            test=predict(artifact,data['test_X'])
            for key,alpha in [('auroc',None),('pauc',.1)]:
                delta=abs(roc_auc_score(data['test_y'],test,max_fpr=alpha)-row[key]);assert delta<=1e-12;stability_max=max(stability_max,delta)
            for kind,x in [('old',source['cal_X']),('new',data['cal_X'])]:
                cut=threshold(predict(artifact,x));assert abs(cut-row[kind]['threshold'])<=1e-12
                for label,key in [(0,'fpr'),(1,'tpr')]:assert abs(float((test[data['test_y']==label]>cut).mean())-row[kind][key])<=1e-12
    oldnames=json.loads((H1/'results/oc-heads/thresholds-20261104.json').read_text())['names']
    oldcal=[]
    for row in read(H1/'results/oc-heads/scores.jsonl'):
        if row['role']=='cal' and row['seed']==20261104:oldcal.append([row['scores'][oldnames.index(name)] for name in NAMES])
    rng=np.random.default_rng(20261124);ix=rng.integers(0,400,size=(1000,400));calibration_max=0.
    for kind,cal in [('old',np.array(oldcal)),('new',frozen['20261104_cal'])]:
        for j,name in enumerate(NAMES):
            stored=stability['calibration_variability'][kind][j];assert stored['id']==name
            cut=np.partition(cal[ix,j],380,axis=1)[:,380]
            np.testing.assert_allclose(np.quantile(cut,[.025,.975]),stored['threshold_range'],rtol=0,atol=1e-12)
            for label,key in [(0,'fpr'),(1,'tpr')]:
                values=np.sort(s[data['test_y']==label,j]);rates=1-np.searchsorted(values,cut,side='right')/len(values)
                delta=float(abs(np.quantile(rates,[.025,.975])-stored[key+'_empirical95_range']).max());assert delta<=1e-12;calibration_max=max(delta,calibration_max)
    support=all(c['decision']=='new_source_positive' for c in primary['comparisons'] if c['left']==NAMES[3] and c['right'] in NAMES[1:3])
    wins=0
    for seed in range(20261114,20261119):
        values={r['id']:r['pauc'] for r in stability['runs'] if r['seed']==seed};wins+=int(values[NAMES[3]]>max(values[NAMES[1]],values[NAMES[2]]))
    assert support==stability['mechanism_new_source_support_gate'] and wins==stability['attention_robust_wins_out_of_5']
    assert (support and wins>=4)==stability['mechanism_entry_gate']
    write(OUT/'verification.json',{'status':'passed','frozen_score_values_replayed':checked,'maximum_frozen_score_difference':maximum,
        'independent_histogram_bootstrap_repeats':10000,'independent_intervals_checked':12,'maximum_interval_difference':interval_max,
        'confusion_counts_exact':True,'phase_one_not_replaced_by_stability':True,'source_lock_sha256':digest(OUT/'source-lock.json'),
        'stability_models_replayed':15,'maximum_stability_metric_difference':stability_max,'calibration_bootstrap_ranges_replayed':True,
        'maximum_calibration_range_difference':calibration_max,'mechanism_gate_replayed':True})
    print('OC_H2_VERIFIED',checked,maximum,interval_max,flush=True)
if __name__=='__main__':main()
