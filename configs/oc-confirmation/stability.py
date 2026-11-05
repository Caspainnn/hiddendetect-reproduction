import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='2'
import json,joblib,time
import numpy as np
import torch
from core import ROOT,H1,OUT,NAMES,digest,write,verify_source,predict,threshold,h1_core
from evaluate import point
def main():
    torch.set_num_threads(2);verify_source();primary=json.loads((OUT/'confirmation.json').read_text())
    assert primary['phase_two_not_started'] and not (OUT/'stability.json').exists()
    write(OUT/'pre-stability-seal.json',{'confirmation_sha256':digest(OUT/'confirmation.json'),'phase_one_scores_sha256':digest(ROOT/'cache/oc-confirmation/frozen-scores.npz'),'primary_models_not_replaced':True})
    assert digest(H1/'cache/oc-heads/features.npz')==json.loads((H1/'results/oc-heads/manifest.json').read_text())['features_sha256']
    source=np.load(H1/'cache/oc-heads/features.npz');data=np.load(ROOT/'cache/oc-confirmation/features.npz')
    x,y=source['train_X'],source['train_y'];testy=data['test_y'];selection=json.loads((H1/'results/oc-heads/selection.json').read_text())
    cache=ROOT/'cache/oc-confirmation/stability-models';cache.mkdir(exist_ok=True);runs=[];started=time.monotonic()
    for seed in range(20261114,20261119):
        rng=np.random.default_rng(seed);ix=np.concatenate([rng.choice(np.flatnonzero(y==k),800) for k in (0,1)])
        index_sha=__import__('hashlib').sha256(ix.tobytes()).hexdigest()
        for name in NAMES[1:]:
            path=cache/f'{name}-{seed}.joblib';meta=path.with_suffix('.json')
            if meta.exists():
                result=json.loads(meta.read_text());assert result['source_lock_sha256']==digest(OUT/'source-lock.json')
            else:
                artifact,cost=h1_core().fit(name,selection['selected'][name]['parameter'],seed,x[ix],y[ix]);joblib.dump(artifact,path)
                test=predict(artifact,data['test_X']);oldcal=predict(artifact,source['cal_X']);newcal=predict(artifact,data['cal_X'])
                # point() expects the four named schemes; fill a dedicated one-head summary directly.
                from sklearn.metrics import roc_auc_score
                result={'id':name,'seed':seed,'train_index_sha256':index_sha,'train_unique_inputs':len(set(ix.tolist())),
                    'auroc':float(roc_auc_score(testy,test)),'pauc':float(roc_auc_score(testy,test,max_fpr=.1)),
                    'cost':cost,'model_sha256':digest(path),'source_lock_sha256':digest(OUT/'source-lock.json')}
                for kind,cal in [('old',oldcal),('new',newcal)]:
                    cut=threshold(cal);result[kind]={'threshold':cut,'tpr':float((test[testy==1]>cut).mean()),'fpr':float((test[testy==0]>cut).mean())}
                write(meta,result)
            assert digest(path)==result['model_sha256'];runs.append(result)
            print('OC_H2_RESAMPLE',seed,name,'PAUC',round(result['pauc'],6),flush=True)
    frozen=np.load(ROOT/'cache/oc-confirmation/frozen-scores.npz');newcal=frozen['20261104_cal'];test=frozen['20261104_test']
    oldrows=[]
    with (H1/'results/oc-heads/scores.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            if row['role']=='cal' and row['seed']==20261104:oldrows.append(row)
    oldnames=json.loads((H1/'results/oc-heads/thresholds-20261104.json').read_text())['names']
    assert digest(H1/'results/oc-heads/scores.jsonl')==json.loads((H1/'results/oc-heads/evaluation-complete.json').read_text())['scores_sha256']
    oldcal=np.array([[row['scores'][oldnames.index(name)] for name in NAMES] for row in oldrows]);assert oldcal.shape==(400,4)
    rng=np.random.default_rng(20261124);ix=rng.integers(0,400,size=(1000,400));calibration={}
    for kind,cal in [('old',oldcal),('new',newcal)]:
        values=[]
        for j,name in enumerate(NAMES):
            cut=np.sort(cal[ix,j],axis=1)[:,380];m={'id':name,'threshold_range':np.quantile(cut,[.025,.975]).tolist()}
            for label,key in [(0,'fpr'),(1,'tpr')]:
                rates=(test[testy==label,j,None]>cut[None,:]).mean(0)
                m[key+'_empirical95_range']=np.quantile(rates,[.025,.975]).tolist()
            values.append(m)
        calibration[kind]=values
    # Mechanism work requires new-source support against both linear heads.
    attention=[p for p in primary['comparisons'] if p['left']==NAMES[3] and p['right'] in NAMES[1:3]]
    support=all(p['decision']=='new_source_positive' for p in attention)
    robust_wins=0
    for seed in range(20261114,20261119):
        values={r['id']:r['pauc'] for r in runs if r['seed']==seed}
        robust_wins+=int(values[NAMES[3]]>max(values[NAMES[1]],values[NAMES[2]]))
    write(OUT/'stability.json',{'run_id':'OC-H2','evidence':'post_confirmation_robustness_diagnostic','training_resamples':5,'fits':15,
        'train_pool_labels':1600,'sample_size':1600,'runs':runs,'calibration_resamples':1000,'calibration_variability':calibration,
        'calibration_scope':'fixed phase-one models and observed test; empirical ranges, not joint confidence intervals',
        'seconds':time.monotonic()-started,'phase_one_confirmation_sha256':digest(OUT/'confirmation.json'),
        'mechanism_new_source_support_gate':support,'attention_robust_wins_out_of_5':robust_wins,
        'mechanism_entry_gate':support and robust_wins>=4,'mechanism_experiments_executed':False})
    print('OC_H2_STABILITY_COMPLETE',len(runs),'MECHANISM_GATE',support,flush=True)
if __name__=='__main__':main()
