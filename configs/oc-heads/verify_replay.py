"""Replay after an exact-bit assertion exposed parallel RF reduction roundoff.

The frozen production implementation and all initial scores remain untouched.
Only score/threshold floating comparisons allow 1e-12; confusion counts stay exact.
"""
import importlib.util,json,joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from core import ROOT,OUT,IDS,SEEDS,digest,read,write,score,baseline,threshold
def main():
    torch.set_num_threads(2);selection=json.loads((OUT/'selection.json').read_text());seal=json.loads((OUT/'pre-evaluation-seal.json').read_text());lock=json.loads((OUT/'source-lock.json').read_text())
    assert seal['selection_sha256']==digest(OUT/'selection.json')
    for name,sha in lock['sha256'].items():assert digest(ROOT/name)==sha
    data=np.load(ROOT/'cache/oc-heads/features.npz');manifest=json.loads((OUT/'manifest.json').read_text());assert digest(ROOT/'cache/oc-heads/features.npz')==manifest['features_sha256']
    rows=read(OUT/'scores.jsonl');difference=0.;checked=0;metric_difference=0.;threshold_difference=0.
    for seed in SEEDS:
        cuts=json.loads((OUT/f'thresholds-{seed}.json').read_text())['values']
        for j,name in enumerate(['B']+IDS):
            if j:
                model=next(m for m in selection['selected'][name]['models'] if m['seed']==seed)
                assert digest(ROOT/model['path'])==model['sha256'];artifact=joblib.load(ROOT/model['path'])
            for role in ['cal','C','RV']:
                s=baseline(data[role+'_X']) if j==0 else score(artifact,data[role+'_X'])
                old=np.array([r['scores'][j] for r in rows if r['role']==role and r['seed']==seed])
                delta=float(abs(s-old).max());difference=max(delta,difference);assert delta<=1e-12,(seed,name,role,delta);checked+=len(s)
                if role=='cal':
                    td=abs(threshold(s)-cuts[j]);threshold_difference=max(td,threshold_difference);assert td<=1e-12
                else:
                    metric=json.loads((OUT/f'seed-{seed}-{role}.json').read_text())[j];y=data[role+'_y'];prediction=s>cuts[j]
                    for key,max_fpr in [('auroc',None),('pauc',.1)]:
                        md=abs(roc_auc_score(y,s,max_fpr=max_fpr)-metric[key]);metric_difference=max(metric_difference,md);assert md<=1e-12,(name,role,key,md)
                    for label,key in [(0,'fpr'),(1,'tpr')]:assert int(prediction[y==label].sum())==metric[key+'_count'],(seed,name,role,key)
    spec=importlib.util.spec_from_file_location('replay_bootstrap',ROOT/'configs/original-positions/core.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    y=np.array([0,0,0,0,1,1,1,1]);s=np.array([0,1,1,2,1,2,2,3.]);rng=np.random.default_rng(44)
    ix=np.column_stack([rng.choice(np.flatnonzero(y==k),(100,len(y)//2)) for k in (0,1)])
    auc,partial=module.bootstrap_metrics(y,s,ix)
    expected=np.array([[roc_auc_score(y[i],s[i]),roc_auc_score(y[i],s[i],max_fpr=.1)] for i in ix])
    np.testing.assert_allclose(np.column_stack([auc,partial]),expected,rtol=0,atol=1e-14)
    search=json.loads((OUT/'search.json').read_text());assert len(search)==8
    for item in search:
        chosen=max(item['candidates'],key=lambda c:c['mean_dev_pauc'])
        assert chosen['index']==selection['selected'][item['id']]['candidate_index']
    write(OUT/'verification.json',{'status':'passed','score_values_replayed':checked,'maximum_score_difference':difference,
        'maximum_threshold_difference':threshold_difference,'maximum_metric_difference':metric_difference,
        'floating_tolerance':1e-12,'confusion_counts_exact':True,'selected_models':24,'selection_replayed':True,
        'original_bitwise_verifier_failed_on_parallel_RF_roundoff':True,'production_scores_unchanged':True,
        'tied_bootstrap_vs_sklearn_replicates':100,'source_lock_sha256':digest(OUT/'source-lock.json'),'replay_verifier_sha256':digest(__file__)})
    print('HEAD_REPLAY_VERIFIED',checked,difference,'METRIC_DELTA',metric_difference,flush=True)
if __name__=='__main__':main()
