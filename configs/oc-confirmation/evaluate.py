import importlib.util,json,joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from core import ROOT,H1,OUT,NAMES,SEEDS,PAIRS,digest,write,verify_source,predict,threshold,h1_core
def point(y,s,cuts):
    result=[]
    for j,name in enumerate(NAMES):
        m={'id':name,'auroc':float(roc_auc_score(y,s[:,j])),'pauc':float(roc_auc_score(y,s[:,j],max_fpr=.1))}
        for kind,cut in cuts.items():
            m[kind]={'threshold':float(cut[j])}
            for label,key in [(0,'fpr'),(1,'tpr')]:
                n=int((y==label).sum());k=int((s[y==label,j]>cut[j]).sum());p=k/n;z=1.959963984540054;d=1+z*z/n
                center=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
                m[kind].update({key:p,key+'_count':k,key+'_n':n,key+'_wilson95':[float(center-half),float(center+half)]})
        result.append(m)
    return result
def main():
    verify_source();assert not (OUT/'confirmation.json').exists()
    manifest=json.loads((OUT/'manifest.json').read_text());extraction=json.loads((OUT/'extraction.json').read_text())
    assert digest(ROOT/'cache/oc-confirmation/features.npz')==extraction['features_sha256']
    selection=json.loads((H1/'results/oc-heads/selection.json').read_text());data=np.load(ROOT/'cache/oc-confirmation/features.npz');y=data['test_y']
    scores={};all_metrics={};all_cuts={}
    for seed in SEEDS:
        values={role:[h1_core().baseline(data[role+'_X'])] for role in ['cal','test']}
        for name in NAMES[1:]:
            saved=next(m for m in selection['selected'][name]['models'] if m['seed']==seed);artifact=joblib.load(H1/saved['path'])
            for role in values:values[role].append(predict(artifact,data[role+'_X']))
        values={k:np.column_stack(v) for k,v in values.items()};scores[str(seed)]=values
        cuts={'old':np.array(manifest['old_thresholds'][str(seed)]),'new':np.array([threshold(values['cal'][:,j]) for j in range(4)])}
        all_cuts[str(seed)]={k:v.tolist() for k,v in cuts.items()};all_metrics[str(seed)]=point(y,values['test'],cuts)
    s=scores[str(SEEDS[0])]['test'];groups=[np.flatnonzero(y==k) for k in (0,1)];rng=np.random.default_rng(20261104)
    ix=np.column_stack([rng.choice(g,(10000,len(g))) for g in groups]);auc=np.empty((10000,4));partial=np.empty_like(auc)
    spec=importlib.util.spec_from_file_location('h2_bootstrap',ROOT/'configs/original-positions/core.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for start in range(0,10000,100):
        for j in range(4):auc[start:start+100,j],partial[start:start+100,j]=module.bootstrap_metrics(y,s[:,j],ix[start:start+100])
    comparisons=[]
    for j,k in PAIRS:
        ci=np.quantile(partial[:,j]-partial[:,k],[.025/6,1-.025/6]);lo,hi=ci
        item={'left':NAMES[j],'right':NAMES[k],'delta_pauc':all_metrics[str(SEEDS[0])][j]['pauc']-all_metrics[str(SEEDS[0])][k]['pauc'],
              'pauc_delta_ci991667':ci.tolist(),'auroc_delta_ci95':np.quantile(auc[:,j]-auc[:,k],[.025,.975]).tolist(),
              'decision':'new_source_positive' if lo>0 else 'worse' if hi<0 else 'unconfirmed'}
        for kind in ['old','new']:
            pred=s>np.asarray(all_cuts[str(SEEDS[0])][kind]);item[kind]={}
            for label,key in [(0,'fpr'),(1,'tpr')]:
                indices=ix[:,:len(groups[0])] if label==0 else ix[:,len(groups[0]):]
                delta=pred[:,j].astype(float)-pred[:,k].astype(float)
                item[kind][key+'_delta_ci95']=np.quantile(delta[indices].mean(1),[.025,.975]).tolist()
        comparisons.append(item)
    archive={}
    for seed,values in scores.items():
        for role,scores_ in values.items():archive[f'{seed}_{role}']=scores_
    np.savez(ROOT/'cache/oc-confirmation/frozen-scores.npz',**archive)
    result={'run_id':'OC-H2','evidence':'new_source_confirmation','source':'OR-Bench 80k benign + toxic; generated benchmark labels',
        'counts':[len(g) for g in groups],'primary_seed':SEEDS[0],'names':NAMES,'metrics':all_metrics[str(SEEDS[0])],'seed_metrics':all_metrics,
        'thresholds':all_cuts,'comparisons':comparisons,'family':6,'repeats':10000,'bootstrap_seed':20261104,'pauc_ci_coverage':1-.05/6,
        'source_lock_sha256':digest(OUT/'source-lock.json'),'features_sha256':extraction['features_sha256'],'frozen_scores_sha256':digest(ROOT/'cache/oc-confirmation/frozen-scores.npz'),
        'phase_two_not_started':True,'no_retraining_or_tuning_in_phase_one':True}
    write(OUT/'confirmation.json',result)
    print('OC_H2_NEW_SOURCE_SCORED_AND_SEALED',result['counts'],flush=True)
if __name__=='__main__':main()
