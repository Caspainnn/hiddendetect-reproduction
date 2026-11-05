import json,joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from core import *

def point(y,s,cuts,names):
    rows=[]
    for j,name in enumerate(names):
        pred=s[:,j]>cuts[j]
        rows.append({'id':name,'auroc':float(roc_auc_score(y,s[:,j])),'pauc':float(roc_auc_score(y,s[:,j],max_fpr=.1)),
            'tpr':float(pred[y==1].mean()),'fpr':float(pred[y==0].mean()),'tp':int(pred[y==1].sum()),'fp':int(pred[y==0].sum()),'threshold':float(cuts[j])})
    return rows

def main():
    verify_lock();assert not (OUT/'evaluation.json').exists()
    training=json.loads((OUT/'training.json').read_text());sealed=digest(OUT/'training.json')
    data=np.load(H2/'cache/oc-confirmation/features.npz');x=data['test_X'];cal=data['cal_X'];y=data['test_y']
    assert x.shape==(1500,14) and cal.shape==(400,14) and (data['cal_y']==0).all()
    assert [(y==k).sum() for k in (0,1)]==[1000,500]
    s1=json.loads((S1/'results/oc-svm/evaluation.json').read_text());s1f=np.load(S1/'cache/oc-svm/frozen-scores.npz')
    check=json.loads((S1/'results/oc-svm/verification.json').read_text());assert check['passed'] and digest(S1/'results/oc-svm/evaluation.json')==check['evaluation_sha256']
    assert digest(S1/'cache/oc-svm/frozen-scores.npz')==s1['scores_sha256'];np.testing.assert_array_equal(y,s1f['y'])
    svm_index=s1['names'].index('OC-4-1-3');names=['B','SVM']+NAMES;metrics={};cuts={};values={}
    for seed in SEEDS:
        roles={role:[np.trapz(data[role+'_X'],axis=1),s1f[f'{seed}_{role}'][:,svm_index]] for role in ['cal','test']}
        for name in NAMES:
            item=next(r for r in training['runs'] if r['id']==name and r['seed']==seed);path=ROOT/item['path']
            assert digest(path)==item['sha256'];art=joblib.load(path)
            for role in roles:roles[role].append(score(art,data[role+'_X']))
        roles={role:np.column_stack(s) for role,s in roles.items()};cut=np.array([threshold(roles['cal'][:,j]) for j in range(len(names))])
        values.update({f'{seed}_{role}':s for role,s in roles.items()});cuts[str(seed)]=cut.tolist();metrics[str(seed)]=point(y,roles['test'],cut,names)
    # Historical regression model is evaluated only after every new model is sealed.
    entry=training['historical_replay'];assert digest(ROOT/entry['path'])==entry['sha256'];old=joblib.load(ROOT/entry['path'])
    oldcore=module('a2_original_head',ROOT/'configs/oc-heads/core.py');oldf=np.load(H3/'cache/oc-retraining/frozen-scores.npz');oldr=json.loads((H3/'results/oc-retraining/retraining.json').read_text())
    column=oldr['names'].index('or_small:'+ATT)
    replay={}
    for role in ['cal','test']:
        replay[role]=oldcore.score(old,data[role+'_X']);np.testing.assert_allclose(replay[role],oldf[f'{SEEDS[0]}_{role}'][:,column],atol=1e-12,rtol=0)
    old_point=point(y,replay['test'].reshape(-1,1),[threshold(replay['cal'])],['H3-attention-replay'])[0]
    s=values[f'{SEEDS[0]}_test'];groups=[np.flatnonzero(y==k) for k in (0,1)];rng=np.random.default_rng(SEEDS[0])
    indices=np.column_stack([rng.choice(g,(10000,len(g))) for g in groups]);auc=np.empty((10000,len(names)));partial=np.empty_like(auc)
    boot=module('a2_boot',ROOT/'configs/original-positions/core.py')
    for j,name in enumerate(names):
        for start in range(0,10000,100):auc[start:start+100,j],partial[start:start+100,j]=boot.bootstrap_metrics(y,s[:,j],indices[start:start+100])
        print('OC_A2_BOOT',name,flush=True)
    comparisons=[];pred=s>np.array(cuts[str(SEEDS[0])]);family=len(CONTRASTS);p=metrics[str(SEEDS[0])]
    for label,terms in CONTRASTS:
        weights=np.array([sum(w for n,w in terms if n==name) for name in names]);delta=partial@weights;ci=np.quantile(delta,[.025/family,1-.025/family]);lo,hi=ci
        item={'id':label,'terms':terms,'delta_pauc':float(sum(p[j]['pauc']*w for j,w in enumerate(weights))),
            'pauc_delta_ci':ci.tolist(),'auroc_delta_ci95':np.quantile(auc@weights,[.025,.975]).tolist(),
            'decision':'positive' if lo>0 else 'worse' if hi<0 else 'unconfirmed'}
        for k,key in [(0,'fpr'),(1,'tpr')]:
            ix=indices[:,:1000] if k==0 else indices[:,1000:];d=pred.astype(float)@weights
            item[key+'_delta_ci95']=np.quantile(d[ix].mean(1),[.025,.975]).tolist()
        comparisons.append(item)
    np.savez(CACHE/'frozen-scores.npz',y=y,**values);np.savez(CACHE/'bootstrap-distributions.npz',auc=auc,pauc=partial)
    write(OUT/'evaluation.json',{'run_id':'OC-A2','evidence':'historical_diagnostic','names':names,'counts':[1000,500],
        'primary_seed':SEEDS[0],'metrics':p,'seed_metrics':metrics,'thresholds':cuts,'comparisons':comparisons,
        'family':family,'repeats':10000,'bootstrap_seed':SEEDS[0],'pauc_ci_coverage':1-.05/family,
        'training_sha256':sealed,'source_lock_sha256':digest(OUT/'source-lock.json'),
        'scores_sha256':digest(CACHE/'frozen-scores.npz'),'bootstrap_sha256':digest(CACHE/'bootstrap-distributions.npz'),
        'historical_replay':old_point,'h3_reproduced_parameters_and_scores':True,'independent_confirmation':False})
    print('OC_A2_SCORED',flush=True)
if __name__=='__main__':main()
