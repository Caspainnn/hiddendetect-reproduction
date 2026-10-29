"""Train on exactly prior1600; OOF expert scores fit small late-fusion heads."""
import itertools
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from fusion_core import ROOT,EXPERTS,LAYOUT,digest,make_folds,expert_matrix,fit_expert,score_expert,fit_meta,meta_scores
from seal import verify_seal
sys.path.insert(0,str(ROOT/'configs/signal-selection'))
from experiment_core import conservative_threshold
from provenance import source_snapshot,verify_sources,verify_cache_binding

def main():
    started=time.monotonic(); sources=source_snapshot(); out=ROOT/'results/fusion'; out.mkdir(parents=True,exist_ok=True)
    if (out/'freeze.json').exists(): print('ALREADY_FROZEN'); return
    old=ROOT/'cache/highdim'; new=ROOT/'cache/fusion'
    verify_cache_binding(old,ROOT/'configs/highdim/manifest.json')
    verify_cache_binding(new,ROOT/'configs/fusion/manifest.json')
    prior_path=ROOT/'configs/highdim/manifest.json'; rows=json.loads(prior_path.read_text())['samples']
    train=np.array([i for i,x in enumerate(rows) if x['role']=='train']); dev=np.array([i for i,x in enumerate(rows) if x['role']=='dev'])
    y=np.array([rows[i]['label'] for i in train]); yd=np.array([rows[i]['label'] for i in dev])
    assert len(train)==1600 and len(dev)==600 and len(set(rows[i]['group_id'] for i in train))==1600
    cache={name:np.load(old/(name+'.npy'),mmap_mode='r') for name in LAYOUT}
    folds=make_folds(y); fold_ids=np.full(len(train),-1)
    for fold,(_,valid) in enumerate(folds): fold_ids[valid]=fold
    oof=np.full((len(train),4),np.nan); dev_scores=np.zeros((len(dev),4)); experts=[]; fit_log=[]
    directory=ROOT/'cache/fusion-folds'; directory.mkdir(exist_ok=True)
    for column,kind in enumerate(EXPERTS):
        x=expert_matrix(cache,train,kind)
        for fold,(fit,valid) in enumerate(folds):
            began=time.monotonic(); head=fit_expert(x[fit],y[fit],kind)
            assert not set(fit)&set(valid)
            oof[valid,column]=score_expert(head,x[valid],kind)
            path=directory/('fold%d_%s.joblib'%(fold,kind)); joblib.dump(head,path)
            fit_log.append({'expert':kind,'fold':fold,'train_indices':fit.tolist(),'valid_indices':valid.tolist(),
                'private_head_sha256':digest(path),'seconds':time.monotonic()-began})
            print('OOF_EXPERT_DONE',kind,fold,flush=True)
        head=fit_expert(x,y,kind); del x
        path=out/('expert_'+kind+'.joblib'); joblib.dump(head,path)
        dev_scores[:,column]=score_expert(head,expert_matrix(cache,dev,kind),kind)
        experts.append({'name':kind,'file':path.name,'sha256':digest(path),
            'oof_auroc':float(roc_auc_score(y,oof[:,column])),'dev_auroc':float(roc_auc_score(yd,dev_scores[:,column]))})
    assert np.isfinite(oof).all() and np.all(fold_ids>=0)
    np.savez(out/'oof.npz',scores=oof,labels=y,fold_ids=fold_ids,source_indices=train)
    np.savez(out/'dev.npz',scores=dev_scores,labels=yd,source_indices=dev)
    (out/'oof-provenance.json').write_text(json.dumps({'folds':5,'seed':20261030,'training_ids':[rows[i]['id'] for i in train],
        'fit_log':fit_log,'scope':'each training expert score excludes that row label; meta training itself is not a second OOF evaluation'},indent=2)+'\n')
    candidates=[]; additive=[]
    for bits in range(1,16):
        mask=np.array([bool(bits&(1<<i)) for i in range(4)]); name='linear_'+''.join(k for k,take in zip(EXPERTS,mask) if take)
        head=fit_meta(oof,y,mask,False); path=out/(name+'.joblib'); joblib.dump(head,path)
        entry={'name':name,'kind':'meta','file':path.name,'sha256':digest(path),'mask':mask.astype(int).tolist(),
            'interactions':False,'dev_auroc':float(roc_auc_score(yd,meta_scores(head,dev_scores)))}
        candidates.append(entry); additive.append(entry)
    head=fit_meta(oof,y,np.ones(4,bool),True); path=out/'pairwise_BPAT.joblib'; joblib.dump(head,path)
    candidates.append({'name':'pairwise_BPAT','kind':'meta','file':path.name,'sha256':digest(path),'mask':[1]*4,
        'interactions':True,'dev_auroc':float(roc_auc_score(yd,meta_scores(head,dev_scores)))})
    for column,kind in enumerate(EXPERTS):
        candidates.append({'name':'expert_'+kind,'kind':'expert','column':column,'dev_auroc':float(roc_auc_score(yd,dev_scores[:,column]))})
    equal={'mean':oof.mean(0),'scale':oof.std(0)}; equal['scale'][equal['scale']==0]=1
    path=out/'equal.npz'; np.savez(path,**equal)
    candidates.append({'name':'equal_BPAT','kind':'equal','file':path.name,'sha256':digest(path),
        'dev_auroc':float(roc_auc_score(yd,((dev_scores-equal['mean'])/equal['scale']).mean(1)))})
    old_freeze=json.loads((ROOT/'results/highdim/freeze.json').read_text()); old_candidates={x['name']:x for x in old_freeze['candidates']}
    for name,kind in [('raw_ridge_positions','legacy_raw_positions'),('linear_FC64','legacy_FC64')]:
        source=ROOT/'results/highdim'/old_candidates[name]['file']; assert digest(source)==old_candidates[name]['sha256']
        target=out/(name+'.joblib'); target.write_bytes(source.read_bytes())
        candidates.append({'name':name,'kind':kind,'file':target.name,'sha256':digest(target),
            'source_freeze_sha256':digest(ROOT/'results/highdim/freeze.json'),'scope':'same exact prior training1600; pipeline comparator'})
    # Error complementarity is an oracle-style diagnostic, not a deployable union bound.
    thresholds=[conservative_threshold(oof[y==0,column],.05) for column in range(4)]
    predicted=oof>np.array(thresholds); base_fn=(y==1)&~predicted[:,0]; base_fp=(y==0)&predicted[:,0]
    complement={'scope':'training expert OOF only; oracle rescue counts do not claim deployable combined performance',
        'base_false_negatives':int(base_fn.sum()),'base_false_positives':int(base_fp.sum()),
        'experts':{kind:{'rescue_base_FN':int((base_fn&predicted[:,column]).sum()),
                        'reject_base_FP':int((base_fp&~predicted[:,column]).sum())} for column,kind in enumerate(EXPERTS)}}
    (out/'complementarity.json').write_text(json.dumps(complement,indent=2)+'\n')
    best=max(additive,key=lambda x:(x['dev_auroc'],-sum(x['mask']),-additive.index(x)))['name']
    freeze={'status':'frozen_before_new_calibration_and_test_scoring','manifest_sha256':digest(ROOT/'configs/fusion/manifest.json'),
        'source_sha256':sources,
        'old_training_manifest_sha256':digest(prior_path),'old_cache_seal_sha256':digest(old/'seal.json'),
        'new_cache_seal_sha256':digest(new/'seal.json'),'train_code_sha256':digest(__file__),
        'fusion_core_sha256':digest(Path(__file__).with_name('fusion_core.py')),'train_rows':1600,'dev_rows':600,
        'experts':experts,'candidates':candidates,'best_dev_additive_subset':best,'oof_sha256':digest(out/'oof.npz'),
        'dev_sha256':digest(out/'dev.npz'),'provenance_sha256':digest(out/'oof-provenance.json'),
        'complementarity_sha256':digest(out/'complementarity.json'),'equal_sha256':digest(out/'equal.npz'),
        'legacy_scaler_sha256':digest(ROOT/'results/highdim/scaler.npz'),'legacy_centroids_sha256':digest(ROOT/'results/highdim/centroids.npy'),
        'legacy_schema_sha256':digest(ROOT/'results/highdim/schema.json'),'data_usage_sha256':digest(ROOT/'logs/fusion-data-usage.json'),
        'seconds':time.monotonic()-started,'meta_protocol':'5-fold OOF experts train meta; full1600 experts refit for deployment; C=.1 fixed',
        'calibration_protocol':'separate new Aegis-safe400 and Toxic-safe400; no target-positive/test calibration'}
    verify_sources(sources)
    tmp=out/'freeze.tmp'; tmp.write_text(json.dumps(freeze,indent=2)+'\n'); tmp.replace(out/'freeze.json')
    print('PRETEST_FREEZE_COMPLETE',len(candidates),'BEST_DEV',best,'SECONDS',freeze['seconds'],flush=True)

if __name__=='__main__': main()
