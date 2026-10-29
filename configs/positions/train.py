"""Repeated group CV reads train only; dev is a separate diagnostic, never selection."""
import json,time,hashlib,joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from position_core import ROOT,digest,source_snapshot,sources_match,verify_cache,models,matrix,make_folds,fit_ridge,ridge_scores,p_auc

def head_sha(h):
    sha=hashlib.sha256()
    for name in ('mean','coef'):sha.update(np.asarray(h[name]).tobytes())
    sha.update(str(h['intercept']).encode());return sha.hexdigest()

def main():
    out=ROOT/'results/positions';out.mkdir(exist_ok=True);assert not (out/'freeze.json').exists()
    began=time.monotonic();sources=source_snapshot();seal=verify_cache();mp=ROOT/'configs/positions/manifest.json'
    rows=json.loads(mp.read_text())['samples'];data=np.load(ROOT/'cache/positions/P.npy',mmap_mode='r')
    ti=np.array([i for i,x in enumerate(rows) if x['role']=='train']);di=np.array([i for i,x in enumerate(rows) if x['role']=='dev'])
    y=np.array([rows[i]['label'] for i in ti]);yd=np.array([rows[i]['label'] for i in di]);groups=np.array([rows[i]['group_id'] for i in ti])
    assert len(ti)==1600 and len(di)==600 and len(set(groups))==1600
    folds=make_folds(y,groups);candidates=[];oof=np.full((17,2,1600),np.nan);fit_log=[]
    for k,c in enumerate(models()):
        t=time.monotonic();x=matrix(data,ti,c['positions']);fold_metrics=[]
        for f in folds:
            h=fit_ridge(x[f['fit']],y[f['fit']]);s=ridge_scores(h,x[f['valid']])
            oof[k,f['repeat'],f['valid']]=s
            value={'repeat':f['repeat'],'fold':f['fold'],'auroc':float(roc_auc_score(y[f['valid']],s)),'partial_auroc':p_auc(y[f['valid']],s),'parameter_sha256':head_sha(h)}
            fold_metrics.append(value);fit_log.append({'candidate':c['name'],**value})
        h=fit_ridge(x,y);path=out/(c['name']+'.joblib');joblib.dump(h,path);del x
        ds=ridge_scores(h,matrix(data,di,c['positions']))
        candidate={**c,'file':path.name,'sha256':digest(path),'dimensions':32*4096*len(c['positions']),
            'cv_auroc_mean':float(np.mean([v['auroc'] for v in fold_metrics])),
            'cv_pauc_mean':float(np.mean([v['partial_auroc'] for v in fold_metrics])),
            'cv_pauc_std':float(np.std([v['partial_auroc'] for v in fold_metrics],ddof=1)),
            'dev_auroc':float(roc_auc_score(yd,ds)),'dev_partial_auroc':p_auc(yd,ds),'seconds':time.monotonic()-t}
        candidates.append(candidate);print('CANDIDATE_TRAINED',c['name'],'CV',candidate['cv_pauc_mean'],'SECONDS',round(candidate['seconds'],2),flush=True)
    assert np.isfinite(oof).all()
    np.savez(out/'oof.npz',scores=oof,labels=y,training_ids=np.array([rows[i]['id'] for i in ti]))
    cv={'folds':[{**{k:f[k] for k in ('repeat','fold')},'fit':f['fit'].tolist(),'valid':f['valid'].tolist()} for f in folds],
        'fits':fit_log,'scope':'repeated group CV on train1600 only; fold variance is diagnostic, not independent-fold inference'}
    (out/'cv.json').write_text(json.dumps(cv,indent=2)+'\n')
    eligible=[c for c in candidates if c['eligible']];best=max(eligible,key=lambda c:(c['cv_pauc_mean'],-len(c['positions'])))
    base=next(c for c in candidates if c['name']=='template_last');threshold=best['cv_pauc_mean']-best['cv_pauc_std']/np.sqrt(10)
    chosen=base if base['cv_pauc_mean']>=threshold else best
    bestsingle=max([c for c in candidates if c['real_alternative'] and len(c['positions'])==1],key=lambda c:c['cv_pauc_mean'])
    bestpair=max([c for c in candidates if c['real_alternative'] and len(c['positions'])==2],key=lambda c:c['cv_pauc_mean'])
    selection={'selected':chosen['name'],'raw_best_cv':best['name'],'best_nonlast_single':bestsingle['name'],'best_real_pair':bestpair['name'],
        'one_se_threshold':threshold,'scope':'selection only from train CV, template preferred within one-SE heuristic; overlapping-fold SE not a formal interval'}
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    previous=joblib.load(ROOT/'results/fusion/expert_B.joblib');current=joblib.load(out/'template_last.joblib')
    identity={name:float(np.max(np.abs(np.asarray(current[name])-np.asarray(previous[name])))) for name in ('mean','coef','intercept')}
    probe=np.array([i for i,x in enumerate(rows) if x['role']=='train'])[:64];px=matrix(data,probe,[7])
    score_difference=float(np.max(np.abs(ridge_scores(current,px)-ridge_scores(previous,px))));assert score_difference<=1e-4
    dependencies=['configs/positions/manifest.json','configs/benchmarks/v1.json','configs/highdim/manifest.json','configs/fusion/manifest.json','configs/swarm/manifest.json',
        'results/fusion/expert_B.joblib','results/fusion/heldout-scores.jsonl','results/swarm/heldout-scores.jsonl',
        'logs/positions-cache-seal.json','logs/positions-readout-identity.json','logs/positions-prefix-diagnostic.json','configs/artifacts-lock.json']
    f={'status':'frozen_before_benchmark_and_new_confirmation_scores','source_sha256':sources,'dependencies_sha256':{n:digest(ROOT/n) for n in dependencies},
        'candidates':candidates,'cv_sha256':digest(out/'cv.json'),'oof_sha256':digest(out/'oof.npz'),'selection_sha256':digest(out/'selection.json'),
        'cache_seal_sha256':digest(ROOT/'logs/positions-cache-seal.json'),'identity_sha256':digest(ROOT/'logs/positions-readout-identity.json'),
        'manifest_sha256':digest(mp),'selection':selection,'old_B_parameter_max_differences':identity,'old_B_training_probe_score_difference':score_difference,
        'train_rows':1600,'dev_rows':600,'fold_fits':170,'ridge':'lambda1, train-only centering, norm sqrt(32*readouts)',
        'confirmatory_family':12,'near_optimality_margin_auroc':.005,'seconds':time.monotonic()-began}
    sources_match(sources);(out/'freeze.json').write_text(json.dumps(f,indent=2)+'\n');print('POSITION_MODELS_FROZEN',selection,'SEC',f['seconds'],flush=True)

if __name__=='__main__':main()
