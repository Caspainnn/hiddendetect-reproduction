"""Train/dev ablations and matched group-mask searches, then immutable freeze."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from features import ROOT, digest, LAYOUT
from detectors import schema, make_matrix, columns_for_mask, raw_fit, raw_scores, fit_detector, detector_scores, fit_centroids
from seal import verify_seal
from ridge import raw_matrix,fit_ridge,ridge_scores
from batch_search import batched_binary_pso
sys.path.insert(0,str(ROOT/'configs/signal-selection'))
from experiment_core import binary_pso, mask_key

def main():
    began=time.monotonic(); out=ROOT/'results/highdim'; out.mkdir(parents=True,exist_ok=True)
    if (out/'freeze.json').exists(): print('ALREADY_FROZEN'); return
    mp=Path(__file__).with_name('manifest.json'); manifest=json.loads(mp.read_text()); rows=manifest['samples']
    directory=ROOT/'cache/highdim'; metadata=json.loads((directory/'metadata.json').read_text())
    assert metadata['manifest_sha256']==digest(mp)
    assert json.loads((directory/'checkpoint.json').read_text())['next_index']==len(rows)
    seal=json.loads((directory/'seal.json').read_text()); verify_seal(directory,seal)
    cache={name:np.load(directory/(name+'.npy'),mmap_mode='r') for name in LAYOUT}
    attention=metadata['provenance']['attention_enabled']; groups=schema(attention); total=groups[-1]['stop']
    train=np.array([i for i,x in enumerate(rows) if x['role']=='train']); dev=np.array([i for i,x in enumerate(rows) if x['role']=='dev'])
    yt=np.array([rows[i]['label'] for i in train]); yd=np.array([rows[i]['label'] for i in dev])
    references=fit_centroids(np.asarray(cache['P'][train,:,0,:]),yt)
    np.save(out/'centroids.npy',references)
    xt=make_matrix(cache,train,references,attention); xd=make_matrix(cache,dev,references,attention)
    mean=xt.mean(0); scale=xt.std(0); scale[scale==0]=1
    zt=(xt-mean)/scale; zd=(xd-mean)/scale
    np.savez(out/'scaler.npz',mean=mean,scale=scale)
    (out/'schema.json').write_text(json.dumps(groups,indent=2)+'\n')
    candidates=[]; runs=[]
    progress=out/'search-progress.json'
    previous_runs=json.loads(progress.read_text()) if progress.exists() else []
    completed={(run['head'],run['method'],run['seed']):run for run in previous_runs}
    workers=8
    executor=ThreadPoolExecutor(max_workers=workers)
    for label,regions in [('last',[0]),('user',[1,2,3,4]),('positions',[0,1,2,3,4])]:
        begin=time.monotonic(); raw=raw_matrix(cache,train,regions); ridge=fit_ridge(raw,yt); del raw
        raw=raw_matrix(cache,dev,regions); auc=float(roc_auc_score(yd,ridge_scores(ridge,raw))); del raw
        path=out/('raw_ridge_'+label+'.joblib'); joblib.dump(ridge,path)
        candidates.append({'name':'raw_ridge_'+label,'kind':'raw_ridge','head':'ridge','file':path.name,'sha256':digest(path),
            'regions':regions,'groups':32*len(regions),'dimensions':32*len(regions)*4096,'dev_auroc':auc,
            'fit_device':ridge['fit_device'],'lambda':1.,'seconds':time.monotonic()-begin})
        print('RAW_VECTOR_CONTROL',label,auc,flush=True)
    for head in ('linear','interaction'):
        fitted={}
        accounting={'objective_calls':0,'cache_hits':0,'new_fits':0,'fit_seconds':0.}
        def fit_one(item):
            key,mask=item; cols=columns_for_mask(mask,groups); fit_start=time.monotonic()
            model=raw_fit(head,zt[:,cols],yt); elapsed=time.monotonic()-fit_start
            auc=float(roc_auc_score(yd,raw_scores(head,model,zd[:,cols])))
            return key,(model,auc,len(cols)),elapsed
        def assess_many(masks):
            accounting['objective_calls']+=len(masks)
            pending={}
            for mask in masks:
                key=mask_key(mask)
                if key in fitted or key in pending: accounting['cache_hits']+=1
                else: pending[key]=mask
            for key,value,elapsed in executor.map(fit_one,pending.items()):
                fitted[key]=value; accounting['new_fits']+=1; accounting['fit_seconds']+=elapsed
            return [fitted[mask_key(mask)][1]-.02*fitted[mask_key(mask)][2]/total for mask in masks]
        def assess(mask): return assess_many([mask])[0]
        def save(name,mask):
            value=assess(mask); model,auc,dimensions=fitted[mask_key(mask)]
            path=out/(name+'.joblib'); joblib.dump(model,path)
            candidates.append({'name':name,'kind':'groups','head':head,'file':path.name,'sha256':digest(path),
                'mask':mask.astype(int).tolist(),'groups':int(mask.sum()),'dimensions':dimensions,'dev_auroc':auc,'fitness':value})
        predicates=[('length_only',lambda g:g['family']=='length'),('F32',lambda g:g['family']=='F'),
            ('FC64',lambda g:g['family'] in ('F','C')),('last_vector',lambda g:g['family']=='H' and g['region']==0),
            ('user_vectors',lambda g:g['family']=='H' and g['region']!=0),('position_vectors',lambda g:g['family']=='H'),
            ('position_trajectory',lambda g:g['family'] in ('H','trajectory')),
            ('position_trajectory_module',lambda g:g['family'] in ('H','trajectory','module'))]
        if attention: predicates.append(('attention_only',lambda g:g['family']=='attention'))
        predicates.append(('all',lambda g:True))
        for label,predicate in predicates:
            mask=np.array([predicate(g) for g in groups]); save(head+'_'+label,mask)
            print('ABLATION',head,label,candidates[-1]['dev_auroc'],flush=True)
        if head=='linear':
            model=raw_fit(head,zt,yt,penalty='l1'); path=out/'L1.joblib'; joblib.dump(model,path)
            nonzero=model.coef_[0]!=0
            candidates.append({'name':'L1','kind':'groups','head':head,'file':path.name,'sha256':digest(path),
                'mask':[1]*len(groups),'groups':sum(bool(nonzero[g['start']:g['stop']].any()) for g in groups),
                'dimensions':int(nonzero.sum()),'dev_auroc':float(roc_auc_score(yd,model.decision_function(zd))),
                'count_scope':'nonzero coefficients; implementation still reads full feature vector'})
        for seed in range(5):
            for method in ('random','bpso'):
                if (head,method,seed) in completed:
                    run=completed[(head,method,seed)]; mask=np.array(run['best_mask'],bool)
                    assert np.isclose(assess(mask),run['best_fitness'],rtol=0,atol=1e-8)
                    save('%s_%s_seed%d'%(head,method,seed),mask)
                    runs.append({**run,'resumed_completed_run':True})
                    print('RESUMED_COMPLETED_SEARCH',head,method,seed,flush=True)
                    continue
                start=time.monotonic()
                before=accounting.copy()
                if method=='bpso':
                    result=batched_binary_pso(len(groups),assess_many,seed=seed,budget=100,particles=10,rounds=10)
                    mask=np.array(result['best_mask'],bool)
                else:
                    rng=np.random.default_rng(seed); proposals={}
                    while len(proposals)<100:
                        proposal=rng.random(len(groups))<.5
                        if not proposal.any(): continue
                        key=mask_key(proposal)
                        if key not in proposals: proposals[key]=proposal
                    values=dict(zip(proposals,assess_many(list(proposals.values()))))
                    key=max(values,key=values.get); mask=np.array([x=='1' for x in key])
                    result={'best_mask':mask.tolist(),'best_fitness':values[key],'unique_evaluations':len(values),'evaluated_masks':list(values)}
                actual={key:accounting[key]-before[key] for key in accounting}
                name='%s_%s_seed%d'%(head,method,seed); save(name,mask)
                runs.append({'head':head,'method':method,'seed':seed,**result,**actual,'seconds':time.monotonic()-start,
                    'workers':workers,'timing_scope':'incremental shared-cache elapsed; fit_seconds sums concurrent model wall times; counts exclude candidate serialization'})
                # Durable progress has no calibration/test metrics.
                (out/'search-progress.json').write_text(json.dumps(runs,indent=2)+'\n')
                print('SEARCH_DONE',name,result['unique_evaluations'],candidates[-1]['dimensions'],round(time.monotonic()-start,2),flush=True)
        del fitted
    logit=fit_detector('linear',cache['L'][train],yt); path=out/'first_logit.joblib'; joblib.dump(logit,path)
    candidates.append({'name':'first_logit','kind':'logit','head':'linear','file':path.name,'sha256':digest(path),
        'groups':None,'dimensions':32000,'dev_auroc':float(roc_auc_score(yd,detector_scores(logit,cache['L'][dev])))})
    for lo,hi in [(16,29),(17,29)]:
        score=np.trapz(np.asarray(cache['F'][dev,lo:hi+1],dtype=np.float64),axis=1)
        candidates.append({'name':'original_%d_%d'%(lo,hi),'kind':'interval','lo':lo,'hi':hi,'groups':hi-lo+1,
            'dimensions':hi-lo+1,'dev_auroc':float(roc_auc_score(yd,score))})
    (out/'search.json').write_text(json.dumps(runs,indent=2)+'\n')
    executor.shutdown()
    freeze={'status':'frozen_before_calibration_and_test_scores','manifest_sha256':digest(mp),
        'train_code_sha256':digest(__file__),'detectors_code_sha256':digest(Path(__file__).with_name('detectors.py')),
        'cache_metadata_sha256':digest(directory/'metadata.json'),'schema_sha256':digest(out/'schema.json'),
        'cache_seal_sha256':digest(directory/'seal.json'),
        'centroids_sha256':digest(out/'centroids.npy'),'scaler_sha256':digest(out/'scaler.npz'),
        'search_sha256':digest(out/'search.json'),'train_rows':len(train),'dev_rows':len(dev),
        'attention_enabled':attention,'candidate_groups':len(groups),'candidate_dimensions':total,
        'head_settings':{'linear':'C=.01, balanced liblinear, max_iter2000',
            'interaction':'ExtraTrees32, depth3, min_leaf20, sqrt features, balanced, seed0, one CPU thread'},
        'objective':'dev AUROC - .02 * retained scalar dimensions / full scalar dimensions',
        'projection':'fixed4096->64 Rademacher, seed20261029; projected coordinates are not neurons',
        'raw_vector_controls':'three unprojected pooled-vector ridge controls, fixed lambda1, train centering, normalize sqrt(number of groups)',
        'parallel_fits':workers,'resume_scope':'completed searches preserved, identical proposal masks/heads/budgets; no calibration or test selection',
        'seconds':time.monotonic()-began,'candidates':candidates}
    temporary=out/'freeze.tmp'; temporary.write_text(json.dumps(freeze,indent=2)+'\n'); temporary.replace(out/'freeze.json')
    print('PRETEST_FREEZE_COMPLETE',len(candidates),'SECONDS',freeze['seconds'],flush=True)

if __name__=='__main__': main()
