"""No benchmark/confirmation scores are read by this train/dev-only program."""
import json,time,joblib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.metrics import roc_auc_score
from core import ROOT,digest,benchmark,source_snapshot,verify_sources,verify_cache_binding,cache,standardized,joined,p_auc,raw_fit
from search import search,key

def main():
    began=time.monotonic();out=ROOT/'results/swarm';out.mkdir(exist_ok=True)
    assert not (out/'freeze.json').exists(),'Already frozen; do not rerun'
    sources=source_snapshot();benchmark()
    for family in ('fusion','highdim'):
        legacy=json.loads((ROOT/'results'/family/'freeze.json').read_text())
        for entry in legacy['candidates']+legacy.get('experts',[]):
            if 'file' in entry:assert digest(ROOT/'results'/family/entry['file'])==entry['sha256']
    verify_cache_binding(ROOT/'cache/highdim',ROOT/'configs/highdim/manifest.json')
    rows=json.loads((ROOT/'configs/highdim/manifest.json').read_text())['samples'];groups=json.loads((ROOT/'results/highdim/schema.json').read_text())
    ti=np.array([x['index'] for x in rows if x['role']=='train']);di=np.array([x['index'] for x in rows if x['role']=='dev'])
    yt=np.array([rows[i]['label'] for i in ti]);yd=np.array([rows[i]['label'] for i in di])
    with np.load(ROOT/'results/fusion/oof.npz') as data:
        assert np.array_equal(data['source_indices'],ti) and np.array_equal(data['labels'],yt);bt=data['scores'][:,0]
    with np.load(ROOT/'results/fusion/dev.npz') as data:
        assert np.array_equal(data['source_indices'],di) and np.array_equal(data['labels'],yd);bd=data['scores'][:,0]
    mean=float(bt.mean());scale=float(bt.std());assert scale>0
    arrays=cache('highdim');zt=standardized(arrays,ti);zd=standardized(arrays,di);del arrays
    assert len(ti)==1600 and len(di)==600 and len(groups)==259 and zt.shape[1]==19270
    fitted={};executor=ThreadPoolExecutor(max_workers=4);runs=[];candidates=[]
    private=ROOT/'cache/swarm-fits';private.mkdir(exist_ok=True)
    def fit_one(item):
        k,mask=item;t=time.monotonic();x=joined(bt,zt,mask,groups,mean,scale)
        model=raw_fit('linear',x,yt);values=model.decision_function(joined(bd,zd,mask,groups,mean,scale))
        dimensions=sum(g['stop']-g['start'] for take,g in zip(mask,groups) if take)
        value={'dev_pauc':p_auc(yd,values),'dev_auroc':float(roc_auc_score(yd,values)),
            'dimensions':dimensions,'fit_seconds':time.monotonic()-t}
        value['fitness']=value['dev_pauc']-.01*dimensions/zt.shape[1]
        path=private/(k+'.joblib');joblib.dump(model,path)
        return k,value
    def assess_many(masks):
        pending={key(m):m for m in masks if key(m) not in fitted}
        for k,value in executor.map(fit_one,pending.items()):fitted[k]=value
        return [fitted[key(m)]['fitness'] for m in masks]
    def save(name,mask,**extra):
        assess_many([mask]);k=key(mask);path=out/(name+'.joblib');path.write_bytes((private/(k+'.joblib')).read_bytes())
        candidates.append({'name':name,'kind':'anchored','file':path.name,'sha256':digest(path),'mask':mask.astype(int).tolist(),
            'groups':int(mask.sum()),**fitted[k],**extra})
    save('anchor_B_only',np.zeros(len(groups),bool));save('anchor_all',np.ones(len(groups),bool))
    x=joined(bt,zt,np.ones(len(groups),bool),groups,mean,scale);model=raw_fit('linear',x,yt,penalty='l1')
    path=out/'anchor_L1.joblib';joblib.dump(model,path)
    candidates.append({'name':'anchor_L1','kind':'anchored','file':path.name,'sha256':digest(path),'mask':[1]*len(groups),
        'nonzero_coefficients':int((model.coef_!=0).sum()),'dimensions':int((model.coef_!=0).sum()),'groups':None,
        'dev_pauc':p_auc(yd,model.decision_function(joined(bd,zd,np.ones(len(groups),bool),groups,mean,scale)))})
    for seed in range(5):
        for method in ('random','bpso','ga','de'):
            t=time.monotonic();before=len(fitted)
            result=search(method,len(groups),assess_many,seed,budget=120,population=12)
            mask=np.array(result['best_mask'],bool);save(method+'_seed'+str(seed),mask)
            runs.append({'method':method,'seed':seed,**result,'seconds':time.monotonic()-t,'incremental_new_fits':len(fitted)-before})
            (out/'search-progress.json').write_text(json.dumps(runs,indent=2)+'\n')
            print('SEARCH_DONE',method,seed,'pAUC',candidates[-1]['dev_pauc'],'dims',candidates[-1]['dimensions'],'sec',round(runs[-1]['seconds'],2),flush=True)
    executor.shutdown();(out/'search.json').write_text(json.dumps(runs,indent=2)+'\n')
    fusion=json.loads((ROOT/'results/fusion/freeze.json').read_text());old=json.loads((ROOT/'results/highdim/freeze.json').read_text())
    for name in ['expert_B','expert_P','expert_A','expert_T','linear_BPAT','pairwise_BPAT','raw_ridge_positions','linear_FC64']:
        candidates.append({'name':name,'kind':'fusion_reference','reference':name})
    for name in ['linear_F32','linear_last_vector','linear_all','interaction_all','L1','first_logit','original_16_29']+[f'linear_{method}_seed{seed}' for method in ('random','bpso') for seed in range(5)]:
        c=next(x for x in old['candidates'] if x['name']==name)
        candidates.append({'name':'old_'+name,'kind':'highdim_reference','reference':c})
    dependencies=['results/highdim/'+n for n in ('freeze.json','schema.json','scaler.npz','centroids.npy')]
    dependencies+=['results/fusion/'+n for n in ('freeze.json','oof.npz','dev.npz')]
    for path in list((ROOT/'results/fusion').glob('*.joblib'))+list((ROOT/'results/highdim').glob('*.joblib')):
        dependencies.append(path.relative_to(ROOT).as_posix())
    f={'status':'all_candidates_frozen_before_benchmark_and_confirmation_scoring','source_sha256':sources,
        'dependencies_sha256':{n:digest(ROOT/n) for n in dependencies},'benchmark_sha256':digest(ROOT/'configs/benchmarks/v1.json'),
        'search_sha256':digest(out/'search.json'),'manifest_sha256':digest(ROOT/'configs/fusion/manifest.json'),
        'candidates':candidates,'base_mean':mean,'base_scale':scale,
        'training_rows':1600,'dev_rows':600,'groups':259,'extra_dimensions':19270,'seeds':list(range(5)),'unique_budget':120,
        'objective':'standardized dev pAUC max_fpr=.1 -.01 * retained extra coordinate fraction',
        'best_dev_method':max([x for x in candidates if x.get('kind')=='anchored' and 'fitness' in x],key=lambda x:x['fitness'])['name'],
        'seconds':time.monotonic()-began,'total_unique_fits':len(fitted),'benchmark_status':'previously observed fixed comparison set; no untouched claim'}
    verify_sources(sources);(out/'freeze.json').write_text(json.dumps(f,indent=2)+'\n');print('SEARCH_FREEZE_COMPLETE',len(candidates),f['seconds'],flush=True)

if __name__=='__main__':main()
