"""Five paired draws, eight variants, nine settings, ten inner folds."""
import json,time,joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from stability_core import (ROOT,digest,write,sources,check_sources,verify_cache,arrays,base,
    summary_features,draws,variants,params,make_folds,center_kernel,span_features,fit_kernel,predict_kernel,p_auc)

def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;began=time.monotonic();snapshot=sources();verify_cache(True)
    out=ROOT/'results/stability';out.mkdir(exist_ok=True);assert not (out/'freeze.json').exists()
    a=arrays();rows=a[0];pool=np.array([i for i,r in enumerate(rows) if r['role']=='pool']);assert len(pool)==3200
    x=base(a,pool);tx=torch.as_tensor(x,device='cuda');gb=(tx@tx.T).cpu().numpy();del x,tx
    grams={'B':gb}
    for family in ('mean','dynamic','random'):
        x=summary_features(np.asarray(a[2][pool]),family);tx=torch.as_tensor(x,device='cuda');grams[family]=gb+(tx@tx.T).cpu().numpy();del x,tx
    lookup={int(v):i for i,v in enumerate(pool)};vs=variants();sets=draws(rows);oof=np.full((5,8,9,2,1600),np.nan);records=[];models=[];foldlog=[];spans=[]
    for si,ix in enumerate(sets):
        pi=np.array([lookup[int(i)] for i in ix]);y=np.array([rows[i]['label'] for i in ix]);groups=np.array([rows[i]['group_id'] for i in ix]);folds=make_folds(y,groups)
        foldlog.append([{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in f.items()} for f in folds])
        for family in ('B','mean','dynamic','random'):
            gram=grams[family][np.ix_(pi,pi)]
            for fi,f in enumerate(folds):
                fit=f['fit'];valid=f['valid'];kf=gram[np.ix_(fit,fit)];kv=gram[np.ix_(valid,fit)];span=span_features(center_kernel(kf)[0],True)
                spans.append({'draw':si,'family':family,'fold_index':fi,'rank':span[2],'relative_gram_error':span[3]})
                for method in ('ridge','logistic'):
                    vi=next(j for j,v in enumerate(vs) if v['method']==method and v['family']==family)
                    for ci,value in enumerate(params(method)):
                        h=fit_kernel(kf,y[fit],{'method':method,'value':value},span=span if method=='logistic' else None,cuda=True)
                        oof[si,vi,ci,f['repeat'],valid]=predict_kernel(h,kv)
            fullspan=span_features(center_kernel(gram)[0],True)
            for method in ('ridge','logistic'):
                vi=next(j for j,v in enumerate(vs) if v['method']==method and v['family']==family);stats=[]
                for ci,value in enumerate(params(method)):
                    foldmetrics=[]
                    for f in folds:
                        valid=f['valid'];s=oof[si,vi,ci,f['repeat'],valid]
                        foldmetrics.append({'auroc':float(roc_auc_score(y[valid],s)),'pauc':p_auc(y[valid],s)})
                    stats.append({'value':value,'folds':foldmetrics,'mean_pauc':float(np.mean([v['pauc'] for v in foldmetrics]))})
                best=max(range(9),key=lambda j:stats[j]['mean_pauc']);name=str(si)+'_'+vs[vi]['name']
                h=fit_kernel(gram,y,{'method':method,'value':params(method)[best]},span=fullspan if method=='logistic' else None,cuda=True)
                h.update(training_indices=ix,training_ids=[rows[i]['id'] for i in ix],family=family,variant=vs[vi]['name'],draw=si)
                path=out/(name+'.joblib');joblib.dump(h,path,compress=3)
                models.append({'name':name,'variant':vs[vi]['name'],'draw':si,'method':method,'family':family,'parameter_index':best,'value':params(method)[best],'file':path.name,'sha256':digest(path)})
                records.append({'draw':si,'variant':vs[vi]['name'],'settings':stats,'selected_index':best})
            print('TRAINED',si,family,'SEC',round(time.monotonic()-began,1),flush=True)
    assert np.isfinite(oof).all() and len(models)==40
    cv={'draws':[ix.tolist() for ix in sets],'folds':foldlog,'records':records,'span_checks':spans,'variants':vs}
    write(out/'cv.json',cv);np.savez_compressed(out/'oof.npz',scores=oof)
    deps=['configs/stability/manifest.json','configs/benchmarks/v1.json','configs/artifacts-lock.json','logs/stability-cache-seal.json','results/stability/cv.json','results/stability/oof.npz','docs/superpowers/plans/2026-10-29-training-stability.md']
    check_sources(snapshot);write(out/'freeze.json',{'status':'frozen_before_scoring','sources':snapshot,'dependencies':{n:digest(ROOT/n) for n in deps},'models':models,'fold_fits':3600,'seconds':time.monotonic()-began})
    print('MODELS_FROZEN',40,'SEC',time.monotonic()-began,flush=True)

if __name__=='__main__':main()
