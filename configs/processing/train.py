"""Repeated train-only CV for a fixed family of user-processing kernels."""
import json,time,joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from processing_core import (ROOT,digest,write,sources,check_sources,verify_cache,arrays,
    baseline,feature_matrix,relation_transform,candidates,select,make_folds,p_auc,fit_joint,predict_joint)

def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;began=time.monotonic();snapshot=sources();verify_cache()
    out=ROOT/'results/processing';out.mkdir(exist_ok=True);assert not (out/'freeze.json').exists()
    a=arrays();rows=a[0];ti=np.array([i for i,r in enumerate(rows) if r['role']=='train']);di=np.array([i for i,r in enumerate(rows) if r['role']=='dev'])
    assert len(ti)==1600 and len(di)==600
    y=np.array([rows[i]['label'] for i in ti]);yd=np.array([rows[i]['label'] for i in di]);groups=np.array([rows[i]['group_id'] for i in ti]);folds=make_folds(y,groups)
    b=baseline(a,ti);bd=baseline(a,di);s=np.asarray(a[1][ti]);sd=np.asarray(a[1][di]);raw=np.asarray(a[2][ti],np.float64);rd=np.asarray(a[2][di],np.float64)
    tb=torch.as_tensor(b,device='cuda');gb=tb@tb.T;del tb
    grams={}
    for family in ('mean','dynamic','processed','relations','random','duplicate_mean'):
        if family=='relations':grams[family]=grams['processed'];continue
        x=feature_matrix(s,family);tx=torch.as_tensor(x,device='cuda');grams[family]=tx@tx.T;del x,tx
    gamma_checks=float(torch.max(torch.abs(grams['mean']-grams['duplicate_mean'])).cpu());assert gamma_checks<1e-10
    oof=np.full((13,2,len(ti)),np.nan);specs=[];fitlog=[];equivalence=[];target=torch.as_tensor(y*2-1,dtype=torch.float64,device='cuda')
    for ci,c in enumerate(candidates()):
        family=c['family'];g=gb.clone()
        if family!='B':g+=c['gamma']*grams[family]
        stats=[]
        for fi,f in enumerate(folds):
            full=g
            if family=='relations':
                _,z,_,_=relation_transform(raw,f['fit']);tz=torch.as_tensor(z,device='cuda');full=g+c['gamma']*.1*(tz@tz.T)
            ix=torch.as_tensor(f['fit'],device='cuda');vi=torch.as_tensor(f['valid'],device='cuda')
            ff=full[ix[:,None],ix];vf=full[vi[:,None],ix];mu=ff.mean(0);grand=mu.mean()
            kff=ff-mu[None,:]-mu[:,None]+grand;kff.diagonal().add_(1.)
            kvf=vf-vf.mean(1)[:,None]-mu[None,:]+grand;offset=target[ix].mean()
            pred=(kvf@torch.linalg.solve(kff,target[ix]-offset)+offset).cpu().numpy();oof[ci,f['repeat'],f['valid']]=pred
            stat={'auroc':float(roc_auc_score(y[f['valid']],pred)),'pauc':p_auc(y[f['valid']],pred)};stats.append(stat);fitlog.append({'candidate':c['name'],'repeat':f['repeat'],'fold':f['fold'],**stat})
            if fi==0 and c['name'] in ('B','mean_g1p0','relations_g1p0'):
                x=np.zeros((len(b),0)) if family=='B' else feature_matrix(s,family)
                h=fit_joint(b[f['fit']],x[f['fit']],raw[f['fit']],y[f['fit']],c['gamma'],family=='relations')
                replay=predict_joint(h,b[f['valid']],x[f['valid']],raw[f['valid']]);delta=float(np.max(np.abs(replay-pred)));assert delta<1e-7,delta
                equivalence.append({'candidate':c['name'],'fold':0,'maximum_primal_dual_difference':delta});del x,h
        x=np.zeros((len(b),0)) if family=='B' else feature_matrix(s,family)
        h=fit_joint(b,x,raw,y,c['gamma'],family=='relations');h['family']=family;del x
        path=out/(c['name']+'.joblib');joblib.dump(h,path)
        xd=np.zeros((len(bd),0)) if family=='B' else feature_matrix(sd,family)
        ds=predict_joint(h,bd,xd,rd);del xd
        vals=[v['pauc'] for v in stats];spec={**c,'file':path.name,'sha256':digest(path),'cv_mean':float(np.mean(vals)),'cv_std':float(np.std(vals,ddof=1)),
            'cv_auroc':float(np.mean([v['auroc'] for v in stats])),'folds':stats,'dev_auroc':float(roc_auc_score(yd,ds)),'dev_pauc':p_auc(yd,ds),'extra_dimensions':len(h['coefs'][1])+len(h['coefs'][2])}
        specs.append(spec);print('TRAINED',c['name'],'CV',spec['cv_mean'],'DEV',spec['dev_auroc'],'SEC',round(time.monotonic()-began,1),flush=True)
    assert np.isfinite(oof).all()
    old=np.load(ROOT/'results/routing/oof.npz');ki=old['keys'].tolist().index('0'*32);old_delta=float(np.max(np.abs(oof[0]-old['scores'][ki,1])));assert old_delta<1e-7
    dupdiff={}
    for gname in ('0p25','1p0'):
        one=next(i for i,c in enumerate(specs) if c['name']=='mean_g'+gname);dup=next(i for i,c in enumerate(specs) if c['name']=='duplicate_mean_g'+gname)
        dupdiff[gname]=float(np.max(np.abs(oof[one]-oof[dup])));assert dupdiff[gname]<1e-7
    selection=select(specs);write(out/'selection.json',selection)
    cv={'folds':[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in f.items()} for f in folds],'fits':fitlog,'equivalence_checks':equivalence,'duplicate_gram_difference':gamma_checks,'duplicate_oof_differences':dupdiff,'maximum_old_fixed_B_oof_difference':old_delta}
    write(out/'cv.json',cv);np.savez_compressed(out/'oof.npz',scores=oof,labels=y,training_ids=np.array([rows[i]['id'] for i in ti]),names=np.array([c['name'] for c in specs]))
    deps=['configs/processing/manifest.json','configs/benchmarks/v1.json','configs/routing/manifest.json','configs/artifacts-lock.json','logs/processing-cache-seal.json','logs/processing-initial-error.json','logs/routing-cache-seal.json','results/routing/fixed_template.joblib','results/routing/oof.npz','results/routing/heldout-scores.jsonl','results/processing/cv.json','results/processing/oof.npz','results/processing/selection.json','docs/superpowers/plans/2026-10-29-user-signal-processing.md']
    frozen={'status':'frozen_before_benchmark_and_fresh_confirmation','sources':snapshot,'dependencies':{n:digest(ROOT/n) for n in deps},'candidates':specs,'selection':selection,'train_rows':1600,'dev_rows':600,'fold_fits':130,'seconds':time.monotonic()-began}
    check_sources(snapshot);write(out/'freeze.json',frozen);print('PROCESSING_FROZEN',selection,flush=True)

if __name__=='__main__':main()
