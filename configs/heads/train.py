"""Matched train-only CV; fixed full coordinates, no heldout hyperparameter choice."""
import json,time,joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from head_core import (ROOT,digest,write,sources,check_sources,verify_cache,arrays,matrix,
    grid,center_kernel,span_features,fit_kernel,predict_kernel,squared_distances,selection,make_folds,p_auc)

def main():
    import torch,sklearn,scipy
    torch.backends.cuda.matmul.allow_tf32=False;began=time.monotonic();snapshot=sources();verify_cache()
    out=ROOT/'results/heads';out.mkdir(exist_ok=True);assert not (out/'freeze.json').exists()
    a=arrays();rows=a[0];ti=np.array([i for i,r in enumerate(rows) if r['role']=='train']);di=np.array([i for i,r in enumerate(rows) if r['role']=='dev'])
    assert len(ti)==1600 and len(di)==600
    y=np.array([rows[i]['label'] for i in ti]);yd=np.array([rows[i]['label'] for i in di]);groups=np.array([rows[i]['group_id'] for i in ti]);folds=make_folds(y,groups)
    x=matrix(a,ti);xd=matrix(a,di);tx=torch.as_tensor(x,device='cuda');k=(tx@tx.T).cpu().numpy();kd=(torch.as_tensor(xd,device='cuda')@tx.T).cpu().numpy();del tx
    norms=np.diag(k);d=squared_distances(k,norms,norms);dd=squared_distances(kd,(xd*xd).sum(1),norms)
    # Train-only real pilot: raw-coordinate and exact span/precomputed formulations.
    probe=np.r_[np.flatnonzero(y==0)[:16],np.flatnonzero(y==1)[:16]];valid=np.r_[np.flatnonzero(y==0)[16:24],np.flatnonzero(y==1)[16:24]];checks=[]
    for method in ('logistic','svm'):
        c={'name':'pilot','method':method,'value':1.};h=fit_kernel(k[np.ix_(probe,probe)],y[probe],c,cuda=True)
        pred=predict_kernel(h,k[np.ix_(valid,probe)]);mu=x[probe].mean(0)
        raw=LogisticRegression(C=1.,solver='lbfgs',tol=1e-9,max_iter=5000) if method=='logistic' else SVC(kernel='linear',C=1.,tol=1e-7)
        raw.fit(x[probe]-mu,y[probe]);rp=raw.decision_function(x[valid]-mu);delta=float(np.max(np.abs(pred-rp)));assert delta<2e-4,(method,delta)
        checks.append({'method':method,'train_rows':32,'valid_rows':16,'max_raw_formulation_difference':delta})
    configs=grid();oof=np.full((36,2,1600),np.nan);records={c['name']:[] for c in configs};span_log=[]
    for fi,f in enumerate(folds):
        ix=f['fit'];vi=f['valid'];kf=k[np.ix_(ix,ix)];kv=k[np.ix_(vi,ix)];df=d[np.ix_(ix,ix)];dv=d[np.ix_(vi,ix)]
        span=span_features(center_kernel(kf)[0],cuda=True);span_log.append({'repeat':f['repeat'],'fold':f['fold'],'fit_rows':len(ix),'rank':span[2],'relative_gram_error':span[3]})
        for ci,c in enumerate(configs):
            h=fit_kernel(kf,y[ix],c,df,span if c['method']=='logistic' else None,cuda=True);s=predict_kernel(h,kv,dv);oof[ci,f['repeat'],vi]=s
            rec={'repeat':f['repeat'],'fold':f['fold'],'auroc':float(roc_auc_score(y[vi],s)),'pauc':p_auc(y[vi],s),'fit_seconds':h['fit_seconds'],'iterations':h.get('iterations'),'support_count':h.get('support_count'),'median':h['median']}
            records[c['name']].append(rec)
        print('CV_FOLD_DONE',fi+1,'/10','SEC',round(time.monotonic()-began,1),flush=True)
    assert np.isfinite(oof).all();stats=[]
    for c in configs:
        vals=[r['pauc'] for r in records[c['name']]];stats.append({**c,'cv_mean':float(np.mean(vals)),'cv_std':float(np.std(vals,ddof=1)),'cv_auroc':float(np.mean([r['auroc'] for r in records[c['name']]])),'folds':records[c['name']]})
    chosen=selection(stats);chosen['selected_head']='ridge_tuned' if chosen['selected']==chosen['winners']['ridge'] else next(m for m,n in chosen['winners'].items() if n==chosen['selected'])
    write(out/'selection.json',chosen);byname={c['name']:c for c in configs};specs=[('B_original',next(c for c in configs if c['method']=='ridge' and c['value']==1.))]+[('ridge_tuned' if m=='ridge' else m,byname[n]) for m,n in chosen['winners'].items()]
    fullspan=span_features(center_kernel(k)[0],cuda=True);models=[];head_checks=[]
    for label,c in specs:
        h=fit_kernel(k,y,c,d,fullspan if c['method']=='logistic' else None,cuda=True)
        score=predict_kernel(h,kd,dd)
        if c['method']!='rbf':
            mu=x.mean(0);tg=torch.as_tensor(x-mu,device='cuda');coef=(tg.T@torch.as_tensor(h['beta'],device='cuda')).cpu().numpy();del tg
            h['mean']=mu;h['coef']=coef;replay=(xd-mu)@coef+h['intercept'];delta=float(np.max(np.abs(score-replay)));assert delta<1e-7,delta
            head_checks.append({'candidate':label,'maximum_primal_dual_difference':delta})
        h['training_ids']=[rows[i]['id'] for i in ti];h['training_norm2']=norms
        path=out/(label+'.joblib');joblib.dump(h,path)
        models.append({'name':label,'method':c['method'],'source_candidate':c['name'],'parameters':c,'file':path.name,'sha256':digest(path),'cv':next(v for v in stats if v['name']==c['name']),'dev_auroc':float(roc_auc_score(yd,score)),'dev_pauc':p_auc(yd,score),'iterations':h.get('iterations'),'support_count':h.get('support_count'),'rank':h.get('rank'),'median':h['median']})
        print('FROZEN_HEAD',label,c,'DEV',models[-1]['dev_auroc'],flush=True)
    old=joblib.load(ROOT/'results/routing/fixed_template.joblib');b=joblib.load(out/'B_original.joblib');identity=float(np.max(np.abs(((xd-b['mean'])@b['coef']+b['intercept'])-((xd-old['mean'])@old['coef']+old['intercept']))));assert identity<1e-7
    cv={'candidates':stats,'folds':[{n:(v.tolist() if isinstance(v,np.ndarray) else v) for n,v in f.items()} for f in folds],'span_checks':span_log,'real_raw_checks':checks,'full_span':{'rank':fullspan[2],'relative_gram_error':fullspan[3]},'head_checks':head_checks,'maximum_old_B_dev_score_difference':identity}
    write(out/'cv.json',cv);np.savez_compressed(out/'oof.npz',scores=oof,labels=y,training_ids=np.array([rows[i]['id'] for i in ti]),names=np.array([c['name'] for c in configs]))
    deps=['configs/heads/manifest.json','configs/benchmarks/v1.json','configs/routing/manifest.json','configs/artifacts-lock.json','logs/heads-cache-seal.json','logs/heads-pilot.json','logs/routing-cache-seal.json','results/routing/fixed_template.joblib','results/routing/heldout-scores.jsonl','results/heads/cv.json','results/heads/oof.npz','results/heads/selection.json','docs/superpowers/plans/2026-10-29-detector-heads.md']
    f={'status':'frozen_before_benchmark_and_fresh_confirmation','sources':snapshot,'dependencies':{n:digest(ROOT/n) for n in deps},'candidates':models,'selection':chosen,'settings':36,'fold_fits':360,'versions':{'sklearn':sklearn.__version__,'scipy':scipy.__version__,'numpy':np.__version__,'torch':torch.__version__},'seconds':time.monotonic()-began}
    check_sources(snapshot);write(out/'freeze.json',f);print('HEADS_FROZEN',chosen,'SEC',f['seconds'],flush=True)

if __name__=='__main__':main()
