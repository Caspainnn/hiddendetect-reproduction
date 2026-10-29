"""Meta-train score corrections and an equal-label-budget full-vector baseline."""
import json,time,joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from anchored_core import (ROOT,FAMILIES,digest,write,sources,check_sources,verify_cache,arrays,matrices,candidates,
    fit_alignment,fuse,select,make_folds,p_auc,fit_kernel,predict_kernel)

def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;began=time.monotonic();snapshot=sources();verify_cache(True)
    out=ROOT/'results/anchored';out.mkdir(exist_ok=True);assert not (out/'freeze.json').exists()
    a=arrays();rows=a[0];di=np.array([i for i,r in enumerate(rows) if r['role']=='meta_train']);assert len(di)==600
    scores=np.load(ROOT/'cache/anchored/base-scores.npy')[di];y=np.array([rows[i]['label'] for i in di]);groups=np.array([rows[i]['group_id'] for i in di]);folds=make_folds(y,groups);cs=candidates();oof=np.full((55,2,600),np.nan);stats=[]
    meta=[{**{k:rows[i][k] for k in ('id','group_id','role','label')},'scores':dict(zip(FAMILIES,map(float,scores[j])))} for j,i in enumerate(di)]
    (out/'meta-scores.jsonl').write_bytes((''.join(json.dumps(r)+'\n' for r in meta)).encode())
    for ci,c in enumerate(cs):
        family=c['family'];e=scores[:,FAMILIES.index(family)];vals=[]
        for f in folds:
            ix,v=f['fit'],f['valid'];h=fit_alignment(scores[ix,0],e[ix],y[ix]);s=fuse(scores[v,0],e[v],c,h);oof[ci,f['repeat'],v]=s
            vals.append({'auroc':float(roc_auc_score(y[v],s)),'pauc':p_auc(y[v],s)})
        stats.append({**c,'cv_mean':float(np.mean([r['pauc'] for r in vals])),'cv_std':float(np.std([r['pauc'] for r in vals],ddof=1)),'folds':vals})
    selection=select(stats);byname={c['name']:c for c in cs};models=[{'name':'B','parameters':cs[0],'alignment':None}]
    models += [{'name':'joint_'+family,'parameters':{'mode':'raw','family':family},'alignment':None} for family in FAMILIES[1:]]
    for label,name in selection['winners'].items():
        c=byname[name];e=scores[:,FAMILIES.index(c['family'])];models.append({'name':label,'parameters':c,'alignment':fit_alignment(scores[:,0],e,y)})
    selection['selected_head']='B' if selection['selected']=='B' else next(c['name'] for c in models if c['parameters'].get('name')==selection['selected'])
    # Equal total supervision: the same frozen draw0 1600 plus these meta600.
    oldrows=json.loads((ROOT/'configs/stability/manifest.json').read_text())['samples'];head=joblib.load(ROOT/'results/stability/0_ridge_B.joblib');ti=head['training_indices'];oldB=np.load(ROOT/'cache/stability/B.npy',mmap_mode='r')
    bt=np.asarray(oldB[ti],np.float64).reshape(1600,-1)/np.sqrt(32);bd,_=matrices(a,di);x=np.concatenate([bt,bd]);del bt,bd
    yt=np.array([oldrows[int(i)]['label'] for i in ti]+y.tolist());gt=np.array([oldrows[int(i)]['group_id'] for i in ti]+groups.tolist());assert len(set(gt))==2200
    tx=torch.as_tensor(x,device='cuda');gram=(tx@tx.T).cpu().numpy();del x,tx
    rf=make_folds(yt,gt);lambdas=[1e4,1e3,1e2,10.,1.,.1,.01,.001,.0001];ro=np.full((9,2,2200),np.nan);refitstats=[]
    for ci,value in enumerate(lambdas):
        vals=[]
        for f in rf:
            ix,v=f['fit'],f['valid'];h=fit_kernel(gram[np.ix_(ix,ix)],yt[ix],{'method':'ridge','value':value},cuda=True);s=predict_kernel(h,gram[np.ix_(v,ix)]);ro[ci,f['repeat'],v]=s;vals.append({'auroc':float(roc_auc_score(yt[v],s)),'pauc':p_auc(yt[v],s)})
        refitstats.append({'value':value,'cv_mean':float(np.mean([r['pauc'] for r in vals])),'folds':vals})
    best=max(range(9),key=lambda i:refitstats[i]['cv_mean']);h=fit_kernel(gram,yt,{'method':'ridge','value':lambdas[best]},cuda=True);h.update(old_training_indices=ti,meta_indices=di,training_ids=head['training_ids']+[rows[i]['id'] for i in di])
    joblib.dump(h,out/'B_refit.joblib',compress=3);models.append({'name':'B_refit','parameters':{'method':'ridge','value':lambdas[best]},'alignment':None})
    assert len(models)==11 and np.isfinite(oof).all() and np.isfinite(ro).all()
    serialize=lambda fs:[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in f.items()} for f in fs]
    cv={'candidates':stats,'folds':serialize(folds),'refit_candidates':refitstats,'refit_folds':serialize(rf),'refit_selected_index':best,'refit_labels':yt.tolist(),'refit_groups':gt.tolist(),'selection':selection}
    write(out/'cv.json',cv);np.savez_compressed(out/'oof.npz',scores=oof,refit_scores=ro);write(out/'selection.json',selection)
    deps=['configs/anchored/manifest.json','configs/benchmarks/v1.json','configs/artifacts-lock.json','logs/anchored-duplicates.json','logs/anchored-cache-seal.json','logs/anchored-score-seal.json','logs/stability-cache-seal.json','results/stability/freeze.json','results/stability/0_ridge_B.joblib','results/stability/0_ridge_mean.joblib','results/stability/0_ridge_dynamic.joblib','results/stability/0_ridge_random.joblib','results/anchored/meta-scores.jsonl','results/anchored/cv.json','results/anchored/oof.npz','results/anchored/selection.json','docs/superpowers/plans/2026-10-29-anchored-summary.md']
    check_sources(snapshot);write(out/'freeze.json',{'status':'frozen_before_confirmation_scoring','sources':snapshot,'dependencies':{n:digest(ROOT/n) for n in deps},'models':models,'refit_sha256':digest(out/'B_refit.joblib'),'selection':selection,'meta_fold_conditions':550,'refit_fold_fits':90,'seconds':time.monotonic()-began})
    print('ANCHORS_FROZEN',selection,'REFIT_LAMBDA',lambdas[best],'SEC',time.monotonic()-began,flush=True)

if __name__=='__main__':main()
