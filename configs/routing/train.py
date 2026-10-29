"""Train-only repeatedCV for matched binary layer routing, no heldout scoring."""
import json,time,joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from routing_core import (ROOT,digest,write,sources,check_sources,verify_seal,load_arrays,
    block,route,matrix,make_folds,p_auc,search,warm_masks,key,fit,scores,replay_selection)

def main():
    import torch
    began=time.monotonic();torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/'results/routing';out.mkdir(exist_ok=True);assert not (out/'freeze.json').exists()
    snapshot=sources();verify_seal();arrays=load_arrays();rows=arrays[0]
    ti=np.array([i for i,r in enumerate(rows) if r['role']=='train']);di=np.array([i for i,r in enumerate(rows) if r['role']=='dev'])
    y=np.array([rows[i]['label'] for i in ti]);yd=np.array([rows[i]['label'] for i in di]);groups=np.array([rows[i]['group_id'] for i in ti])
    assert len(ti)==1600 and len(di)==600
    folds=make_folds(y,groups);grams={};deltas={};cache={};all_oof={}
    for mode in ('variable','fixed'):
        a=block(arrays,ti,mode);g=[];delta=[]
        for l in range(32):
            t=torch.as_tensor(a[:,l,0,:],dtype=torch.float64,device='cuda')/np.sqrt(32);u=torch.as_tensor(a[:,l,1,:],dtype=torch.float64,device='cuda')/np.sqrt(32)
            gt=t@t.T;gu=u@u.T;g.append(gt);delta.append(gu-gt)
        grams[mode]=torch.stack(g).sum(0);deltas[mode]=torch.stack(delta);del a,g,delta
    tensors=[(torch.as_tensor(f['fit'],device='cuda'),torch.as_tensor(f['valid'],device='cuda')) for f in folds]
    target=torch.as_tensor(y*2-1,dtype=torch.float64,device='cuda')
    def assess(masks):
        result=[]
        for mask in masks:
            k=key(mask)
            if k not in cache:
                stats={};oof=np.full((2,2,len(ti)),np.nan)
                for mi,mode in enumerate(('variable','fixed')):
                    g=grams[mode].clone()
                    if np.any(mask):g+=deltas[mode][torch.as_tensor(np.flatnonzero(mask),device='cuda')].sum(0)
                    metrics=[]
                    for fi,(fitix,valix) in enumerate(tensors):
                        ff=g[fitix[:,None],fitix];vf=g[valix[:,None],fitix];mu=ff.mean(0);grand=mu.mean()
                        kff=ff-mu[None,:]-mu[:,None]+grand;kff.diagonal().add_(1)
                        kvf=vf-vf.mean(1)[:,None]-mu[None,:]+grand;offset=target[fitix].mean()
                        pred=(kvf@torch.linalg.solve(kff,target[fitix]-offset)+offset).cpu().numpy()
                        fold=folds[fi];oof[mi,fold['repeat'],fold['valid']]=pred
                        metrics.append({'auroc':float(roc_auc_score(y[fold['valid']],pred)),'pauc':p_auc(y[fold['valid']],pred)})
                    vals=np.array([v['pauc'] for v in metrics]);stats[mode]={'folds':metrics,'mean':float(vals.mean()),'std':float(vals.std(ddof=1))}
                assert np.isfinite(oof).all()
                value=min(s['mean']-.25*s['std'] for s in stats.values())-.002*np.mean(mask)
                cache[k]={'fitness':float(value),'stats':stats};all_oof[k]=oof
                if len(cache)%10==0:print('CV_MASKS',len(cache),'SEC',round(time.monotonic()-began,1),flush=True)
            result.append(cache[k]['fitness'])
        return result
    warm=warm_masks(32);assess(warm)
    # Real-data check: additive dual prediction must match independently fitted explicit heads.
    checks=[]
    for mask in [warm[0],warm[2],np.arange(32)%2==0]:
        assess([mask]);f=folds[0]
        for mi,mode in enumerate(('variable','fixed')):
            x=route(block(arrays,ti,mode),mask);h=fit(x[f['fit']],y[f['fit']]);pred=scores(h,x[f['valid']])
            delta=float(np.max(np.abs(pred-all_oof[key(mask)][mi,0,f['valid']])));assert delta<1e-7,delta
            checks.append({'mask':key(mask),'mode':mode,'maximum_primal_dual_score_difference':delta})
    runs=[]
    for method in ('random','bpso','ga','de','scan'):
        for seed in range(20261201,20261206):
            r=search(method,32,assess,seed);runs.append(r);print('SEARCH_DONE',method,seed,r['best_fitness'],sum(r['best_mask']),flush=True)
    baseline='0'*32
    single=max([key(m) for m in warm[2:]],key=lambda k:cache[k]['fitness'])
    specs=[('template',baseline),('all_user','1'*32),('single',single)]+[(r['method']+'_'+str(r['seed']),key(r['best_mask'])) for r in runs]
    best=max([k for _,k in specs],key=lambda k:cache[k]['fitness']);se=max(s['std'] for s in cache[best]['stats'].values())/np.sqrt(10)
    chosen=baseline if cache[baseline]['fitness']>=cache[best]['fitness']-se else best
    # Publish the actual CV-selected mask even if it is not a run winner.
    assert chosen in [k for _,k in specs],('selected_unpublished',chosen)
    selected_label=next(n for n,k in specs if k==chosen)
    candidates=[];head_cache={}
    for name,k in specs:
        for mode in ('variable','fixed'):
            c={'name':mode+'_'+name,'mode':mode,'mask':[x=='1' for x in k],'dimensions':131072}
            hk=(mode,k)
            if hk not in head_cache:
                x=matrix(arrays,ti,c);head_cache[hk]=fit(x,y);del x
            h=head_cache[hk];path=out/(c['name']+'.joblib');joblib.dump(h,path)
            ds=scores(h,matrix(arrays,di,c));c.update(file=path.name,sha256=digest(path),cv=cache[k],dev_auroc=float(roc_auc_score(yd,ds)),dev_pauc=p_auc(yd,ds));candidates.append(c)
            print('HEAD_FROZEN',c['name'],flush=True)
    for pos in (1,2):
        c={'name':'fixed_suffix_minus'+str(3-pos),'mode':'fixed','suffix':pos,'dimensions':131072}
        h=fit(matrix(arrays,ti,c),y);path=out/(c['name']+'.joblib');joblib.dump(h,path);ds=scores(h,matrix(arrays,di,c))
        c.update(file=path.name,sha256=digest(path),dev_auroc=float(roc_auc_score(yd,ds)),dev_pauc=p_auc(yd,ds));candidates.append(c)
    legacy=out/'legacy_B.joblib';legacy.write_bytes((ROOT/'results/fusion/expert_B.joblib').read_bytes())
    candidates.append({'name':'legacy_B','mode':'variable','mask':[False]*32,'legacy':True,'file':legacy.name,'sha256':digest(legacy),'dimensions':131072})
    assert len(candidates)==59
    cv={'folds':[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in f.items()} for f in folds],'evaluations':cache,'runs':runs,'equivalence_checks':checks}
    write(out/'cv.json',cv)
    np.savez_compressed(out/'oof.npz',keys=np.array(list(all_oof)),scores=np.stack(list(all_oof.values())),labels=y,training_ids=np.array([rows[i]['id'] for i in ti]))
    selection=replay_selection(cache,runs)
    write(out/'selection.json',selection)
    deps=['configs/routing/manifest.json','configs/benchmarks/v1.json','configs/positions/manifest.json','configs/artifacts-lock.json','logs/routing-cache-seal.json','logs/positions-cache-seal.json','docs/superpowers/plans/2026-10-29-layer-readout-routing.md','results/routing/cv.json','results/routing/oof.npz','results/routing/selection.json','results/fusion/expert_B.joblib','results/positions/heldout-scores.jsonl']
    frozen={'status':'frozen_before_benchmark_and_fresh_confirmation','sources':snapshot,'dependencies':{n:digest(ROOT/n) for n in deps},'records_sha256':json.loads((ROOT/'configs/routing/manifest.json').read_text())['records_sha256'],'candidates':candidates,'selection':selection,'unique_cv_masks':len(cache),'charged_evaluations':sum(r['unique_evaluations'] for r in runs),'seconds':time.monotonic()-began}
    check_sources(snapshot);write(out/'freeze.json',frozen);print('ROUTING_FROZEN',selection,'SEC',frozen['seconds'],flush=True)

if __name__=='__main__':main()
