import json,time,joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from core import ROOT,OUT,CACHE,SEEDS,BASE,COST,GRIDS,digest,write,verify_lock,fit,score

def main():
    verify_lock();assert not (OUT/'training.json').exists()
    data=np.load(CACHE/'training-only.npz');x=data['X'];y=data['y']
    splits=list(StratifiedKFold(5,shuffle=True,random_state=SEEDS[0]).split(x,y))
    started=time.monotonic();cv={};selected={};runs=[]
    for name,grid in GRIDS.items():
        candidates=[]
        for j,param in enumerate(grid):
            folds=[]
            for fold,(tr,va) in enumerate(splits):
                artifact,cost=fit(name,param,x[tr],y[tr])
                value=float(roc_auc_score(y[va],score(artifact,x[va]),max_fpr=.1))
                folds.append({'fold':fold,'pauc':value,**cost})
            mean=float(np.mean([f['pauc'] for f in folds]))
            candidates.append({'param':param,'folds':folds,'mean_pauc':mean})
            print('OC_S1_CV',name,j,round(mean,6),flush=True)
            write(OUT/'training-progress.json',{'name':name,'candidate':j,'completed_cv':cv,'current_cv':candidates,'elapsed_seconds':time.monotonic()-started})
        best=0
        for j in range(1,len(candidates)):
            if candidates[j]['mean_pauc']>candidates[best]['mean_pauc']+1e-12:best=j
        cv[name]=candidates;selected[name]=candidates[best]['param']
    tasks=[(name,selected[name],seed) for name in GRIDS for seed in SEEDS]
    tasks.extend([('H3-C10',{'C':10.},SEEDS[0]),('cost-fixed-C-unweighted',{'C':selected[COST]['C']},SEEDS[0])])
    for name,param,seed in tasks:
        artifact,cost=fit(BASE if name not in GRIDS else name,param,x,y,seed)
        path=CACHE/'models'/f'{name}-{seed}.joblib';path.parent.mkdir(exist_ok=True)
        assert not path.exists();joblib.dump(artifact,path)
        runs.append({'id':name,'param':param,'seed':seed,'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),**cost})
    write(OUT/'training.json',{'run_id':'OC-S1','cv':cv,'selected':selected,'folds':5,'split_seed':SEEDS[0],
        'runs':runs,'total_fits':225+len(tasks),'seconds':time.monotonic()-started,'source_lock_sha256':digest(OUT/'source-lock.json'),
        'models_sealed_before_target_scoring':True,'deterministic_seed_repeats_not_independent_evidence':True})
    print('OC_S1_MODELS_SEALED',len(runs),flush=True)
if __name__=='__main__':main()
