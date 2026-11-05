import joblib,json,time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from core import *

def main():
    verify_lock();assert not (OUT/'training.json').exists();started=time.monotonic()
    d=np.load(CACHE/'training-only.npz');x=d['X'];y=d['y']
    # Training-only regression: exact old architecture, optimizer and batches.
    original=module('a2_h1_reference',ROOT/'configs/oc-heads/core.py')
    original_artifact,original_cost=original.fit(ATT,.003,SEEDS[0],x,y)
    history=json.loads((H3/'results/oc-retraining/training.json').read_text())
    old=next(r for r in history['runs'] if r['id']==ATT and r['group']=='or_small' and r['seed']==SEEDS[0])
    old_artifact=joblib.load(H3/old['path']);assert digest(H3/old['path'])==old['sha256']
    for k,value in old_artifact['state'].items():torch.testing.assert_close(original_artifact['state'][k],value,rtol=0,atol=0)
    regression_path=CACHE/'historical-replay.joblib';regression_path.parent.mkdir(exist_ok=True);joblib.dump(original_artifact,regression_path)
    cv={};selected={};runs=[]
    splits=list(StratifiedKFold(5,shuffle=True,random_state=SEEDS[0]).split(x,y))
    for name in NAMES:
        candidates=[]
        for lr in LRS:
            folds=[]
            for fold,(tr,va) in enumerate(splits):
                art,cost=fit(name,lr,SEEDS[0],x[tr],y[tr]);value=float(roc_auc_score(y[va],score(art,x[va]),max_fpr=.1))
                folds.append({'fold':fold,'pauc':value,**cost})
            mean=float(np.mean([f['pauc'] for f in folds]));candidates.append({'lr':lr,'folds':folds,'mean_pauc':mean})
            print('OC_A2_CV',name,lr,round(mean,6),flush=True)
            write(OUT/'training-progress.json',{'name':name,'lr':lr,'cv':cv,'current':candidates,'elapsed_seconds':time.monotonic()-started})
        best=0
        for j in range(1,len(candidates)):
            if candidates[j]['mean_pauc']>candidates[best]['mean_pauc']+1e-12:best=j
        cv[name]=candidates;selected[name]=candidates[best]['lr']
    for name in NAMES:
        for seed in SEEDS:
            art,cost=fit(name,selected[name],seed,x,y);path=CACHE/'models'/f'{name}-{seed}.joblib';path.parent.mkdir(exist_ok=True)
            assert not path.exists();joblib.dump(art,path);runs.append({'id':name,'seed':seed,'lr':selected[name],'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),**cost})
        print('OC_A2_FIT',name,flush=True)
    write(OUT/'training.json',{'run_id':'OC-A2','cv':cv,'selected':selected,'runs':runs,'total_fits':165+33+1,
        'folds':5,'split_seed':SEEDS[0],'steps':STEPS,'full_batch':True,'seeds':SEEDS,'seconds':time.monotonic()-started,
        'historical_replay':{'path':regression_path.relative_to(ROOT).as_posix(),'sha256':digest(regression_path),'training_state_exact_H3':True,**original_cost},
        'source_lock_sha256':digest(OUT/'source-lock.json'),'models_sealed_before_target_scoring':True})
    print('OC_A2_MODELS_SEALED',len(runs),flush=True)
if __name__=='__main__':main()
