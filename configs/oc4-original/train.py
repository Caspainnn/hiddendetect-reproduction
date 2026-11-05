import time,joblib
from sklearn.metrics import roc_auc_score
from core import *
def main():
    check_lock();assert not (OUT/'training.json').exists()
    data=np.load(CACHE/'training-only.npz');x,y,dx,dy=[data[k] for k in ['train_X','train_y','dev_X','dev_y']]
    started=time.monotonic();search={};selected={};runs=[];fits=0
    for name in SVM+NEURAL:
        grid=svm.GRIDS[name] if name in SVM else attention.LRS;candidates=[]
        for index,param in enumerate(grid):
            entries=[]
            for seed in [SEEDS[0]] if name in SVM else SEEDS:
                path=CACHE/'models'/f'{name}-{index}-{seed}.joblib';meta=path.with_suffix('.json');path.parent.mkdir(exist_ok=True)
                if path.exists():
                    info=load(meta);assert digest(path)==info['sha256'] and info['source_lock_sha256']==digest(OUT/'source-lock.json')
                else:
                    art,cost=svm.fit(name,param,x,y,seed) if name in SVM else attention.fit(name,param,seed,x,y)
                    value=float(roc_auc_score(dy,score(art,dx),max_fpr=.1));joblib.dump(art,path)
                    info={'id':name,'seed':seed,'candidate':index,'param':param,'dev_pauc':value,'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path),'cost':cost,'source_lock_sha256':digest(OUT/'source-lock.json')};write(meta,info)
                fits+=1;entries.append(info)
            value=float(np.mean([m['dev_pauc'] for m in entries]));candidates.append({'param':param,'mean_dev_pauc':value,'runs':entries})
            print('OC_O1_CANDIDATE',name,index,round(value,6),flush=True)
            write(OUT/'training-progress.json',{'id':name,'candidate':index,'fits':fits,'elapsed':time.monotonic()-started})
        best=0
        for k in range(1,len(candidates)):
            if candidates[k]['mean_dev_pauc']>candidates[best]['mean_dev_pauc']+1e-12:best=k
        search[name]=candidates;selected[name]=candidates[best]['param'];runs+=candidates[best]['runs']
        write(OUT/'search-progress.json',{'search':search,'selected':selected})
    assert fits==144 and len(runs)==38
    write(OUT/'training.json',{'run_id':'OC-O1','search':search,'selected':selected,'runs':runs,'total_fits':fits,'selected_models':len(runs),'seconds':time.monotonic()-started,'source_lock_sha256':digest(OUT/'source-lock.json'),'no_cal_or_test_used':True,'models_sealed':True})
    print('OC_O1_MODELS_SEALED',len(runs),flush=True)
if __name__=='__main__':main()
