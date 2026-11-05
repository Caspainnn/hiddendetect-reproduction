import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='2'
import json,joblib,time,subprocess
import numpy as np
import torch
from core import ROOT,OUT,IDS,GRIDS,SEEDS,digest,write,fit,score,pauc

def main():
    torch.set_num_threads(2)
    seal=json.loads((OUT/'source-lock.json').read_text())
    for path,sha in seal['sha256'].items():assert digest(ROOT/path)==sha,path
    assert json.loads((OUT/'identity-training.json').read_text())['status']=='passed'
    assert not (OUT/'selection.json').exists(),'Refuse to overwrite an already selected run'
    data=np.load(ROOT/'cache/oc-heads/features.npz');x=data['train_X'];y=data['train_y'];dev=data['dev_X'];dy=data['dev_y']
    assert x.shape==(1600,14) and dev.shape==(600,14)
    assert digest(ROOT/'cache/oc-heads/features.npz')==json.loads((OUT/'manifest.json').read_text())['features_sha256']
    cache=ROOT/'cache/oc-heads/models';cache.mkdir(exist_ok=True)
    search=[];selected={};began=time.monotonic()
    for name in IDS:
        candidates=[]
        for index,param in enumerate(GRIDS[name]):
            runs=[]
            for seed in SEEDS:
                path=cache/f'{name}-{index}-{seed}.joblib';meta=path.with_suffix('.json')
                if path.exists():
                    info=json.loads(meta.read_text());assert info['source_lock_sha256']==digest(OUT/'source-lock.json')
                else:
                    artifact,cost=fit(name,param,seed,x,y);s=score(artifact,dev)
                    assert s.shape==(600,) and np.isfinite(s).all()
                    joblib.dump(artifact,path)
                    info={'id':name,'candidate_index':index,'parameter':param,'seed':seed,'dev_pauc':pauc(dy,s),'dev_scores':s.tolist(),'sha256':digest(path),'cost':cost,'source_lock_sha256':digest(OUT/'source-lock.json')}
                    write(meta,info)
                assert digest(path)==info['sha256'];runs.append(info)
                print('HEAD_FIT',name,index,seed,'DEV_PAUC',round(info['dev_pauc'],6),flush=True)
            candidates.append({'parameter':param,'index':index,'mean_dev_pauc':float(np.mean([r['dev_pauc'] for r in runs])),'runs':runs})
        best=max(candidates,key=lambda c:c['mean_dev_pauc'])
        selected[name]={'candidate_index':best['index'],'parameter':best['parameter'],'mean_dev_pauc':best['mean_dev_pauc'],
                        'models':[{'seed':r['seed'],'path':str((cache/f"{name}-{best['index']}-{r['seed']}.joblib").relative_to(ROOT)),'sha256':r['sha256'],'cost':r['cost']} for r in best['runs']]}
        search.append({'id':name,'candidates':candidates});write(OUT/'search-progress.json',search)
        print('HEAD_SELECTED',name,best['index'],flush=True)
    write(OUT/'search.json',search)
    write(OUT/'selection.json',{'run_id':'OC-H1','selected':selected,'criterion':'mean of three fixed-seed dev pAUC; first grid entry wins exact ties','primary_seed':SEEDS[0],'seeds':SEEDS,'search_sha256':digest(OUT/'search.json'),'source_lock_sha256':digest(OUT/'source-lock.json'),'seconds':time.monotonic()-began,'fit_count':72,'test_or_calibration_used':False})
    print('HEAD_TRAINING_COMPLETE_SELECTION_FROZEN',flush=True)
if __name__=='__main__':main()
