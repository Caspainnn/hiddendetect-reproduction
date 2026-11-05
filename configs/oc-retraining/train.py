import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,joblib
import numpy as np
import torch
from core import ROOT,H1,OUT,HEADS,SEEDS,digest,write,verify_source,h1_core
PARAMS={HEADS[0]:.1,HEADS[1]:10.,HEADS[2]:.003}

def main():
    verify_source();assert not (OUT/'training.json').exists();torch.set_num_threads(2)
    manifest=json.loads((OUT/'manifest.json').read_text());extraction=json.loads((OUT/'extraction.json').read_text())
    assert digest(ROOT/'cache/oc-retraining/features.npz')==extraction['features_sha256']
    old=np.load(H1/'cache/oc-heads/features.npz');new=np.load(ROOT/'cache/oc-retraining/features.npz');ix=manifest['old_indices']
    datasets={'old_small':(old['train_X'][ix],old['train_y'][ix]),'or_small':(new['train_X'],new['train_y'])};runs=[]
    for group,(x,y) in datasets.items():
        assert len(y)==2*manifest['n_per_class'] and (y==0).sum()==(y==1).sum()
        for name in HEADS:
            for seed in SEEDS:
                artifact,cost=h1_core().fit(name,PARAMS[name],seed,x,y)
                path=ROOT/'cache/oc-retraining/models'/f'{group}-{name}-{seed}.joblib';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();joblib.dump(artifact,path)
                runs.append({'group':group,'id':name,'seed':seed,'param':PARAMS[name],'path':str(path.relative_to(ROOT)),'sha256':digest(path),'train_count':len(y),'epochs':120 if name==HEADS[2] else None,'optimizer_steps':120*int(np.ceil(len(y)/256)) if name==HEADS[2] else None,**cost})
                print('OC_H3_FIT',group,name,seed,flush=True)
    write(OUT/'training.json',{'runs':runs,'features_sha256':extraction['features_sha256'],'source_lock_sha256':digest(OUT/'source-lock.json'),'before_test_scoring':True,'fixed_hyperparameters':PARAMS,'fresh_initialization':True})
    print('OC_H3_MODELS_SEALED',len(runs),flush=True)
if __name__=='__main__':main()
