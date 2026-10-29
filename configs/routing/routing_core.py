"""Fixed-width layer routing and centered additive-kernel ridge."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'configs/positions'))
from position_core import (digest,read_jsonl,stream_hash,make_folds,p_auc,
    bootstrap_metrics,conservative_threshold,normalize_prompt)
sys.path.insert(0,str(ROOT/'configs/swarm'))
from search import ga_child,de_trial,key

def pad_ids(ids,length,pad):
    assert len(ids)<=length
    return ids+[pad]*(length-len(ids)),[1]*len(ids)+[0]*(length-len(ids))

def readouts(row):
    last=row['input_length']-1;user=row['readout_indices'][6]
    assert 0<=user<last-2
    return [user,last-2,last-1,last]

def route(a,mask):
    mask=np.asarray(mask,dtype=int);assert a.shape[1]==len(mask) and a.shape[2]==2
    return np.asarray(a[:,np.arange(len(mask)),mask,:],dtype=np.float64).reshape(len(a),-1)/np.sqrt(len(mask))

def centered_predict(g,fit,valid,y):
    ff=g[np.ix_(fit,fit)];vf=g[np.ix_(valid,fit)];means=ff.mean(0);grand=means.mean()
    k=ff-means[None,:]-means[:,None]+grand
    kv=vf-vf.mean(1)[:,None]-means[None,:]+grand
    offset=y[fit].mean();return kv@np.linalg.solve(k+np.eye(len(fit)),y[fit]-offset)+offset

def warm_masks(d):
    return np.concatenate([np.zeros((1,d),bool),np.ones((1,d),bool),np.eye(d,dtype=bool)])

def search(method,d,objective_many,seed,budget=64,population=12):
    assert method in ('random','ga','bpso','de','scan') and budget>=d+2
    rng=np.random.default_rng(seed);cache={};proposals=duplicates=restarts=0
    def assess(items):
        nonlocal proposals,duplicates
        pending={}
        for m in items:
            proposals+=1;k=key(m)
            if k in cache or k in pending:duplicates+=1
            elif len(cache)+len(pending)<budget:pending[k]=m.copy()
        vals=objective_many(list(pending.values())) if pending else []
        assert len(vals)==len(pending) and np.isfinite(vals).all()
        cache.update(zip(pending,map(float,vals)))
        return np.array([cache.get(key(m),-np.inf) for m in items])
    warm=warm_masks(d);wv=assess(warm);keep=np.argsort(-wv,kind='stable')[:population]
    masks=warm[keep].copy();values=wv[keep].copy();latent=masks.astype(float)
    personal=masks.copy();pv=values.copy();vel=rng.uniform(-1,1,masks.shape);stagnant=0
    singles=wv[2:];best_single=int(np.argmax(singles));pairs=[]
    for j in rng.permutation([j for j in range(d) if j!=best_single]):
        m=np.zeros(d,bool);m[[j,best_single]]=True;pairs.append(m)
    if method=='scan':assess(pairs[:max(0,budget-len(cache))])
    def randoms():return np.array([rng.random(d)<rng.choice([1/d,.05,.1,.25,.5]) for _ in range(population)])
    while len(cache)<budget:
        before=len(cache)
        if method in ('random','scan'):assess(randoms())
        elif method=='ga':
            def parent():
                ix=rng.choice(population,3,replace=False);return masks[ix[np.argmax(values[ix])]]
            children=np.array([ga_child(parent(),parent(),rng) if rng.random()<.9 else np.logical_xor(parent(),rng.random(d)<1/d) for _ in range(population)])
            cv=assess(children);pool=np.concatenate([masks,children]);v=np.concatenate([values,cv]);ix=np.argsort(-v,kind='stable')[:population];masks=pool[ix];values=v[ix]
        elif method=='de':
            trials=np.array([de_trial(latent,i,rng)[0] for i in range(population)]);children=trials>=.5;cv=assess(children);take=cv>=values
            latent[take]=trials[take];masks[take]=children[take];values[take]=cv[take]
        else:
            best=np.array([x=='1' for x in max(cache,key=cache.get)])
            vel=np.clip(.729*vel+1.494*rng.random(masks.shape)*(personal.astype(float)-masks)+1.494*rng.random(masks.shape)*(best.astype(float)-masks),-6,6)
            masks=rng.random(masks.shape)<1/(1+np.exp(-vel));values=assess(masks);take=values>pv;personal[take]=masks[take];pv[take]=values[take]
        stagnant=stagnant+1 if len(cache)==before else 0
        if stagnant>=3:
            cv=assess(randoms());restarts+=1;stagnant=0
    best=max(cache,key=cache.get)
    return {'method':method,'seed':seed,'best_mask':[x=='1' for x in best],'best_fitness':cache[best],
        'unique_evaluations':len(cache),'evaluated_masks':list(cache),'fitness_values':list(cache.values()),
        'proposal_count':proposals,'duplicate_count':duplicates,'diversity_restarts':restarts}

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')

def sources():
    paths=list((ROOT/'configs/routing').glob('*.py'))+[ROOT/p for p in (
        'configs/positions/position_core.py','configs/swarm/search.py','configs/swarm/core.py',
        'configs/signal-selection/evaluate_frozen.py','configs/signal-selection/experiment_core.py',
        'configs/fusion/fusion_core.py','configs/highdim/ridge.py','configs/highdim/seal.py','configs/highdim/features.py')]
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths}

def check_sources(snapshot):
    for name,sha in snapshot.items():assert digest(ROOT/name)==sha,name

def load_arrays():
    rows=json.loads((ROOT/'configs/routing/manifest.json').read_text())['samples']
    return rows,np.load(ROOT/'cache/routing/fixed.npy',mmap_mode='r'),np.load(ROOT/'cache/routing/fresh_variable.npy',mmap_mode='r'),np.load(ROOT/'cache/positions/P.npy',mmap_mode='r')

def block(arrays,indices,mode):
    rows,fixed,fresh,old=arrays
    if mode=='fixed':return np.asarray(fixed[indices,:,:,:][:,:,[3,0],:],np.float32)
    result=[]
    for i in indices:
        row=rows[i]
        result.append(np.asarray(old[row['previous_index'],:,[7,6],:].transpose(1,0,2) if 'previous_index' in row else fresh[row['fresh_index'],:,[1,0],:].transpose(1,0,2),np.float32))
    return np.stack(result)

def matrix(arrays,indices,c):
    if c.get('suffix') is not None:
        return np.asarray(arrays[1][indices,:,c['suffix'],:],np.float64).reshape(len(indices),-1)/np.sqrt(32)
    return route(block(arrays,indices,c['mode']),c['mask'])

def scores(head,x):return (x-head['mean'])@head['coef']+head['intercept']

def fit(x,y):
    import torch
    with torch.no_grad():
        t=torch.as_tensor(x,dtype=torch.float64,device='cuda');mu=t.mean(0);t=t-mu;offset=float(np.mean(y*2-1))
        g=t@t.T;g.diagonal().add_(1);alpha=torch.linalg.solve(g,torch.as_tensor(y*2-1-offset,dtype=torch.float64,device='cuda'))
        coef=(t.T@alpha).cpu().numpy();mean=mu.cpu().numpy()
    return {'mean':mean,'coef':coef,'intercept':offset,'lambda':1.,'arithmetic':'float64'}

def verify_seal(actual=True):
    seal=json.loads((ROOT/'logs/routing-cache-seal.json').read_text());m=ROOT/'configs/routing/manifest.json'
    assert digest(m)==seal['manifest_sha256']
    for name,sha in seal['files'].items():
        if actual:assert stream_hash(ROOT/name)==sha,name
    return seal

def verify_freeze(actual=True):
    out=ROOT/'results/routing';f=json.loads((out/'freeze.json').read_text());check_sources(f['sources'])
    for name,sha in f['dependencies'].items():assert digest(ROOT/name)==sha,name
    for c in f['candidates']:assert digest(out/c['file'])==c['sha256']
    pre=json.loads((out/'pre-evaluation-seal.json').read_text());assert pre['freeze_sha256']==digest(out/'freeze.json')
    if actual:
        verify_seal();assert digest(ROOT/'data/routing/records.jsonl')==f['records_sha256']
        for name,sha in {'results/routing/freeze.json':pre['freeze_sha256'],**f['sources'],**f['dependencies'],**{'results/routing/'+c['file']:c['sha256'] for c in f['candidates']}}.items():
            assert hashlib.sha256(subprocess.check_output(['git','show',pre['commit']+':'+name],cwd=ROOT)).hexdigest()==sha,name
        subprocess.run(['git','merge-base','--is-ancestor',pre['commit'],'HEAD'],cwd=ROOT,check=True)
    return f
