"""Explicit numpy population variants, equal unique-mask budgets, no test inputs."""
import numpy as np

def key(mask):return ''.join('1' if x else '0' for x in mask)

def random_mask(d,rng):return rng.random(d)<rng.choice([.05,.1,.25,.5])

def initial_population(d,seed,n=12):
    if d<1 or n>2**min(d,30):raise ValueError('Insufficient distinct masks')
    rng=np.random.default_rng(seed); masks=[np.zeros(d,bool),np.ones(d,bool)]; seen={key(m) for m in masks}
    while len(masks)<n:
        m=random_mask(d,rng)
        if key(m) not in seen: masks.append(m);seen.add(key(m))
    return np.array(masks[:n])

def ga_child(a,b,rng,mutation=None):
    child=np.where(rng.random(len(a))<.5,a,b).copy()
    return np.logical_xor(child,rng.random(len(a))<(1/len(a) if mutation is None else mutation))

def de_trial(pop,i,rng,f=.5,cr=.9):
    donors=rng.choice(np.delete(np.arange(len(pop)),i),3,replace=False)
    a,b,c=pop[donors];mutant=np.clip(a+f*(b-c),0,1)
    take=rng.random(pop.shape[1])<cr;take[rng.integers(pop.shape[1])]=True
    return np.where(take,mutant,pop[i]),donors.tolist()

def search(method,d,objective_many,seed,budget=120,population=12):
    if method not in ('random','bpso','ga','de') or budget<population or budget>2**min(d,30):raise ValueError('Invalid search')
    rng=np.random.default_rng(seed); masks=initial_population(d,seed,population); cache={};history=[]
    proposals=0;duplicates=0;restarts=0
    def assess(items):
        nonlocal proposals,duplicates
        pending={}
        for m in items:
            proposals+=1;k=key(m)
            if k in cache or k in pending:duplicates+=1
            elif len(cache)+len(pending)<budget:pending[k]=m.copy()
        values=objective_many(list(pending.values())) if pending else []
        assert len(values)==len(pending) and np.isfinite(values).all()
        for k,v in zip(pending,values):cache[k]=float(v);history.append(max(cache.values()))
        return np.array([cache.get(key(m),-np.inf) for m in items])
    values=assess(masks); personal=masks.copy();pvalues=values.copy();vel=rng.uniform(-1,1,masks.shape)
    latent=masks.astype(float);stagnant=0
    while len(cache)<budget:
        before=len(cache)
        if method=='random':
            children=np.array([random_mask(d,rng) for _ in range(population)]);assess(children)
        elif method=='ga':
            def parent():
                ix=rng.choice(population,3,replace=False);return masks[ix[np.argmax(values[ix])]]
            children=np.array([ga_child(parent(),parent(),rng) if rng.random()<.9 else np.logical_xor(parent(),rng.random(d)<1/d) for _ in range(population)])
            cv=assess(children); pool=np.concatenate([masks,children]);pv=np.concatenate([values,cv])
            keep=np.argsort(-pv,kind='stable')[:population];masks=pool[keep];values=pv[keep]
        elif method=='de':
            trials=np.array([de_trial(latent,i,rng)[0] for i in range(population)])
            children=trials>=.5;cv=assess(children);take=cv>=values
            latent[take]=trials[take];masks[take]=children[take];values[take]=cv[take]
        else:
            best=np.array([x=='1' for x in max(cache,key=cache.get)])
            vel=np.clip(.729*vel+1.494*rng.random(masks.shape)*(personal.astype(float)-masks)+1.494*rng.random(masks.shape)*(best.astype(float)-masks),-6,6)
            masks=rng.random(masks.shape)<1/(1+np.exp(-vel));values=assess(masks)
            take=values>pvalues;personal[take]=masks[take];pvalues[take]=values[take]
        stagnant=stagnant+1 if len(cache)==before else 0
        if stagnant>=3 and len(cache)<budget:
            children=np.array([random_mask(d,rng) for _ in range(population)]);cv=assess(children);restarts+=1;stagnant=0
            if method in ('ga','de'):
                take=cv>values;masks[take]=children[take];values[take]=cv[take];latent[take]=children[take].astype(float)
    best=max(cache,key=cache.get)
    return {'best_mask':[x=='1' for x in best],'best_fitness':cache[best],'unique_evaluations':len(cache),
        'evaluated_masks':list(cache),'fitness_values':list(cache.values()),'best_history':history,
        'proposal_count':proposals,'duplicate_count':duplicates,'diversity_restarts':restarts,
        'initial_masks':[key(m) for m in initial_population(d,seed,population)]}
