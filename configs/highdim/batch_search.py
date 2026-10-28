"""PSO's proposal sequence is unchanged; each round's independent fits may batch."""
import numpy as np

def batched_binary_pso(dimensions,objective_many,seed,budget=100,particles=10,rounds=10):
    rng=np.random.default_rng(seed); positions=rng.random((particles,dimensions))<.5; positions[0]=True
    velocities=rng.uniform(-1,1,(particles,dimensions)); personal=positions.copy(); personal_values=np.full(particles,-np.inf)
    best=positions[0].copy(); best_value=-np.inf; cache={}
    def result(): return {'best_mask':best.tolist(),'best_fitness':best_value,'unique_evaluations':len(cache),'evaluated_masks':list(cache)}
    for _ in range(rounds):
        pending={}; stop=None
        for i in range(particles):
            if not positions[i].any(): positions[i,rng.integers(dimensions)]=True
            key=''.join('1' if x else '0' for x in positions[i])
            if key not in cache and key not in pending:
                if len(cache)+len(pending)>=budget: stop=i; break
                pending[key]=positions[i].copy()
        values=objective_many(list(pending.values())) if pending else []
        assert len(values)==len(pending)
        cache.update((key,float(value)) for key,value in zip(pending,values))
        for i in range(particles if stop is None else stop):
            key=''.join('1' if x else '0' for x in positions[i]); value=cache[key]
            if value>personal_values[i]: personal_values[i]=value; personal[i]=positions[i].copy()
            if value>best_value: best_value=value; best=positions[i].copy()
        if stop is not None: return result()
        velocities=.729*velocities+1.494*rng.random(positions.shape)*(personal.astype(float)-positions)+1.494*rng.random(positions.shape)*(best.astype(float)-positions)
        velocities=np.clip(velocities,-6,6); positions=rng.random(positions.shape)<1/(1+np.exp(-velocities))
    return result()
