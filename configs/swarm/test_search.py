import numpy as np
from search import search,initial_population,de_trial,ga_child

for method in ('random','bpso','ga','de'):
    seen=[]
    def objective_many(masks):
        seen.extend(''.join('1' if b else '0' for b in m) for m in masks)
        return [float(m[:4].sum()-.1*m[4:].sum()) for m in masks]
    a=search(method,20,objective_many,seed=3,budget=40,population=8)
    assert a['unique_evaluations']==40 and len(seen)==len(set(seen))==40
    assert a['evaluated_masks'][:8]==[''.join('1' if b else '0' for b in m) for m in initial_population(20,3,8)]
    assert np.all(np.diff(a['best_history'])>=0)
    b=search(method,20,lambda masks:[float(m[:4].sum()-.1*m[4:].sum()) for m in masks],seed=3,budget=40,population=8)
    assert a['evaluated_masks']==b['evaluated_masks'] and a['best_mask']==b['best_mask']
rng=np.random.default_rng(7)
child=ga_child(np.zeros(20,bool),np.ones(20,bool),rng,mutation=0)
assert child.dtype==bool and child.any() and not child.all()
pop=rng.random((8,20)); trial,donors=de_trial(pop,2,rng,cr=0)
assert len(set(donors))==3 and 2 not in donors
assert np.all((trial>=0)&(trial<=1)) and np.any(trial!=pop[2])
print('PASS: common initialization, exact unique budgets, reproducibility, monotone best, GA/DE invariants')
from core import deduplicate,p_auc
from build_manifest import remove_later_duplicates
rows=[{'id':str(i),'role':'excluded' if i==0 else 'new','prompt':t} for i,t in enumerate(['a fairly long normal question about growing garden flowers '*8,'a fairly long normal question about growing garden flowers '*8,'completely unrelated question on the origin of black holes','a fairly long normal question about growing garden flowers '*8+' today'])]
assert deduplicate(rows)==remove_later_duplicates(rows)
assert p_auc([0,0,1,1],[0,1,2,3])==1
assert p_auc([0,0,1,1],[2,3,0,1])<.5
print('PASS: vectorized dedup equals legacy grouping and low-FPR pAUC direction')
from core import bootstrap_metrics
from sklearn.metrics import roc_auc_score,average_precision_score
y=np.array([0]*20+[1]*20);s=np.round(np.random.default_rng(4).normal(size=40),1)
rng=np.random.default_rng(5);b0=rng.choice(np.arange(20),size=(20,20));b1=rng.choice(np.arange(20,40),size=(20,20))
result=bootstrap_metrics(y,s,b0,b1)
for i,ix in enumerate(np.column_stack([b0,b1])):
    assert np.isclose(result['auroc'][i],roc_auc_score(y[ix],s[ix]),atol=1e-12)
    assert np.isclose(result['average_precision'][i],average_precision_score(y[ix],s[ix]),atol=1e-12)
    assert np.isclose(result['partial_auroc'][i],roc_auc_score(y[ix],s[ix],max_fpr=.1),atol=1e-12)
print('PASS: vectorized paired bootstrap matches sklearn, including score ties')
