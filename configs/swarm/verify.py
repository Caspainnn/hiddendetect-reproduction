"""Independent complete metrics, equal-budget search, and confirmation exclusion checks."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from core import ROOT,digest,verify_freeze
from fusion_core import read_jsonl
from search import initial_population,key

def main():
    out=ROOT/'results/swarm';f=verify_freeze(out);r=json.loads((out/'evaluation.json').read_text())
    assert r['freeze_sha256']==digest(out/'freeze.json') and r['scores_sha256']==digest(out/'heldout-scores.jsonl')
    names={x['name'] for x in f['candidates']};assert len(names)==48
    required={'test_aegis_holdout:global','test_aegis_official:global','test_toxic_holdout:global','test_toxic_holdout:source',
        'test_toxicchat:global','test_toxicchat:source','confirmation_aegis:global'}
    assert set(r['tests'])==set(r['method_comparison'])==required
    runs=json.loads((out/'search.json').read_text());assert len(runs)==20
    assert {(x['method'],x['seed']) for x in runs}=={(m,s) for m in ('random','bpso','ga','de') for s in range(5)}
    for x in runs:
        masks=x['evaluated_masks'];assert len(masks)==len(set(masks))==x['unique_evaluations']==120
        assert x['initial_masks']==masks[:12]==[key(m) for m in initial_population(259,x['seed'],12)]
        assert len(x['best_history'])==120 and np.all(np.diff(x['best_history'])>=0)
        assert x['best_fitness']==max(x['fitness_values'])==x['best_history'][-1]
        winner=next(c for c in f['candidates'] if c['name']==x['method']+'_seed'+str(x['seed']))
        assert key(winner['mask']) in masks and np.isclose(winner['fitness'],x['best_fitness'])
    benchmark=json.loads((ROOT/'configs/benchmarks/v1.json').read_text())['samples']
    confirm=json.loads((ROOT/'configs/swarm/manifest.json').read_text())['samples']
    old=[]
    for family in ('signal-selection','highdim','fusion'):old+=json.loads((ROOT/'configs'/family/'manifest.json').read_text())['samples']
    oldids={x['id'] for x in old};oldgroups={x['group_id'] for x in old}
    for x in confirm:
        if x['role']!='probe_train':assert x['id'] not in oldids and x['group_id'] not in oldgroups
    rows=benchmark+[x for x in confirm if x['role']!='probe_train'];lookup={x['id']:x for x in rows};assert len(lookup)==len(rows)
    scores=read_jsonl(out/'heldout-scores.jsonl');assert len(scores)==len(lookup) and {x['id'] for x in scores}==set(lookup)
    for x in scores:
        assert set(x['scores'])==names
        for k in ('role','label','group_id'):assert x[k]==lookup[x['id']][k]
    for policy,role in [('global','calibration_aegis'),('source','calibration_toxic')]:
        safe=[x for x in scores if x['role']==role and x['label']==0];assert len(safe)==400
        for name in names:assert np.isclose(r['thresholds'][policy][name],np.sort([x['scores'][name] for x in safe])[379],rtol=0,atol=1e-12)
    checked=0
    for test,values in r['tests'].items():
        role,policy=test.split(':');items=[x for x in scores if x['role']==role];y=np.array([x['label'] for x in items]);assert set(values)==names
        for name,v in values.items():
            s=np.array([x['scores'][name] for x in items]);pred=s>r['thresholds'][policy][name]
            actual={'auroc':roc_auc_score(y,s),'partial_auroc':roc_auc_score(y,s,max_fpr=.1),'average_precision':average_precision_score(y,s),
                'tp':int(((y==1)&pred).sum()),'fp':int(((y==0)&pred).sum()),'tn':int(((y==0)&~pred).sum()),'fn':int(((y==1)&~pred).sum()),
                'tpr':float(pred[y==1].mean()),'fpr':float(pred[y==0].mean())}
            for k,value in actual.items():assert np.isclose(v[k],value,rtol=0,atol=1e-12),(test,name,k)
            if policy=='source':assert v['auroc']==r['tests'][role+':global'][name]['auroc']
            checked+=1
        for method,record in r['method_comparison'][test].items():
            for metric,entry in record.items():
                individual=[values[method+'_seed'+str(i)][metric] for i in range(5)]
                assert np.allclose(individual,entry['seeds']) and np.isclose(np.mean(individual),entry['mean'])
                for ref in ('random','bpso'):
                    delta=np.array(individual)-[values[ref+'_seed'+str(i)][metric] for i in range(5)]
                    assert np.isclose(delta.mean(),entry['vs_'+ref]['mean_delta'])
        # Reconstruct B's bootstrap with independent sklearn metric calls.
        if policy=='global':
            rng=np.random.default_rng(20261030);g0=np.flatnonzero(y==0);g1=np.flatnonzero(y==1)
            b0=rng.choice(g0,size=(1000,len(g0)),replace=True);b1=rng.choice(g1,size=(1000,len(g1)),replace=True)
            base=np.array([x['scores']['expert_B'] for x in items]);independent={'auroc':[],'partial_auroc':[],'average_precision':[]}
            for ix in np.column_stack([b0,b1]):
                independent['auroc'].append(roc_auc_score(y[ix],base[ix]));independent['partial_auroc'].append(roc_auc_score(y[ix],base[ix],max_fpr=.1))
                independent['average_precision'].append(average_precision_score(y[ix],base[ix]))
            for metric,distribution in independent.items():assert np.allclose(np.quantile(distribution,[.025,.975]),values['expert_B']['ci95'][metric],rtol=0,atol=1e-12)
    assert checked==336
    result={'status':'passed','scored_rows':len(scores),'candidate_test_conditions':checked,'search_runs':20,'unique_masks_per_run':120,
        'base_bootstrap_replayed_with_sklearn':True,'evaluation_sha256':digest(out/'evaluation.json'),
        'benchmark':'fixed previously observed comparison','confirmation':'fresh disjoint same-source; separate report'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
