"""Offline point metrics, role/fold isolation and bootstrap replay verification."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from position_core import ROOT,digest,verify_freeze,read_jsonl,upper_bounds

def main():
    out=ROOT/'results/positions';f=verify_freeze(out);r=json.loads((out/'evaluation.json').read_text())
    assert r['freeze_sha256']==digest(out/'freeze.json') and r['scores_sha256']==digest(out/'heldout-scores.jsonl')
    names={c['name'] for c in f['candidates']};assert len(names)==17
    expected={'test_aegis_holdout:global','test_aegis_official:global','test_toxic_holdout:global','test_toxic_holdout:source',
        'test_toxicchat:global','test_toxicchat:source','historical_confirmation:global','confirmation_aegis:global'}
    assert set(r['tests'])==set(r['restricted_optimality'])==expected
    m=json.loads((ROOT/'configs/positions/manifest.json').read_text());rows=m['samples'];train=[x for x in rows if x['role']=='train']
    assert len(train)==1600 and len({x['group_id'] for x in rows})==len(rows)
    for family in ('signal-selection','highdim','fusion','swarm'):
        old=json.loads((ROOT/'configs'/family/'manifest.json').read_text())['samples'];ids={x['id'] for x in old};groups={x['group_id'] for x in old}
        for x in rows:
            if x['role']=='confirmation_aegis':assert x['id'] not in ids and x['group_id'] not in groups
    oof=np.load(out/'oof.npz');assert oof['scores'].shape==(17,2,1600) and np.isfinite(oof['scores']).all()
    assert np.array_equal(oof['labels'],[x['label'] for x in train]) and np.array_equal(oof['training_ids'],[x['id'] for x in train])
    cv=json.loads((out/'cv.json').read_text());assert len(cv['folds'])==10 and len(cv['fits'])==170
    for repeat in range(2):
        seen=[]
        for fold in [v for v in cv['folds'] if v['repeat']==repeat]:
            assert not set(fold['fit'])&set(fold['valid']);assert set(fold['fit'])|set(fold['valid'])==set(range(1600));seen+=fold['valid']
        assert sorted(seen)==list(range(1600))
    for entry in cv['fits']:
        fold=next(v for v in cv['folds'] if v['repeat']==entry['repeat'] and v['fold']==entry['fold']);k=[c['name'] for c in f['candidates']].index(entry['candidate'])
        ix=fold['valid'];s=oof['scores'][k,entry['repeat'],ix];y=oof['labels'][ix]
        assert np.isclose(roc_auc_score(y,s),entry['auroc'],atol=1e-12) and np.isclose(roc_auc_score(y,s,max_fpr=.1),entry['partial_auroc'],atol=1e-12)
    reconstructed=[]
    for candidate in f['candidates']:
        values=[entry['partial_auroc'] for entry in cv['fits'] if entry['candidate']==candidate['name']]
        assert len(values)==10
        mean=float(np.mean(values));std=float(np.std(values,ddof=1))
        assert np.isclose(mean,candidate['cv_pauc_mean'],rtol=0,atol=1e-12) and np.isclose(std,candidate['cv_pauc_std'],rtol=0,atol=1e-12)
        reconstructed.append({**candidate,'mean':mean,'std':std})
    best=max([c for c in reconstructed if c['eligible']],key=lambda c:(c['mean'],-len(c['positions'])))
    base=next(c for c in reconstructed if c['name']=='template_last');cut=best['mean']-best['std']/np.sqrt(10)
    selected=base if base['mean']>=cut else best
    assert f['selection']['selected']==selected['name'] and f['selection']['raw_best_cv']==best['name']
    assert np.isclose(f['selection']['one_se_threshold'],cut,rtol=0,atol=1e-12)
    for label,width in [('best_nonlast_single',1),('best_real_pair',2)]:
        winner=max([c for c in reconstructed if c['real_alternative'] and len(c['positions'])==width],key=lambda c:c['mean'])
        assert f['selection'][label]==winner['name']
    samples=read_jsonl(out/'heldout-scores.jsonl');lookup={x['id']:x for x in rows if x['role'] not in ('train','dev')}
    assert len(samples)==len(lookup) and {x['id'] for x in samples}==set(lookup)
    for x in samples:
        assert set(x['scores'])==names
        for k in ('group_id','role','label'):assert x[k]==lookup[x['id']][k]
    for policy,role in [('global','calibration_aegis'),('source','calibration_toxic')]:
        safe=[x for x in samples if x['role']==role and x['label']==0];assert len(safe)==400
        for name in names:assert np.isclose(r['thresholds'][policy][name],np.sort([x['scores'][name] for x in safe])[379],rtol=0,atol=1e-12)
    checked=0
    for condition,values in r['tests'].items():
        role,policy=condition.split(':');items=[x for x in samples if x['role']==role];y=np.array([x['label'] for x in items]);assert set(values)==names
        for name,v in values.items():
            s=np.array([x['scores'][name] for x in items]);p=s>r['thresholds'][policy][name]
            actual={'auroc':roc_auc_score(y,s),'partial_auroc':roc_auc_score(y,s,max_fpr=.1),'average_precision':average_precision_score(y,s),
                'tp':int(((y==1)&p).sum()),'tn':int(((y==0)&~p).sum()),'fp':int(((y==0)&p).sum()),'fn':int(((y==1)&~p).sum()),
                'tpr':p[y==1].mean(),'fpr':p[y==0].mean()}
            for k,a in actual.items():assert np.isclose(v[k],a,rtol=0,atol=1e-12),(condition,name,k)
            checked+=1
        assert r['restricted_optimality'][condition]['family']==12
        assert set(r['restricted_optimality'][condition]['upper_delta_auroc'])=={c['name'] for c in f['candidates'] if c['real_alternative']}
        assert r['restricted_optimality'][condition]['supports_margin']==all(v<.005 for v in r['restricted_optimality'][condition]['upper_delta_auroc'].values())
    assert checked==136
    # Independent sklearn replay of primary confirmation ROC and simultaneous bounds.
    items=[x for x in samples if x['role']=='confirmation_aegis'];y=np.array([x['label'] for x in items]);rng=np.random.default_rng(20261119)
    g0=np.flatnonzero(y==0);g1=np.flatnonzero(y==1);b0=rng.choice(g0,size=(5000,len(g0)),replace=True);b1=rng.choice(g1,size=(5000,len(g1)),replace=True)
    ix=np.column_stack([b0,b1]);base=np.array([x['scores']['template_last'] for x in items]);aucs=np.array([roc_auc_score(y[b],base[b]) for b in ix])
    assert np.allclose(np.quantile(aucs,[.025,.975]),r['tests']['confirmation_aegis:global']['template_last']['ci95']['auroc'],rtol=0,atol=1e-12)
    checked_bounds=0
    for name,upper in r['restricted_optimality']['confirmation_aegis:global']['upper_delta_auroc'].items():
        s=np.array([x['scores'][name] for x in items]);delta=np.array([roc_auc_score(y[b],s[b]) for b in ix])-aucs
        assert np.isclose(np.quantile(delta,1-.05/12),upper,rtol=0,atol=1e-12)
        checked_bounds+=1;print('BOUND_REPLAYED',name,flush=True)
    assert checked_bounds==12
    result={'status':'passed','scored_rows':len(samples),'candidate_test_conditions':136,'cv_fits_checked':170,
        'independent_confirmation_base_bootstrap_replayed':True,'confirmation_alternative_bounds_replayed':checked_bounds,
        'cv_selection_rule_replayed':True,'evaluation_sha256':digest(out/'evaluation.json')}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
