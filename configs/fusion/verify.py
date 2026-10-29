"""Independent public-artifact metric, fold and threshold verification; no GPU cache needed."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from fusion_core import ROOT,digest,read_jsonl
from provenance import verify_freeze,check_test_conditions

def main():
    out=ROOT/'results/fusion'; freeze=verify_freeze(out)
    manifest=json.loads((ROOT/'configs/fusion/manifest.json').read_text()); rows=manifest['samples']
    old=json.loads((ROOT/'configs/highdim/manifest.json').read_text())['samples']
    old_ids={x['id'] for x in old}; old_groups={x['group_id'] for x in old}
    old_train={x['id'] for x in old if x['role']=='train'}
    assert len({x['id'] for x in rows})==len(rows)==len({x['group_id'] for x in rows})
    for x in rows:
        if x['role']=='probe_train': assert x['id'] in old_train
        else: assert x['id'] not in old_ids and x['group_id'] not in old_groups
    for folder,key,log in [('highdim','old_cache_seal_sha256','highdim-cache-seal.json'),
                            ('fusion','new_cache_seal_sha256','fusion-cache-seal.json')]:
        assert digest(ROOT/'logs'/log)==freeze[key]
    oof=np.load(out/'oof.npz'); assert oof['scores'].shape==(1600,4) and np.isfinite(oof['scores']).all()
    provenance=json.loads((out/'oof-provenance.json').read_text()); assert len(provenance['fit_log'])==20
    for expert in 'BPAT':
        seen=[]
        for fold in [x for x in provenance['fit_log'] if x['expert']==expert]:
            fit=set(fold['train_indices']); valid=set(fold['valid_indices'])
            assert not fit&valid and fit|valid==set(range(1600))
            assert np.all(oof['fold_ids'][list(valid)]==fold['fold'])
            seen+=list(valid)
        assert sorted(seen)==list(range(1600))
    report=json.loads((out/'evaluation.json').read_text())
    check_test_conditions(report['tests'])
    assert report['freeze_sha256']==digest(out/'freeze.json')
    assert report['manifest_sha256']==digest(ROOT/'configs/fusion/manifest.json')
    assert report['scores_sha256']==digest(out/'heldout-scores.jsonl')
    scored=read_jsonl(out/'heldout-scores.jsonl'); expected={x['id']:x for x in rows if x['role']!='probe_train'}
    assert len(scored)==len(expected) and {x['id'] for x in scored}==set(expected)
    names={c['name'] for c in freeze['candidates']}; assert len(names)==23
    for x in scored:
        assert set(x['scores'])==names
        for key in ['role','label','group_id']: assert x[key]==expected[x['id']][key]
    for policy,role in [('global','calibration_aegis'),('source','calibration_toxic')]:
        safe=[x for x in scored if x['role']==role and x['label']==0]; assert len(safe)==400
        for name in names:
            values=np.sort([x['scores'][name] for x in safe])
            threshold=report['thresholds'][policy][name]
            assert np.isclose(threshold,values[379],rtol=0,atol=1e-12)
            assert (values>threshold).sum()<=20
    checks=0
    for test,candidates in report['tests'].items():
        role,policy=test.split(':'); subset=[x for x in scored if x['role']==role]
        y=np.array([x['label'] for x in subset]); assert set(y)=={0,1}
        assert set(candidates)==names
        for name,value in candidates.items():
            score=np.array([x['scores'][name] for x in subset]); pred=score>report['thresholds'][policy][name]
            counts={'tp':int(((y==1)&pred).sum()),'fp':int(((y==0)&pred).sum()),
                    'tn':int(((y==0)&~pred).sum()),'fn':int(((y==1)&~pred).sum())}
            actual={**counts,'auroc':roc_auc_score(y,score),'average_precision':average_precision_score(y,score),
                    'tpr':counts['tp']/(y==1).sum(),'fpr':counts['fp']/(y==0).sum()}
            for key,result in actual.items(): assert np.isclose(value[key],result,rtol=0,atol=1e-12),(test,name,key)
            if policy=='source': assert value['auroc']==report['tests'][role+':global'][name]['auroc']
            checks+=1
    assert checks==138
    result={'status':'passed','scored_rows':len(scored),'candidate_test_conditions':checks,
        'folds_checked':20,'candidates':23,'evaluation_sha256':digest(out/'evaluation.json'),
        'scope':'independent point metrics and thresholds; cache bytes verified separately on compute host; bootstrap not independently regenerated'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))

if __name__=='__main__': main()
