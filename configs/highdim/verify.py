"""Verify published hashes, group budgets and every heldout point metric."""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from features import ROOT,digest,read_jsonl
sys.path.insert(0,str(ROOT/'configs/signal-selection'))
from evaluate_frozen import metrics

def main():
    out=ROOT/'results/highdim'; mp=Path(__file__).with_name('manifest.json')
    manifest=json.loads(mp.read_text()); freeze=json.loads((out/'freeze.json').read_text()); evaluation=json.loads((out/'evaluation.json').read_text())
    assert manifest['raw_lock_sha256']==digest(ROOT/'logs/signal-selection-download.json')
    assert manifest['old_manifest_sha256']==digest(ROOT/'configs/signal-selection/manifest.json')
    assert freeze['manifest_sha256']==evaluation['manifest_sha256']==digest(mp)
    assert evaluation['freeze_sha256']==digest(out/'freeze.json')
    assert evaluation['scores_sha256']==digest(out/'heldout-scores.jsonl')
    for name,key in [('schema.json','schema_sha256'),('centroids.npy','centroids_sha256'),('scaler.npz','scaler_sha256'),('search.json','search_sha256')]: assert digest(out/name)==freeze[key]
    groups=json.loads((out/'schema.json').read_text()); assert groups[0]['start']==0
    assert all(left['stop']==right['start'] for left,right in zip(groups,groups[1:]))
    for c in freeze['candidates']:
        if 'file' in c: assert digest(out/c['file'])==c['sha256'],c['name']
        if c['kind']=='groups':
            assert len(c['mask'])==len(groups) and any(c['mask'])
            if c['name']!='L1':
                assert sum(c['mask'])==c['groups']
                assert sum(g['stop']-g['start'] for take,g in zip(c['mask'],groups) if take)==c['dimensions']
    runs=json.loads((out/'search.json').read_text()); counts=Counter()
    for run in runs:
        keys=run['evaluated_masks']; assert len(keys)==len(set(keys))==run['unique_evaluations']<=100
        assert all(len(key)==len(groups) and '1' in key and set(key)<={'0','1'} for key in keys)
        counts[run['head']+':'+run['method']]+=1
    assert counts=={'linear:random':5,'linear:bpso':5,'interaction:random':5,'interaction:bpso':5}
    records=read_jsonl(out/'heldout-scores.jsonl'); assert len(records)==len({x['id'] for x in records})==1500
    lookup={x['id']:x for x in manifest['samples']}
    for row in records: assert all(row[key]==lookup[row['id']][key] for key in ('group_id','role','label'))
    for role in ('calibration','test_aegis','test_toxicchat'):
        subset=[x for x in records if x['role']==role]; y=np.array([x['label'] for x in subset])
        for c in freeze['candidates']:
            name=c['name']; values=np.array([x['scores'][name] for x in subset]); actual=metrics(y,values,evaluation['thresholds'][name])
            expected=evaluation['calibration'][name] if role=='calibration' else evaluation['tests'][role][name]
            assert all(np.isclose(actual[key],expected[key],rtol=0,atol=1e-12) for key in actual),(role,name)
            if role=='calibration': assert actual['fpr']<=.05
    report={'status':'pass','manifest_rows':3700,'heldout_rows':1500,'candidates':len(freeze['candidates']),
        'search_counts':dict(counts),'all_point_metrics_recomputed':True,'all_artifact_hashes_verified':True,
        'freeze_sha256':evaluation['freeze_sha256']}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))

if __name__=='__main__': main()
