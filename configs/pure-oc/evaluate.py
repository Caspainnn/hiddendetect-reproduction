import json
import numpy as np
from analysis import ROOT,aggregate,digest,evaluate,read,write

out=ROOT/'results/pure-oc';seal=json.loads((ROOT/'configs/pure-oc/run-lock.json').read_text())
for name,sha in seal['sha256'].items():assert digest(ROOT/name)==sha,name
rows=read(out/'confirmation-scores.jsonl');manifest=json.loads((ROOT/'configs/pure-oc/manifest.json').read_text())
ex=json.loads((out/'extraction.json').read_text());assert ex['status']=='completed' and ex['rows']==len(rows)==2000
assert ex['scores_sha256']==digest(out/'confirmation-scores.jsonl')
assert ex['manifest_sha256']==digest(ROOT/'configs/pure-oc/manifest.json')
assert ex['run_lock_sha256']==digest(ROOT/'configs/pure-oc/run-lock.json')
assert [r['id'] for r in rows]==[r['id'] for r in manifest['samples']]
for r,m in zip(rows,manifest['samples']):
    assert all(r[k]==v for k,v in m.items())
    assert r['run_lock_sha256']==digest(ROOT/'configs/pure-oc/run-lock.json')
assert np.array_equal(aggregate([r['F'] for r in rows]),[r['scores'] for r in rows])
q=json.loads((out/'thresholds.json').read_text());assert not set(q['calibration_ids'])&set(r['id'] for r in rows)
result=evaluate(rows,q['thresholds'],'OC-A1-N');result.pop('scores')
result.update(manifest_sha256=digest(ROOT/'configs/pure-oc/manifest.json'),scores_sha256=digest(out/'confirmation-scores.jsonl'),
    run_lock_sha256=digest(ROOT/'configs/pure-oc/run-lock.json'),scope='new same-source Aegis holdout; now historical, not cross-domain')
write(out/'confirmation.json',result)
print('OC_CONFIRMATION_EVALUATED',[(m['name'],m['auroc'],m['pauc'],m.get('decision')) for m in result['metrics']])
