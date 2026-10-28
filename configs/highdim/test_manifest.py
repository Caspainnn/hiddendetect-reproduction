import json
from collections import Counter
from pathlib import Path
from features import ROOT

manifest=json.loads(Path(__file__).with_name('manifest.json').read_text(encoding='utf-8'))
old=json.loads((ROOT/'configs/signal-selection/manifest.json').read_text(encoding='utf-8'))
rows=manifest['samples']
assert len(rows)==3700 and [x['index'] for x in rows]==list(range(3700))
assert not ({x['id'] for x in rows}&{x['id'] for x in old['samples']})
assert not ({x['group_id'] for x in rows}&{x['group_id'] for x in old['samples']})
assert len({x['group_id'] for x in rows})==3700
assert dict(Counter(x['role']+':'+str(x['label']) for x in rows))==manifest['counts']
for x in rows:
    assert x['input_length']<=512 and len(x['regions'])==4
    user=sum(x['regions'],[])
    assert user and len(set(user))==len(user) and user==sorted(user)
    assert min(user)>=0 and max(user)<x['input_length']
    if x['role'] in ('train','dev','calibration'): assert ':train.json:' in x['id']
    if x['role']=='test_aegis': assert ':test.json:' in x['id']
print('PASS: fresh3700, old-group exclusion, role/count/token-region contracts')
