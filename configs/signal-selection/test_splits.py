from build_manifest import remove_later_duplicates

rows = [{'id':'old', 'prompt':'How do I stop a Python process?', 'role':'excluded'},
        {'id':'a', 'prompt':'How do I stop a python process!', 'role':'train_pool'},
        {'id':'b', 'prompt':'Tell me about forest ecosystems.', 'role':'train_pool'},
        {'id':'c', 'prompt':'Tell me about forest ecosystems!', 'role':'test'},
        {'id':'d', 'prompt':'Describe the moons of Jupiter.', 'role':'test'}]
kept, removed = remove_later_duplicates(rows, similarity=0.92)
assert [x['id'] for x in kept] == ['old', 'b', 'd']
assert {x['id'] for x in removed} == {'a','c'}
assert all(x['duplicate_of'] in {'old','b'} for x in removed)
long = 'A detailed description of forests rivers mountains and ecosystems ' * 8
near_rows = [{'id': 'a', 'prompt': long + 'blue', 'role': 'train'},
             {'id': 'b', 'prompt': long + 'green', 'role': 'test'},
             {'id': 'c', 'prompt': 'Unrelated short input', 'role': 'test'}]
kept, removed = remove_later_duplicates(near_rows)
assert [x['id'] for x in kept] == ['a', 'c'] and removed[0]['duplicate_of'] == 'a'

import json
from collections import Counter
from pathlib import Path
manifest = json.loads(Path(__file__).with_name('manifest.json').read_text(encoding='utf-8'))
samples = manifest['samples']
assert len(samples) == 4800
assert [x['index'] for x in samples] == list(range(4800))
for key in ('id', 'group_id', 'prompt_sha256'):
    assert len({x[key] for x in samples}) == 4800, key
assert dict(Counter(x['role'] + ':' + str(x['label']) for x in samples)) == manifest['counts']
assert all(0 < x['input_length'] <= 512 and not x['id'].startswith('old-') for x in samples)
assert all(':train.json:' in x['id'] for x in samples if x['role'] in ('train', 'dev'))
assert all(':validation.json:' in x['id'] for x in samples if x['role'] == 'calibration')
assert all(':test.json:' in x['id'] for x in samples if x['role'] == 'test_aegis')
print('PASS: exact/near duplicate priority, immutable 4800 indices and disjoint prompt groups/official splits')
