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
print('PASS: normalized/near duplicate priority excludes old450 and protects official split boundaries')
