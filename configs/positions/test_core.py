import numpy as np
from position_core import token_indices,models,make_folds,pack,upper_bounds

row={'id':'a','input_length':15,'regions':[[3,4],[5,6],[7,8],[9,10]]}
a=token_indices(row);assert a[:8]==[2,3,4,6,8,9,10,14]
assert a==token_indices(row) and 3<=a[-1]<=10
short={'id':'b','input_length':5,'regions':[[2],[],[],[]]}
assert token_indices(short)[1:7]==[2]*6
assert len(models())==17
x=np.random.default_rng(3).normal(size=(6,32,9,7)).astype('float32')
single=pack(x,[7]);duplicate=pack(x,[7,7]);assert np.allclose(single@single.T,duplicate@duplicate.T,atol=1e-6)
y=np.array([0]*50+[1]*50);groups=np.array(['g'+str(i) for i in range(100)]);folds=make_folds(y,groups)
assert len(folds)==10
for repeat in range(2):
    seen=[]
    for f in [x for x in folds if x['repeat']==repeat]:
        assert not set(groups[f['fit']])&set(groups[f['valid']]);seen+=f['valid'].tolist()
    assert sorted(seen)==list(range(100))
assert not upper_bounds({'x':np.ones(5000)*.006},family=12)['supports_margin']
assert upper_bounds({'x':np.zeros(5000)},family=12)['supports_margin']
print('PASS: actual token alignment, duplicates, group folds and restricted-optimality upper bounds')
