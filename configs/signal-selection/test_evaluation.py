import numpy as np
from evaluate_frozen import metrics, wilson

y = np.array([0, 0, 1, 1])
observed = metrics(y, np.array([0., 1., 2., 3.]), 1.)
assert observed['auroc'] == 1 and observed['tpr'] == 1 and observed['fpr'] == 0
lo, hi = wilson(0, 100)
assert lo == 0 and 0.03 < hi < 0.04
lo, hi = wilson(100, 100)
assert 0.96 < lo < 0.97 and hi == 1
assert wilson(50, 100)[0] < .5 < wilson(50, 100)[1]
print('PASS: heldout orientation and finite-sample binomial intervals including zero errors')
