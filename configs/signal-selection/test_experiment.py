import math
import numpy as np
from experiment_core import normalize_prompt, prompt_label, conservative_threshold, confusion, mask_key, binary_pso

assert normalize_prompt('  HELLO,  World! ') == normalize_prompt('hello world')
assert prompt_label('safe') == 0 and prompt_label('unsafe') == 1
for label in [None, 'refusal', 'unknown']:
    try:
        prompt_label(label)
    except ValueError:
        pass
    else:
        raise AssertionError('Ambiguous prompt label accepted')
for values in [np.array([0, 0, 1, 1, 2]), np.arange(100)]:
    threshold = conservative_threshold(values, 0.05)
    assert np.mean(values > threshold) <= 0.05
assert confusion(np.array([0, 0, 1, 1]), np.array([0., 1., 1., 2.]), 1.) == {'tn': 2, 'fp': 0, 'fn': 1, 'tp': 1}
assert mask_key(np.array([1, 0, 1], bool)) == mask_key(np.array([True, False, True]))
def objective(mask):
    assert mask.any()
    return float(mask[0]) - float(mask[1:].sum()) / 10
first = binary_pso(6, objective, seed=0, budget=20, particles=4, rounds=8)
second = binary_pso(6, objective, seed=0, budget=20, particles=4, rounds=8)
assert first['best_mask'] == second['best_mask']
assert first['unique_evaluations'] <= 20 and len(set(first['evaluated_masks'])) == first['unique_evaluations']
assert all(key != '000000' for key in first['evaluated_masks'])
assert math.isfinite(first['best_fitness'])
print('PASS: label semantics, normalized text, threshold ties, orientation, deterministic BPSO and unique-mask budget')
