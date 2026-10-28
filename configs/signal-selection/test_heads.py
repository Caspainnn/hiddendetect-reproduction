import numpy as np
from train_search import fit_centroids, centroid_features, fit_head, head_scores

rng = np.random.default_rng(7)
h = rng.normal(size=(12, 3, 8)).astype(np.float32)
y = np.array([0] * 6 + [1] * 6)
reference = fit_centroids(h, y)
assert reference.shape == (2, 3, 8)
assert np.allclose(np.linalg.norm(reference, axis=-1), 1)
held_out = rng.normal(size=(4, 3, 8)).astype(np.float32)
before = reference.copy()
features = centroid_features(held_out, reference)
assert features.shape == (4, 3) and np.isfinite(features).all()
held_out[:] = 1e6
assert np.array_equal(reference, before)
x = np.array([[-2., 1], [-1., 1], [1., 1], [2., 1]])
head = fit_head(x, np.array([0, 0, 1, 1]))
assert np.array_equal(head['mean'], x.mean(axis=0))
assert np.isfinite(head_scores(head, x)).all()
assert head_scores(head, x)[0] < head_scores(head, x)[-1]
assert np.array_equal(head['mean'], x.mean(axis=0))
print('PASS: training-only normalized references/scaler and unsafe score orientation')
