"""AUROC-first fixed interval selection. Input F is exact original 32-layer F."""
import numpy as np
from scipy.stats import rankdata

INTERVALS = [(a, b) for a in range(32) for b in range(a + 1, 32)]
CONTRASTS = {'B1-B0': ('B1', 'B0'), 'A-B0': ('A', 'B0'),
             'B-B0': ('B', 'B0'), 'B-A': ('B', 'A'), 'B-B1': ('B', 'B1')}


def auc(y, scores):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    assert set(np.unique(y)) == {0, 1} and len(y) == len(scores)
    assert np.isfinite(scores).all()
    n1, n0 = int(y.sum()), int((y == 0).sum())
    ranks = rankdata(scores, method='average', axis=0)
    return (ranks[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n0 * n1)


def interval_scores(f):
    f = np.asarray(f, dtype=np.float64)
    assert f.ndim == 2 and f.shape[1] == 32 and np.isfinite(f).all()
    integral = np.c_[np.zeros(len(f)), np.cumsum((f[:, :-1] + f[:, 1:]) / 2, axis=1)]
    return np.column_stack([integral[:, b] - integral[:, a] for a, b in INTERVALS])


def choose(objective, full_auc):
    objective, full_auc = np.asarray(objective), np.asarray(full_auc)
    best = np.flatnonzero(objective >= objective.max() - 1e-12)
    def tie(i):
        a, b = INTERVALS[i]
        return (-full_auc[i], abs(a - 16) + abs(b - 29), b - a, a)
    return int(min(best, key=tie))


def select(f, y, group_ids, seed=20261030):
    """Require one representative per near-duplicate group before selection."""
    y = np.asarray(y)
    assert len(group_ids) == len(y) == len(set(group_ids))
    classes = [np.flatnonzero(y == k) for k in (0, 1)]
    assert all(len(ix) >= 2 and len(ix) % 2 == 0 for ix in classes)
    scores = interval_scores(f)
    full = auc(y, scores)
    rng, halves, counts = np.random.default_rng(seed), [], []
    for _ in range(100):
        shuffled = [rng.permutation(ix) for ix in classes]
        for side in (0, 1):
            ix = np.concatenate([v[side * (len(v)//2):(side+1) * (len(v)//2)] for v in shuffled])
            halves.append(auc(y[ix], scores[ix]))
            counts.append([int((y[ix] == k).sum()) for k in (0, 1)])
    lower = np.quantile(halves, .1, axis=0, method='linear')
    fdv = np.asarray(f)[y == 1].mean(axis=0) - np.asarray(f)[y == 0].mean(axis=0)
    eligible = np.flatnonzero(fdv > fdv[-1])
    b1 = [int(eligible[0]), int(eligible[-1])] if len(eligible) >= 2 else None
    picks = {'B0': [16, 29], 'B1': b1,
             'A': list(INTERVALS[choose(full, full)]),
             'B': list(INTERVALS[choose(lower, full)])}
    return {'intervals': picks, 'full_selection_auroc': full.tolist(),
            'lower_tail_auroc': lower.tolist(), 'FDV': fdv.tolist(),
            'half_sample_class_counts': counts, 'seed': seed,
            'B1_status': 'defined' if b1 else 'undefined_original_endpoint_rule'}


def paired_bootstrap(y, scores, repeats=10000, seed=20261031):
    """Five preregistered AUROC contrasts; paired stratified 99% percentile CI."""
    assert set(scores) == {'B0', 'B1', 'A', 'B'}
    y = np.asarray(y)
    names = ['B0', 'B1', 'A', 'B']
    matrix = np.column_stack([scores[n] for n in names])
    point = dict(zip(names, map(float, auc(y, matrix))))
    classes = [np.flatnonzero(y == k) for k in (0, 1)]
    rng = np.random.default_rng(seed)
    samples = np.empty((repeats, 4))
    for i in range(repeats):
        ix = np.concatenate([rng.choice(v, len(v), replace=True) for v in classes])
        samples[i] = auc(y[ix], matrix[ix])
    out = {}
    for name, (a, b) in CONTRASTS.items():
        delta = samples[:, names.index(a)] - samples[:, names.index(b)]
        out[name] = {'delta': point[a] - point[b],
                     'ci99': np.quantile(delta, [.005, .995], method='linear').tolist()}
    return {'auroc': point, 'contrasts': out, 'bootstrap_repeats': repeats, 'seed': seed}
