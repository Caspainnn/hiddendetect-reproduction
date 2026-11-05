"""Independent aggregate-only audit of existing OC-H1 scores; no torch or SSH.

Run: python cache/verify_oc4_statistics_remote.py [--root WORKTREE]
Test the independent tied-score ROC implementation: add --self-test.
Only verification-statistics-independent.json is written; no sample data printed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

IDS = ['OC-4-1-1', 'OC-4-2-1', 'OC-4-3-1', 'OC-4-4-1',
       'OC-4-1-2', 'OC-4-1-3', 'OC-4-3-2', 'OC-4-4-2']
SEEDS = [20261104, 20261105, 20261106]
TOL = 1e-12


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def histogram_roc(safe_bins, danger_bins, size):
    """Bin tied observations before constructing ROC vertices, high scores first."""
    neg = np.bincount(safe_bins, minlength=size)[::-1]
    pos = np.bincount(danger_bins, minlength=size)[::-1]
    # Empty bins must be removed: their repeated vertices need no interpolation.
    used = neg + pos > 0
    fp = np.r_[0., np.cumsum(neg[used]) / len(safe_bins)]
    tp = np.r_[0., np.cumsum(pos[used]) / len(danger_bins)]
    auc = float(np.sum(np.diff(fp) * (tp[1:] + tp[:-1]) / 2))
    end = np.searchsorted(fp, .1, side='right')
    boundary = np.interp(.1, fp[end-1:end+1], tp[end-1:end+1])
    x = np.r_[fp[:end], .1]
    z = np.r_[tp[:end], boundary]
    area = np.sum(np.diff(x) * (z[1:] + z[:-1]) / 2)
    return auc, float(.5 * (1 + (area - .005) / .095))


def metric(y, values):
    unique, bins = np.unique(values, return_inverse=True)
    return histogram_roc(bins[y == 0], bins[y == 1], len(unique))


def self_test():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(524)
    y = np.array([0] * 13 + [1] * 9)
    maximum = 0.
    count = 0
    for case in range(43):
        s = (np.ones(22) if case == 0 else np.arange(22) if case == 1 else
             -np.arange(22) if case == 2 else rng.integers(0, 6, 22))
        for _ in range(100):
            ix = np.concatenate([rng.choice(np.flatnonzero(y == k), (y == k).sum()) for k in (0, 1)])
            actual = metric(y[ix], s[ix])
            expected = [roc_auc_score(y[ix], s[ix]), roc_auc_score(y[ix], s[ix], max_fpr=.1)]
            maximum = max(maximum, float(np.max(np.abs(np.array(actual) - expected))))
            assert maximum <= TOL, 'Independent ROC self-test failed'
            count += 1
    return {'replicates': count, 'maximum_sklearn_difference': maximum}


def main(root):
    out = root / 'results/oc-heads'
    complete = load(out / 'evaluation-complete.json')
    score_path = out / 'scores.jsonl'
    assert sha(score_path) == complete['scores_sha256'], 'Scores hash mismatch'
    assert sha(out / 'selection.json') == complete['selection_sha256']
    selection = load(out / 'selection.json')
    lock = load(out / 'source-lock.json')
    assert sha(out / 'source-lock.json') == selection['source_lock_sha256']
    assert selection['primary_seed'] == SEEDS[0] and selection['seeds'] == SEEDS
    assert lock['ids'] == IDS and lock['seeds'] == SEEDS
    seal = load(out / 'pre-evaluation-seal.json')
    assert seal['selection_sha256'] == sha(out / 'selection.json')
    assert seal['source_lock_sha256'] == sha(out / 'source-lock.json')
    assert sha(out / 'manifest.json') == lock['manifest_sha256']
    assert sha(out / 'search.json') == selection['search_sha256']
    for path, digest in lock['sha256'].items():
        assert sha(root / path) == digest, 'Frozen source mismatch'
    rows = [json.loads(line) for line in score_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 3 * (400 + 1000 + 1082)
    assert {r['seed'] for r in rows} == set(SEEDS)
    assert {r['role'] for r in rows} == {'cal', 'C', 'RV'}
    arrays = {}
    expected_counts = {'cal': [400, 0], 'C': [500, 500], 'RV': [300, 782]}
    for role in expected_counts:
        reference = None
        for seed in SEEDS:
            chosen = [r for r in rows if r['role'] == role and r['seed'] == seed]
            keys = [(r['id'], r['label']) for r in chosen]
            assert len({r['id'] for r in chosen}) == len(chosen)
            if reference is None:
                reference = keys
            assert keys == reference, 'Sample order/labels differ across seeds'
            y = np.array([r['label'] for r in chosen])
            s = np.array([r['scores'] for r in chosen], dtype=float)
            assert s.shape == (sum(expected_counts[role]), 9) and np.isfinite(s).all()
            assert [int((y == k).sum()) for k in (0, 1)] == expected_counts[role]
            arrays[seed, role] = (y, s)
    maximum_point = maximum_threshold = maximum_interval = 0.
    count = intervals = 0
    for seed in SEEDS:
        stored_cut = load(out / f'thresholds-{seed}.json')
        assert stored_cut['names'] == ['B'] + IDS
        assert stored_cut['ordinal'] == 381 and stored_cut['strict_greater'] is True
        assert stored_cut['calibration_count'] == 400
        cuts = np.sort(arrays[seed, 'cal'][1], axis=0)[380]
        maximum_threshold = max(maximum_threshold, float(np.max(np.abs(cuts - stored_cut['values']))))
        assert maximum_threshold <= TOL, 'Threshold replay mismatch'
        for role in ['C', 'RV']:
            y, s = arrays[seed, role]
            stored = load(out / f'seed-{seed}-{role}.json')
            assert [m['id'] for m in stored] == ['B'] + IDS
            for j, m in enumerate(stored):
                auc, partial = metric(y, s[:, j])
                maximum_point = max(maximum_point, abs(auc - m['auroc']), abs(partial - m['pauc']))
                assert maximum_point <= TOL, 'Point metric mismatch'
                assert abs(m['threshold'] - cuts[j]) <= TOL
                for label, key in [(0, 'fpr'), (1, 'tpr')]:
                    n = int((y == label).sum())
                    k = int((s[y == label, j] > cuts[j]).sum())
                    assert (k, n) == (m[key + '_count'], m[key + '_n']), 'Confusion count mismatch'
                    assert abs(k / n - m[key]) <= TOL
                count += 1
            if seed != SEEDS[0]:
                continue
            evaluation = load(out / f'evaluation-{role}.json')
            assert evaluation['evidence'] == 'historical_diagnostic_only'
            assert evaluation['primary_seed'] == seed
            assert evaluation['family'] == 8 and evaluation['repeats'] == 10000
            assert evaluation['bootstrap_seed'] == SEEDS[0]
            assert evaluation['pauc_ci_coverage'] == .99375
            for expected, primary in zip(stored, evaluation['metrics']):
                for key, value in expected.items():
                    assert primary[key] == value, 'Primary and seed summary mismatch'
            groups = [np.flatnonzero(y == k) for k in (0, 1)]
            rng = np.random.default_rng(SEEDS[0])
            # Preserve predeclared RNG ordering: all normal draws, then dangerous.
            draw0 = rng.choice(groups[0], (10000, len(groups[0])))
            draw1 = rng.choice(groups[1], (10000, len(groups[1])))
            distribution = np.empty((10000, 9, 2))
            for j in range(9):
                unique, bins = np.unique(s[:, j], return_inverse=True)
                for b in range(10000):
                    distribution[b, j] = histogram_roc(bins[draw0[b]], bins[draw1[b]], len(unique))
            for j, m in enumerate(evaluation['metrics'][1:], 1):
                delta = distribution[:, j] - distribution[:, 0]
                for index, key, probs in [(0, 'auc_delta_ci95', [.025, .975]),
                                           (1, 'pauc_delta_ci99375', [.025/8, 1-.025/8])]:
                    ci = np.quantile(delta[:, index], probs)
                    maximum_interval = max(maximum_interval, float(np.max(np.abs(ci - m[key]))))
                    assert maximum_interval <= TOL, 'Bootstrap interval mismatch'
                    intervals += 1
                assert abs(m['delta_pauc'] - (stored[j]['pauc'] - stored[0]['pauc'])) <= TOL
                lo, hi = m['pauc_delta_ci99375']
                decision = 'positive_interval' if lo > 0 else 'worse' if hi < 0 else 'unconfirmed'
                assert m['historical_decision'] == decision
    result = {'status': 'passed', 'script_sha256': sha(Path(__file__)),
              'scores_sha256': sha(score_path), 'source_lock_sha256': sha(out / 'source-lock.json'),
              'point_metrics_checked': count, 'bootstrap_repeats_per_cohort': 10000,
              'bootstrap_intervals_checked': intervals, 'maximum_point_metric_difference': maximum_point,
              'maximum_threshold_difference': maximum_threshold, 'maximum_interval_difference': maximum_interval,
              'confusion_counts_exact': True, 'tolerance': TOL,
              'independent_tied_ROC': True, 'sklearn_self_test': self_test(),
              'independent_confirmation': False}
    (out / 'verification-statistics-independent.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    else:
        main(args.root)
