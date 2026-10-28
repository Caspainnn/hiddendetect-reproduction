"""Evaluate every pre-frozen candidate with independent safe calibration."""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from experiment_core import conservative_threshold, confusion
from train_search import centroid_features, head_scores

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z / total
    center = (p + z*z / (2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [0. if successes == 0 else max(0., center-half),
            1. if successes == total else min(1., center+half)]

def metrics(y, score, threshold):
    counts = confusion(y, score, threshold)
    return {**counts, 'auroc': float(roc_auc_score(y, score)),
        'average_precision': float(average_precision_score(y, score)),
        'tpr': counts['tp'] / (counts['tp'] + counts['fn']),
        'fpr': counts['fp'] / (counts['fp'] + counts['tn'])}

def main():
    begin = time.monotonic()
    out = ROOT / 'results/signal-selection'
    target = out / 'evaluation.json'
    if target.exists():
        print('ALREADY_EVALUATED; refusing overwrite', flush=True)
        return
    freeze_path = out / 'freeze.json'
    freeze_hash = digest(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    manifest_path = Path(__file__).parent / 'manifest.json'
    assert digest(manifest_path) == freeze['manifest_sha256']
    assert digest(out / 'centroids.npy') == freeze['centroids_sha256']
    assert digest(out / 'search.json') == freeze['search_sha256']
    rows = json.loads(manifest_path.read_text())['samples']
    cache = ROOT / 'cache/signal-selection'
    assert json.loads((cache / 'checkpoint.json').read_text())['next_index'] == len(rows)
    assert json.loads((cache / 'metadata.json').read_text())['manifest_sha256'] == freeze['manifest_sha256']
    for candidate in freeze['candidates']:
        if 'file' in candidate:
            assert digest(out / candidate['file']) == candidate['sha256']
    references = np.load(out / 'centroids.npy')
    h = np.load(cache / 'H.npy', mmap_mode='r')
    f = np.load(cache / 'F.npy', mmap_mode='r')
    logits = np.load(cache / 'L.npy', mmap_mode='r')
    scores = {}
    role_indices = {}
    for role in ('calibration', 'test_aegis', 'test_jbb', 'test_toxicchat'):
        ix = np.array([i for i, row in enumerate(rows) if row['role'] == role])
        role_indices[role] = ix
        # Dedupe selected one row per group, so group resampling == row resampling.
        assert len(set(rows[i]['group_id'] for i in ix)) == len(ix)
        x = np.concatenate([f[ix], centroid_features(h[ix], references)], axis=1)
        for candidate in freeze['candidates']:
            if candidate['kind'] == 'interval':
                values = np.trapz(np.asarray(f[ix, candidate['lo']:candidate['hi']+1], dtype=np.float64), axis=1)
            else:
                with np.load(out / candidate['file']) as artifact:
                    head = {key: artifact[key] for key in ('mean', 'scale', 'coef', 'intercept')}
                    values = head_scores(head, logits[ix] if candidate['kind'] == 'logit' else x[:, artifact['mask']])
            assert np.isfinite(values).all()
            scores[(role, candidate['name'])] = values
    thresholds = {}
    cy = np.array([rows[i]['label'] for i in role_indices['calibration']])
    for candidate in freeze['candidates']:
        name = candidate['name']
        values = scores[('calibration', name)]
        thresholds[name] = conservative_threshold(values[cy == 0], .05)
    report = {'freeze_sha256': freeze_hash, 'manifest_sha256': freeze['manifest_sha256'],
        'code_sha256': digest(Path(__file__)), 'threshold_rule': 'strict score > safe calibration 95th order statistic',
        'bootstrap': {'iterations': 500, 'seed': 20261028, 'unit': 'deduplicated prompt group',
            'stratified_by_label': True, 'paired_across_methods': True,
            'scope': 'conditional on frozen training, selection and calibration; no multiplicity adjustment'},
        'label_scope': {'test_aegis': 'human prompt harmfulness', 'test_jbb': 'benchmark benign/harmful behavior',
                        'test_toxicchat': 'human toxicity proxy; separate task'},
        'thresholds': thresholds, 'calibration': {}, 'tests': {}}
    score_path = out / 'heldout-scores.jsonl'
    with score_path.open('w') as stream:
        for role, ix in role_indices.items():
            for j, index in enumerate(ix):
                stream.write(json.dumps({'id': rows[index]['id'], 'group_id': rows[index]['group_id'], 'role': role,
                    'label': rows[index]['label'], 'scores': {c['name']: float(scores[(role, c['name'])][j]) for c in freeze['candidates']}}) + '\n')
    for candidate in freeze['candidates']:
        name = candidate['name']
        report['calibration'][name] = metrics(cy, scores[('calibration', name)], thresholds[name])
        assert report['calibration'][name]['fpr'] <= .05
    for role in ('test_aegis', 'test_jbb', 'test_toxicchat'):
        ix = role_indices[role]
        y = np.array([rows[i]['label'] for i in ix])
        rng = np.random.default_rng(20261028)
        groups = [np.flatnonzero(y == label) for label in (0, 1)]
        resamples = [np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]) for _ in range(500)]
        baseline_values = scores[(role, 'all64')]
        baseline_boot = [metrics(y[b], baseline_values[b], thresholds['all64']) for b in resamples]
        report['tests'][role] = {}
        for candidate in freeze['candidates']:
            name = candidate['name']
            values = scores[(role, name)]
            observed = metrics(y, values, thresholds[name])
            observed['binomial_wilson_ci95'] = {
                'tpr': wilson(observed['tp'], observed['tp'] + observed['fn']),
                'fpr': wilson(observed['fp'], observed['fp'] + observed['tn'])}
            boot = [metrics(y[b], values[b], thresholds[name]) for b in resamples]
            observed['ci95'] = {key: np.quantile([v[key] for v in boot], [.025, .975]).tolist()
                for key in ('auroc', 'average_precision', 'tpr', 'fpr')}
            observed['paired_delta_vs_all64_ci95'] = {key: np.quantile([v[key]-ref[key] for v, ref in zip(boot, baseline_boot)], [.025, .975]).tolist()
                for key in ('auroc', 'tpr', 'fpr')}
            report['tests'][role][name] = observed
        print('TEST_COMPLETE', role, len(ix), flush=True)
    assert digest(freeze_path) == freeze_hash
    report['seconds'] = time.monotonic()-begin
    report['scores_sha256'] = digest(score_path)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(target)
    print('SEALED_EVALUATION_COMPLETE', report['seconds'], flush=True)

if __name__ == '__main__':
    main()
