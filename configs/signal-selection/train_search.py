"""Train/dev-only fitting and bounded selection; no calibration/test score access."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from experiment_core import binary_pso, mask_key

ROOT = Path(__file__).resolve().parents[2]

def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

def fit_centroids(train_hidden, train_labels):
    hidden = unit(train_hidden)
    return unit(np.stack([hidden[np.asarray(train_labels) == label].mean(axis=0) for label in (0, 1)]))

def centroid_features(hidden, references):
    return np.einsum('nld,ld->nl', unit(hidden), references[1] - references[0])

def fit_head(train_x, train_y, penalty='l2', c=1.0):
    x = np.asarray(train_x, dtype=np.float32)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    model = LogisticRegression(C=c, penalty=penalty, solver='liblinear',
        class_weight='balanced', random_state=0, max_iter=1000)
    model.fit((x - mean) / scale, train_y)
    assert model.n_iter_.max() < 1000, 'Head failed to converge'
    return {'mean': mean, 'scale': scale, 'coef': model.coef_[0], 'intercept': float(model.intercept_[0])}

def head_scores(head, x):
    return ((np.asarray(x, dtype=np.float32) - head['mean']) / head['scale']) @ head['coef'] + head['intercept']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    started = time.monotonic()
    out = ROOT / 'results/signal-selection'
    out.mkdir(parents=True, exist_ok=True)
    freeze_path = out / 'freeze.json'
    if freeze_path.exists():
        print('ALREADY_FROZEN; refusing post-test reselection', flush=True)
        return
    manifest_path = Path(__file__).parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    rows = manifest['samples']
    cache = ROOT / 'cache/signal-selection'
    assert json.loads((cache / 'checkpoint.json').read_text())['next_index'] == len(rows)
    assert json.loads((cache / 'metadata.json').read_text())['manifest_sha256'] == digest(manifest_path)
    train = np.array([i for i, row in enumerate(rows) if row['role'] == 'train'])
    dev = np.array([i for i, row in enumerate(rows) if row['role'] == 'dev'])
    ytrain = np.array([rows[i]['label'] for i in train])
    ydev = np.array([rows[i]['label'] for i in dev])
    h = np.load(cache / 'H.npy', mmap_mode='r')
    f = np.load(cache / 'F.npy', mmap_mode='r')
    logits = np.load(cache / 'L.npy', mmap_mode='r')
    references = fit_centroids(h[train], ytrain)
    np.save(out / 'centroids.npy', references)
    # Only train/dev rows are materialized before the pre-test freeze.
    xtrain = np.concatenate([f[train], centroid_features(h[train], references)], axis=1)
    xdev = np.concatenate([f[dev], centroid_features(h[dev], references)], axis=1)
    assert xtrain.shape == (1600, 64) and xdev.shape == (600, 64)
    candidates = []
    search_logs = []
    fitted = {}
    def assess(mask):
        key = mask_key(mask)
        if key not in fitted:
            head = fit_head(xtrain[:, mask], ytrain)
            scores = head_scores(head, xdev[:, mask])
            auc = float(roc_auc_score(ydev, scores))
            fpr, tpr, _ = roc_curve(ydev, scores)
            fitted[key] = (head, auc, float(tpr[fpr <= .05].max()))
        head, auc, tpr = fitted[key]
        return auc - .02 * float(np.mean(mask))
    def save_mask(name, mask):
        fitness = assess(mask)
        head, auc, tpr = fitted[mask_key(mask)]
        path = out / (name + '.npz')
        np.savez(path, **head, mask=mask)
        candidates.append({'name': name, 'kind': 'mask', 'file': path.name,
            'sha256': digest(path), 'groups': int(np.sum(mask)), 'mask': mask.astype(int).tolist(),
            'dev_auroc': auc, 'dev_tpr_at_descriptive_fpr05': tpr, 'fitness': fitness})
    for name, mask in [('F32', np.arange(64) < 32), ('C32', np.arange(64) >= 32), ('all64', np.ones(64, bool))]:
        save_mask(name, mask)
    for lo, hi in [(16, 29), (17, 29)]:
        candidates.append({'name': 'original_%d_%d' % (lo, hi), 'kind': 'interval', 'lo': lo, 'hi': hi,
            'groups': hi - lo + 1, 'dev_auroc': float(roc_auc_score(ydev, np.trapz(np.asarray(f[dev, lo:hi+1], dtype=np.float64), axis=1)))})
    intervals = []
    for lo in range(31):
        for hi in range(lo + 1, 32):
            auc = float(roc_auc_score(ydev, np.trapz(np.asarray(f[dev, lo:hi+1], dtype=np.float64), axis=1)))
            intervals.append({'lo': lo, 'hi': hi, 'dev_auroc': auc})
    chosen = max(intervals, key=lambda item: item['dev_auroc'])
    candidates.append({'name': 'dev_interval', 'kind': 'interval', 'groups': chosen['hi']-chosen['lo']+1, **chosen})
    search_logs.append({'method': 'interval', 'evaluations': len(intervals), 'candidates': intervals})
    begin = time.monotonic()
    logit_head = fit_head(logits[train], ytrain)
    path = out / 'first_logit.npz'
    np.savez(path, **logit_head)
    candidates.append({'name': 'first_logit', 'kind': 'logit', 'file': path.name, 'sha256': digest(path),
        'groups': None, 'dimensions': 32000, 'dev_auroc': float(roc_auc_score(ydev, head_scores(logit_head, logits[dev]))),
        'fit_seconds': time.monotonic()-begin, 'scope': 'First-to-Know-style adapted final next-token full-logit probe'})
    l1_options = []
    for c in [.01, .1, 1., 10.]:
        head = fit_head(xtrain, ytrain, penalty='l1', c=c)
        mask = head['coef'] != 0
        auc = float(roc_auc_score(ydev, head_scores(head, xdev)))
        l1_options.append((auc-.02*mask.mean(), c, head, mask, auc))
    fitness, c, head, mask, auc = max(l1_options, key=lambda item: item[0])
    path = out / 'L1.npz'
    np.savez(path, **head, mask=np.ones(64, bool))
    candidates.append({'name': 'L1', 'kind': 'mask', 'file': path.name, 'sha256': digest(path),
        'groups': int(mask.sum()), 'nonzero_mask': mask.astype(int).tolist(), 'C': c, 'dev_auroc': auc, 'fitness': float(fitness)})
    search_logs.append({'method': 'L1', 'evaluations': 4, 'options': [{'C': z[1], 'groups': int(z[3].sum()), 'dev_auroc': z[4], 'fitness': float(z[0])} for z in l1_options]})
    # Forward greedy counts every unique candidate proposed, including rejected additions.
    begin = time.monotonic()
    current = np.zeros(64, bool)
    best_value = -np.inf
    evaluated = []
    while len(evaluated) < 500:
        proposals = []
        for index in np.flatnonzero(~current):
            proposal = current.copy(); proposal[index] = True
            key = mask_key(proposal)
            if key in evaluated:
                continue
            if len(evaluated) == 500:
                break
            value = assess(proposal); evaluated.append(key)
            proposals.append((value, proposal))
        if not proposals:
            break
        value, proposal = max(proposals, key=lambda item: item[0])
        if value <= best_value:
            break
        best_value, current = value, proposal
    save_mask('greedy', current)
    search_logs.append({'method': 'greedy', 'unique_evaluations': len(evaluated), 'evaluated_masks': evaluated, 'seconds': time.monotonic()-begin})
    for seed in range(5):
        for method in ('random', 'bpso'):
            begin = time.monotonic()
            if method == 'bpso':
                result = binary_pso(64, assess, seed=seed, budget=500, particles=20, rounds=25)
                mask = np.array(result['best_mask'], bool)
            else:
                rng = np.random.default_rng(seed)
                values = {}
                # Same dimensionality and 500 distinct-mask maximum as BPSO.
                while len(values) < 500:
                    proposal = rng.random(64) < .5
                    if not proposal.any():
                        continue
                    key = mask_key(proposal)
                    if key not in values:
                        values[key] = assess(proposal)
                key = max(values, key=values.get)
                mask = np.array([x == '1' for x in key])
                result = {'best_mask': mask.tolist(), 'best_fitness': values[key], 'unique_evaluations': len(values), 'evaluated_masks': list(values)}
            save_mask('%s_seed%d' % (method, seed), mask)
            search_logs.append({'method': method, 'seed': seed, **result, 'seconds': time.monotonic()-begin})
            print('SEARCH_DONE', method, seed, result['unique_evaluations'], int(mask.sum()), flush=True)
    (out / 'search.json').write_text(json.dumps(search_logs, indent=2) + '\n')
    freeze = {'status': 'frozen_before_calibration_and_test_scoring', 'manifest_sha256': digest(manifest_path),
        'code_sha256': digest(Path(__file__)), 'centroids_sha256': digest(out / 'centroids.npy'),
        'objective': 'dev AUROC - 0.02 * selected_groups/64', 'train_rows': len(train), 'dev_rows': len(dev),
        'centroid_scope': 'training-only cosine centroid simplification, not original RCS',
        'unique_heads_fitted': len(fitted), 'seconds': time.monotonic()-started, 'candidates': candidates,
        'search_sha256': digest(out / 'search.json')}
    temporary = freeze_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(freeze, indent=2) + '\n')
    temporary.replace(freeze_path)
    print('PRETEST_FREEZE_COMPLETE', len(candidates), 'SECONDS', freeze['seconds'], flush=True)

if __name__ == '__main__':
    main()
