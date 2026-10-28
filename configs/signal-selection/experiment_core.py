"""Small deterministic contracts shared by data, search and evaluation."""
import math
import re
import unicodedata
import numpy as np

def normalize_prompt(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))

def prompt_label(value):
    if value not in ('safe', 'unsafe'):
        raise ValueError('Unsupported prompt-harmfulness label')
    return int(value == 'unsafe')

def conservative_threshold(safe_scores, target_fpr=0.05):
    scores = np.asarray(safe_scores, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all() or not 0 < target_fpr < 1:
        raise ValueError('Invalid safe calibration scores/FPR')
    index = min(len(scores) - 1, max(0, math.ceil((1 - target_fpr) * len(scores)) - 1))
    return float(np.sort(scores)[index])

def confusion(labels, scores, threshold):
    truth = np.asarray(labels, dtype=int)
    predicted = np.asarray(scores) > threshold
    return {'tn': int(((truth == 0) & ~predicted).sum()), 'fp': int(((truth == 0) & predicted).sum()),
            'fn': int(((truth == 1) & ~predicted).sum()), 'tp': int(((truth == 1) & predicted).sum())}

def mask_key(mask):
    return ''.join('1' if x else '0' for x in mask)

def binary_pso(dimensions, objective, seed, budget=500, particles=20, rounds=25):
    if dimensions < 1 or budget < 1:
        raise ValueError('Empty search')
    rng = np.random.default_rng(seed)
    positions = rng.random((particles, dimensions)) < 0.5
    positions[0] = True
    velocities = rng.uniform(-1, 1, (particles, dimensions))
    personal = positions.copy()
    personal_values = np.full(particles, -np.inf)
    best, best_value = positions[0].copy(), -np.inf
    cache = {}
    for _ in range(rounds):
        for i in range(particles):
            if not positions[i].any():
                positions[i, rng.integers(dimensions)] = True
            key = mask_key(positions[i])
            if key not in cache:
                if len(cache) >= budget:
                    return {'best_mask': best.tolist(), 'best_fitness': best_value,
                            'unique_evaluations': len(cache), 'evaluated_masks': list(cache)}
                cache[key] = float(objective(positions[i].copy()))
            value = cache[key]
            if value > personal_values[i]:
                personal_values[i], personal[i] = value, positions[i].copy()
            if value > best_value:
                best_value, best = value, positions[i].copy()
        velocities = (0.729 * velocities + 1.494 * rng.random(positions.shape) * (personal.astype(float) - positions)
                      + 1.494 * rng.random(positions.shape) * (best.astype(float) - positions))
        velocities = np.clip(velocities, -6, 6)
        positions = rng.random(positions.shape) < 1 / (1 + np.exp(-velocities))
    return {'best_mask': best.tolist(), 'best_fitness': best_value,
            'unique_evaluations': len(cache), 'evaluated_masks': list(cache)}
