"""Resample cached calibration traces only; never select a range using test labels."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from run_selection import select_range

ROOT = Path(__file__).resolve().parents[1]
calibration = [json.loads(x) for x in (ROOT / 'results/original-calibration-traces.jsonl').read_text().splitlines()]
original = json.loads((ROOT / 'results/original-layer-selection.json').read_text())
signals = np.asarray([x['F'] for x in calibration])
labels = np.asarray([x['label'] for x in calibration])
assert signals.shape == (12, 32) and np.isfinite(signals).all()
assert list(select_range((signals[labels == 1].mean(0) - signals[labels == 0].mean(0)).tolist())[:2]) == original['interval_zero_based_inclusive']
groups = {f'{label}:{modality}': np.asarray([i for i, row in enumerate(calibration) if row['label'] == label and row['has_image'] == modality])
          for label in (0, 1) for modality in (False, True)}
assert sorted(map(len, groups.values())) == [2, 2, 4, 4]
seed, repeats = 20261028, 1000
generator = np.random.default_rng(seed)
counts = Counter()
for _ in range(repeats):
    chosen = np.concatenate([generator.choice(indices, size=len(indices), replace=True) for indices in groups.values()])
    sampled = signals[chosen]
    sampled_labels = labels[chosen]
    discrepancy = sampled[sampled_labels == 1].mean(0) - sampled[sampled_labels == 0].mean(0)
    try:
        s, e, _ = select_range(discrepancy.tolist())
        counts[f'{s}:{e}'] += 1
    except ValueError:
        counts['undefined'] += 1
assert sum(counts.values()) == repeats
report = {'seed': seed, 'repeats': repeats,
          'procedure': 'bootstrap within each class x modality stratum, retaining 2 text + 4 image samples per class; cached forwards only',
          'original_interval': original['interval_zero_based_inclusive'],
          'interval_counts': dict(counts.most_common()),
          'unique_defined_intervals': len([x for x in counts if x != 'undefined']),
          'original_interval_frequency': counts[':'.join(map(str, original['interval_zero_based_inclusive']))] / repeats,
          'default_interval_frequency': counts['16:29'] / repeats,
          'test_scores_used_for_selection': False,
          'scope': 'Sensitivity to resampling the same 12 calibration queries. Not new-source robustness, independent repeated datasets, or an interval optimization.'}
(ROOT / 'results/calibration-bootstrap1000.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
