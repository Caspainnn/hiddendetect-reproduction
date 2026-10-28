"""Read-only integrity and metric recomputation from published artifacts."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from evaluate_frozen import digest, metrics

ROOT = Path(__file__).resolve().parents[2]

def main():
    out = ROOT / 'results/signal-selection'
    manifest_path = Path(__file__).with_name('manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert digest(ROOT / 'logs/signal-selection-download.json') == manifest['raw_lock_sha256']
    freeze = json.loads((out / 'freeze.json').read_text())
    evaluation = json.loads((out / 'evaluation.json').read_text())
    assert freeze['manifest_sha256'] == digest(manifest_path) == evaluation['manifest_sha256']
    assert digest(out / 'freeze.json') == evaluation['freeze_sha256']
    assert digest(out / 'centroids.npy') == freeze['centroids_sha256']
    assert digest(out / 'search.json') == freeze['search_sha256']
    assert digest(out / 'heldout-scores.jsonl') == evaluation['scores_sha256']
    for c in freeze['candidates']:
        if 'file' in c:
            assert digest(out / c['file']) == c['sha256'], c['name']
    searches = json.loads((out / 'search.json').read_text())
    assert searches[0]['evaluations'] == 496
    counts = Counter()
    for run in searches:
        if run['method'] not in ('random', 'bpso', 'greedy'):
            continue
        keys = run['evaluated_masks']
        assert len(set(keys)) == len(keys) == run['unique_evaluations'] <= 500
        assert all(len(key) == 64 and '1' in key and set(key) <= {'0', '1'} for key in keys)
        if run['method'] != 'greedy':
            counts[run['method']] += 1
            assert len(run['best_mask']) == 64 and any(run['best_mask'])
    assert counts == {'random': 5, 'bpso': 5}
    heldout = [json.loads(line) for line in (out / 'heldout-scores.jsonl').read_text().splitlines()]
    lookup = {row['id']: row for row in manifest['samples']}
    assert len({row['id'] for row in heldout}) == len(heldout) == 2600
    for row in heldout:
        assert all(row[k] == lookup[row['id']][k] for k in ('role', 'group_id', 'label'))
    for role in ('calibration', 'test_aegis', 'test_jbb', 'test_toxicchat'):
        subset = [row for row in heldout if row['role'] == role]
        y = np.array([row['label'] for row in subset])
        for c in freeze['candidates']:
            name = c['name']
            score = np.array([row['scores'][name] for row in subset])
            actual = metrics(y, score, evaluation['thresholds'][name])
            expected = evaluation['calibration'][name] if role == 'calibration' else evaluation['tests'][role][name]
            assert all(np.isclose(actual[k], expected[k], rtol=0, atol=1e-12) for k in actual), (role, name)
            if role == 'calibration':
                assert actual['fpr'] <= .05
    report = {'status': 'pass', 'manifest_rows': 4800, 'heldout_rows': len(heldout),
        'frozen_candidates': len(freeze['candidates']), 'five_seeds_each': dict(counts),
        'all_hashes_verified': True, 'all_point_metrics_recomputed': True,
        'freeze_sha256': evaluation['freeze_sha256']}
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
