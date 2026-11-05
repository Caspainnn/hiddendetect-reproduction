"""Independent numeric contracts; synthetic data are never experiment evidence."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from core import INTERVALS, auc, choose, interval_scores, select, paired_bootstrap


class Contract(unittest.TestCase):
    def test_jsonl_unicode_line_separator_in_prompt(self):
        from run import read
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'records.jsonl'
            row = {'prompt': 'left\u2028right'}
            p.write_text(json.dumps(row, ensure_ascii=False)+'\n', encoding='utf-8')
            self.assertEqual(read(p), [row])

    def test_auc_ties_and_all_intervals(self):
        rng = np.random.default_rng(11)
        f = rng.integers(-2, 3, (20, 32)).astype(float)
        y = np.tile([0, 1], 10)
        s = interval_scores(f)
        self.assertEqual(s.shape, (20, 496))
        for i, (a, b) in enumerate(INTERVALS):
            np.testing.assert_allclose(s[:, i], np.trapz(f[:, a:b+1], axis=1), atol=1e-12)
            self.assertAlmostEqual(auc(y, s[:, i]), roc_auc_score(y, s[:, i]))

    def test_selection_determinism_and_ties(self):
        f, y = np.zeros((16, 32)), np.tile([0, 1], 8)
        result = select(f, y, list(range(16)))
        self.assertEqual(result['intervals'], {'B0': [16, 29], 'B1': None, 'A': [16, 29], 'B': [16, 29]})
        self.assertEqual(result['half_sample_class_counts'], [[4, 4]] * 200)
        self.assertEqual(result, select(f, y, list(range(16))))
        with self.assertRaises(AssertionError):
            select(f, y, [0] * 16)

    def test_paired_bootstrap_identical_scores(self):
        y = np.tile([0, 1], 10)
        scores = {n: np.arange(20) for n in ('B0', 'B1', 'A', 'B')}
        r = paired_bootstrap(y, scores, repeats=100)
        self.assertEqual(len(r['contrasts']), 5)
        for v in r['contrasts'].values():
            self.assertEqual(v, {'delta': 0., 'ci99': [0., 0.]})


if __name__ == '__main__':
    unittest.main()
