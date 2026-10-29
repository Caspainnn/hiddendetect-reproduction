# Independent whole-branch review and pre-scoring fixes

One fresh-context read-only reviewer examined the complete processing branch and
dependencies. Its five CPU tests passed. No concrete error was found in gamma
scaling, fold-only relation standardization, duplicate-mean normalization, lambda1,
CV selection, or ten-contrast bootstrap correction.

Two important findings addressed before supervised fitting/scoring:

1. Resume metadata omitted artifact lock and incompletely checked environment.
   Metadata now binds the artifact-lock hash, torch/numpy/transformers versions,
   full extraction-source map, shape and anchors. Resume compares every field
   except the historical commit ID; the original source commit is preserved.
2. Extraction-path tests omitted module mapping, masked user support and negative
   source/completion checks. Added exact change/delta/attention/MLP summary checks,
   prefix/suffix/padding perturbation test, incomplete-cache/source/lock rejection.
   New test import failed before implementation, then all seven tests passed.

Operational findings before any completed extraction checkpoint:

* Initial array creation failed because JSON shape lists were passed directly to
  NumPy memmap. The preserved128-byte header proves no array payload completed;
  tuple conversion has its own regression test. Original metadata/header/log kept.
* The next strict identity gate stopped on training row2 with old B coordinate
  difference1.52587890625e-05, prefix0, peak15277868032 bytes. Normalizing only the
  two gathered positions changes the GPU reduction shape. A train32 diagnostic
  using the original whole-sequence normalization restored old B difference0,
  prefix0, peak15286223872 bytes. No thresholds were relaxed. The extractor now
  normalizes the whole512-token tensor before gathering, exactly as R16.

Ruling: preserve failed-start metadata and diagnostic slices privately, reset only
these uncommitted processing outputs (no completed checkpoint), retain the locked
manifest and all old sealed caches. No heldout metrics or labels were examined
for these repairs. This changes implementation arithmetic, not feature recipes,
candidate families, training labels, selection rules or evaluation policy.
