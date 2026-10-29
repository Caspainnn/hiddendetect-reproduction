# Whole-branch review and fixes before scoring

One fresh-context read-only reviewer examined routing and dependencies. It found
no index-map or Gram-centering error, and independently compared tied-score
bootstrap metrics with sklearn (100 resamples, max error about1.1e-16).

Three important findings, all addressed before any routing model fitting:

1. GA/DE restart candidates were assessed but did not replace population members.
   A forced-stagnation DE regression test failed, then passed after restoring the
   previous round's better-candidate replacement semantics.
2. Cache file hashes did not alone bind extraction source or full completion.
   The extraction source is committed in85b2c51 (same source as3dfbce9), before
   inference. Completed metadata/state are validated against that historical
   commit; final training source can differ in search-only logic. Old cache seal,
   shapes/dtype and full row completion are independently checked. Incomplete and
   incorrect-source negative tests added, observed RED then GREEN.
3. Archive verifier trusted saved labels/folds/SD/selection. It now aligns IDs and
   labels to manifest, rebuilds group folds, recomputes sample SD, validates all
   prescribed methods/seeds, and independently replays singleton/raw-best/oneSE
   selection. Selection preference has a dedicated synthetic test.

Ruling: preserve original extraction code commit and metadata rather than
re-extracting for a search-only fix. Readout and padding functions did not change.
No test scores were read to make these fixes. Unit suite: six tests green.
