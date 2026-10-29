# R16–R18 preregistered verification

Previous round: template-last beat the six sampled user positions; adding user-last
had no confirmed gain. Prefix drift vanished on 32 fixed512 train probes.

## Locked hypotheses

* R16: rerun full template-last and user-last heads with right-padding to512,
  identical valid token IDs, proper attention mask. Check protected prefix drift
  over all rows. Keep variable-shape counterparts; do not count numerical changes
  as feature-selection improvements.
* R17: fixed512 template-last-minus1/minus2, all32 layers and4096 coordinates,
  test whether a nearby suffix token improves on template-last.
* R18: a32-bit mask picks user-last(1) or template-last(0) independently per layer.
  Every candidate has exactly131072 coordinates, lambda1 ridge, train-only
  centering. Full hidden states, no projection, no base-model tuning.

## Data and independence

Train1600/dev600 and immutable benchmarkv1 nonprobe2760 rows stay unchanged.
Previous confirmation cohorts are excluded from new confirmation, not called fresh
again. Fresh Aegis target500/class, seed20261129, candidate cap1100/class,
token limit512 and same exact/near-duplicate exclusion over all old private data.
If counts fall short, floor minimum class count to25, minimum200/class; counts-only
amendment before inference. Same-source confirmation is not a new-domain test.

Cache fixed512 readouts[user_last,last-2,last-1,last] for all rows, variable
[user_last,last] for fresh rows only; old variable readouts reused by immutable ID.
Additional cache cap7GiB. Never delete old sealed caches. Hash actual caches,
input records, tokenizer/model files, code and frozen models before evaluation.

## Selection and algorithm fairness

Repeated stratified group5-fold CV, seeds20261115/16, on train1600 only.
Float64 additive layer Gram ridge, mathematically checked against explicit primal
ridge; all candidates use the same arithmetic. Retain frozen old float32 B as
reference. Objective=min(variable,fixed) of(mean fold standardized pAUC(.1)
minus.25*fold SD), minus.002*user-layer fraction. Dev is diagnostic only.

Five methods(random,BPSO,GA,DE,single-swap-then-pair scan), five seeds20261201–05,
64 unique masks/run including common34 warm masks(all-template,all-user,32 singles).
Warm masks charged to every run; same best12 warm population. Then30 proposals
accepted uniquely. Global deterministic objective cache may be shared but run
budgets remain charged independently. Scan expands best nonzero singleton into
30 seeded distinct two-swap masks, keeps baseline eligible. Other operators reuse
the preceding round parameters. Random density includes1/32,.05,.1,.25,.5.
Publish all winners, no test-based seed choice. Final candidate selected by trainCV;
prefer all-template within one-SE heuristic of raw best (overlapping-fold SE is
not a confidence interval). Best singleton is also a simple control.

## Freeze, evaluation, interpretation

Before scoring: commit manifest, completed cache seal, all heads, CV, masks and
selection. Verify actual bytes against frozen commit. Threshold strictly exceeds
safe calibration400 sorted[379], global Aegis/source Toxic separately. Same fixed
v1 conditions plus fresh confirmation. AUROC/AP/pAUC(.1)/TPR/FPR;5000 stratified
paired bootstrap, seed20261209, conditional on fitted heads/calibration.

Confirmatory family8 contrasts on fresh: suffix-1/-2 vs fixed-template,
CV-selected fixed/variable vs matched template, GA fixed five-seed mean vs random
and scan, DE/BPSO fixed five-seed mean vs random. Two-sided Bonferroni intervals
at .05/(2*8). Algorithm contrasts average metric differences, not ensemble scores.
Other comparisons descriptive. Positive claim requires corrected interval above0;
report recall/FPR tradeoffs. No exhaustive-position/global-optimum claims.

## Implementation/verification

Tests first: additive-kernel equivalence, fold-only centering, mask-to-layer mapping,
equal64 budgets/common34, padding/masked positions, fresh exclusion and seals.
Reuse existing extraction/metric helpers. Run full verification on compute host and
portable score archive; one independent whole-branch review before scoring.
