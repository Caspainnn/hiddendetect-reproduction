# R19–R22: processing user information, preregistration

User approved the processing direction; previous opt-out of clarification and
authorization to execute/publish experiments remains in force.

## Hypotheses and finite family

R19 ordinary user mean is a control. R20 dynamic attention/change weighting tests
whether local details survive better than averaging. R21 adds residual and module
increments. R22 adds local/global relations. No base-model training or generation.

Use raw decoder outputs at four fixed anchors (layers8,16,24,32, one-based).
For each user token, unit h; delta=raw layer output−raw layer input. Seven full4096
summaries per anchor, stored float16:
0 ordinary mean(unit h);
1 mean-head attention from last valid query, restricted/renormalized to user,
  weighted mean(unit h), uniform fallback when attention mass is zero;
2 change weights=softmax(clip(zscore(norm(delta)/max(norm(input),eps)),−5,5)),
  weighted mean(unit h), population SD and uniform fallback for equal changes;
3 same change weights applied to unit(delta);
4 same change weights applied to unit(attention module output);
5 same change weights applied to unit(MLP output);
6 deterministic ID-seeded random logits, same clip/zscore/softmax recipe, weighted
  mean(unit h). Random weights are a negative control, not fitted or selected.

All token computations exclude prefix/suffix/padding. Fixed512 right padding and
mask, same valid IDs as earlier. Extract all32 template-last states only to check
old baseline equality; save baseline only for fresh samples. Bound new cache2GiB.
Never delete previous sealed caches. Public manifests have hashes/IDs, no text.

Relations: each summary's cosine to matching-layer global template-last, distance
and pooled norm (4×7×3=84); per-anchor attention entropy, change-weight entropy,
random entropy, relative-change mean/std/max, attention-vs-change weight cosine,
attention-vs-MLP pooled cosine (4×8=32), total116 scalars. They are mathematical
relations, not causal attribution. Constants and weights are label-independent.

## Classifier, equal controls and selection

Baseline B: same fixed512 full32×4096 template-last, float64 lambda1 centered ridge
as R16. Preserve original legacy B separately by point-score replay if available.
Joint additive kernel ridge G=G_B+gamma G_E, gamma in[.25,1], lambda1 throughout:
mean=[0], dynamic=[1,2], processed=[1,2,3,4,5], relations=processed plus .1 times
116 standardized relation-coordinate Gram; random=[6]; duplicate_mean=[0]*5.
Vector families normalized by sqrt(4×number of summaries), no train-fitted
projection. Thus duplicate_mean and mean are mathematically identical despite
different nominal dimensions. Relations standardized on each fitting fold only,
constant coordinates std=1; divide by sqrt116. For final fitting use train only.

13 frozen candidates: B plus six families×two gamma values. These explicit ridge
models learn combinations; no PSO in this round. No hyperparameter/test seed search.
Repeated stratified group5fold×2, seeds20261115/16, train1600 only, unchanged dev600.
Select by mean pAUC(.1), prefer B within one-SE heuristic; otherwise best eligible
real family among mean/dynamic/processed/relations. Overlapping-fold SE is not a
formal interval. Publish all candidates, dev diagnostic only. No test selection.

## Data and gates

Immutable benchmarkv1 nonprobe2760 unchanged; same train1600/dev600. Exclude all
earlier signal-selection/highdim/fusion/swarm/positions/routing records, XSTest450
and author12 calibration from fresh selection. Aegis same-source train partition,
new seed20261219, cap1100/class, tokens<=512, target500/class. Same exact and .92
char3–5 TFIDF near-dedup. If counts short, floor minimum class count to25, minimum
200/class, count-only amendment before inference. Confirmatory claims remain
same-source; newly scored confirmation cannot be reused as untouched in future.

Commit manifest before inference. Hash actual model/tokenizer, records, all new
cache bytes plus original fixed cache and seal. Verify source commit/shape/dtype/
completed count and old global-state equality. Commit heads/code/CV/selection
before benchmark/new confirmation scoring; record pre-evaluation commit hash.

## Evaluation and confirmation family

Global Aegis safe400 and source Toxic non-toxic400 thresholds sorted[379], strict
score>threshold; no test adjustment. Seven conditions as before; AUROC/AP/pAUC(.1)/
TPR/FPR, Wilson and5000 stratified paired bootstrap, seed20261229, conditional on
fixed fits/calibration, not full training uncertainty.

Ten prespecified contrasts at gamma1 on fresh confirmation:
mean−B, dynamic−B, dynamic−mean, dynamic−random, processed−B,
processed−dynamic, relations−B, relations−processed, selected−B,
duplicate_mean−mean. Two-sided Bonferroni .05/(2×10), per metric; AUROC and pAUC
interpreted separately, no joint-family significance claim. Gamma.25 and v1
comparisons descriptive. Report all seeds/families even when negative.

## Verification and execution

Tests RED→GREEN: weights/masked token support, zero/constant cases, module/delta
summary mapping, relations dimensions, duplicate-mean kernel equivalence,
fold-only centering/scaling, additive dual vs explicit primal predictions, source
and incomplete-cache rejection. One fresh-context whole-branch review before
scoring. Independently replay all CV/selection/metrics/bootstrap on host and local
score archive. Report processing gains only against matched strong B and controls.
