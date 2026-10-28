# Second improvement attempt: conditional late fusion and source calibration

> **For agentic workers:** Use superpowers:executing-plans inline, one fresh-context final reviewer. User explicitly authorizes autonomous iteration without questions. Keep every old branch/result.

**Goal:** Verify whether small cross-fitted expert fusion adds to a strong unprojected last-position detector, and independently whether safe-source calibration corrects transferred thresholds.

**Architecture:** Reuse exactly the prior1600 train/600 dev prompts and sealed features. Five-fold stratified OOF scores from four frozen-model experts train a low-dimensional meta detector. New disjoint calibration and prospective heldouts are extracted once using the prior validated instrumentation. All candidates are frozen before any new heldout/calibration scoring.

**Tech Stack:** Existing Python3.10/torch2.1.2/sklearn1.2.2/numpy, no new dependencies/models, no paid APIs, no base-model training, no attack generation.

## Design and alternatives

- Increasing PSO budget is rejected: prior full-vector last ridge is stronger and no stable multi-position increment was established.
- Early highdim concatenation repeats a failure mode. Keep it only as a frozen comparator.
- Recommended: four expert scores preserve a strong base and permit conditional complementary information through trained late fusion. Explicit pair products test a small interaction head; four experts permit exhaustive15 subsets, so PSO is unnecessary for this subproblem.
- Source calibration is a separate intervention on thresholds, with unchanged model/ranking; evaluate it only on the separate toxicity proxy scope, never merge toxicity with harmful-input labels.

## Frozen protocol before new model inference

- Branch experiment/conditional-fusion, parent a3fb895. Train1600 and dev600 are reused developmental data from highdim manifest; every prior4800/3700/450/12 prompt excluded from NEW evaluation/calibration except64 clearly marked probe_train rows for extraction consistency only.
- Four experts: B = unprojected last32x4096 ridge(lambda1); P = unprojected four-user-region32x4x4096 ridge(lambda1); A =8192 attention summaries L2 logistic(C=.01); T =512 trajectory+256 module summaries L2 logistic(C=.01). Scaling/centering fit within each training fold only. Same captured pre-final-norm unit vectors as prior round.
- Stratified5fold shuffle seed20261030, OOF scores for meta training. Full1600 refit experts used for deployment (standard stacking); OOF/full score-distribution shift is a limitation tested by independent calibration. Retain B-full as primary comparator and P/A/T experts as secondary. No resubstitution scores train meta.
- Train all15 nonempty additive score subsets, fixed meta logistic C=.1 balanced liblinear max_iter2000. One full four-score explicit pairwise-products head (4 main+6 products), same C; train-only generated-feature centering/scaling. Equal-weight standardized-score mean control. One best-dev additive subset designated before final tests; retain all15 regardless of test. No extra tuning after heldout inspection.
- Include frozen prior early raw-vector concatenation and prior F/C64 comparator (same exact training IDs) as pipeline references; imported serialized heads must match old frozen SHA. Comparison across estimators/representations does not isolate projection loss.
- Primary NEW prospective Aegis-train holdout1000 (500/class), separate NEW Aegis-train calibration800 (400/class), NEW official-Aegis-test up to200 (100/class), NEW ToxicChat1123-test up to100 (50/class), NEW ToxicChat1123-train safe-only calibration400. Smaller secondary tests may use min(availability,target) rounded down5, >=30/class, fixed from counts before extraction. Official test nearly exhausted; cannot invent sample counts. Primary remains disjoint prospective same-source, not official-test or unseen-source proof.
- New raw file is pinned ToxicChat train at29df8e4dba60e1f4af4b4075c0705c5b313548a8, human_annotation=true only, toxicity=0 for source calibration. Verify regular Git blob SHA1 dbc99e1b2648e53636e9c5b6d9f60f8964ab9682 and bytes7,134,932, retain SHA256/license cc-by-nc-4.0. WildJailbreak gate auto confirmed, no accepting/share-contact gate or bypass.
- NFKC exact + prior char3-5gram TFIDF>=.92 connected-group dedup; old prompts highest priority. Full template<=512, no truncation. Fast offsets must match original slow IDs. Preserve old data only as training/development or explicit probes, never as new final tests.
- Feature cache uses copied audited highdim extractor with only paths and probe-role substitutions. Retain all validated arrays, cache cap6GiB, fresh64 old-training probes for resource/F checks. Freeze actual array/model/dependency bytes before fitting/evaluation. Old files stay unchanged.
- All model masks/meta heads and code sealed/committed before calibration/heldout scores. Common Aegis safe-calibration400 sets strict score>95% order threshold. Toxic source alternative uses independent400 safe toxicity-source rows only. No positive target labels for thresholding, no test threshold optimization; actual FPR reported.
- Primary comparison full linear fusion vs B-full on fresh Aegis holdout. Pairwise vs additive fusion separates fixed interaction-head comparison; best-dev subset retained as exploratory selection. Report all candidates, AUROC/AP/TPR/FPR, paired group bootstrap1000, Wilson CIs; no multiplicity adjustment, conditional on frozen fitted pipeline. Source-calibration ranking must stay identical; target recall gain needs realized FPR and its uncertainty.
- OOF complementarity report is diagnostic: safe OOF95% thresholds, base FN rescued by other experts and base FP rejected. Oracle rescue counts are not achievable combined performance claims. New final sets cannot be reused as untouched tests later.

## Tasks and tests

- [ ] Acquire pinned extra data; failing tests for JSONL/role exclusivity/fold exclusion/interaction transforms; lock new manifest before extraction.
- [ ] Run copied extractor probe and bounded new features; verify old probe compatibility, model/cache bytes and source provenance.
- [ ] Generate five-fold OOF experts, full refits, diagnostic complementarity and15 additive/one pairwise heads; choose best-dev subset only.
- [ ] Commit frozen heads/dependency seals; score new calibration/holdouts once, bootstrap and independently recompute metrics and threshold policies.
- [ ] Fresh review, direction/method/results/conclusion report, progress/web sync, GitHub/server experimental branch publication.

## Review focus

Training-fold-only expert preprocessing; no same-row label in OOF expert fit; no test selection; explicit old-data/probe exceptions; source threshold calibration disjointness and toxicity scope; score-invariant calibration effect; strong-base and all-subset reporting; actual byte seals and mixed old/new cache indexing.

## Ledger

- 2026-10-28: user requests next feasible round through complete verification, without questions. This is the second improvement attempt after the highdim attempt; artifact folder conditional-fusion distinguishes it from earlier64-feature and highdim rounds.
