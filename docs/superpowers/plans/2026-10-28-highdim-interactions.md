# High-dimensional information and interaction pilot

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. User explicitly authorizes autonomous analysis, experiment and conclusion while away; no routine approval pause. One fresh-context whole-branch reviewer at end.

**Goal:** Test whether multi-position vectors, trajectories, attention and module summaries add information, whether an interaction-capable head helps, and whether grouped PSO adds value under matched budgets.

**Architecture:** One frozen-model pass caches four user-region pooled hidden vectors plus the last template-position vector, fixed random projections, trajectories, attention/module summaries and final logits. Train-only preprocessing, grouped selection and two fixed detector families precede a pre-test freeze. All old runs remain unchanged.

**Tech Stack:** Existing Python3.10, torch2.1.2, transformers4.37.2, numpy/sklearn; no new packages, models, paid APIs or base-model training.

**Spec:** User's 2026-10-28 high-dimensional-information request and autonomous-run authorization; local research design `readings/研究笔记/内部安全信号-高维分组与交互选择方案-2026-10-28.md` supplies rationale. This plan contains the executable pilot limits.

## Global constraints

- Branch experiment/highdim-interactions based on c83a1fc; clean in-place branch isolation avoids copying models/cache. Preserve main, reproduce/smoke and experiment/signal-selection.
- First new-data target: train1600, dev600, independent calibration800 from unused Aegis official train; final Aegis500 from unused official test; ToxicChat200 human-label toxicity proxy, separate task. Counts must be verified before extraction. Exclude every old4800/450/12 input, exact/near duplicates first; no new attacks.
- Full templated length <=512, no truncation. Verify fast offsets against exact original slow-tokenizer IDs before defining user masks. Four deterministic nonempty-as-available user-token regions; empty regions explicit.
- Cache ceiling6GiB. Pool normalized raw decoder-layer outputs, keep raw norm summaries separately; all32 layers use the same pre-final-norm boundary. Preserve author F's final repeated norm separately and verify8 training samples.
- Fixed Rademacher projection4096->64 shared across groups, seed20261029, no label fitting. Group identity layer x last/four user regions; keep full pooled4096 vectors for future channel analysis, do not call projected coordinates individual neurons.
- Trajectory16 scalars/layer, attention8 scalars/head/layer, module8 scalars/layer, F32, length/missing indicators. Training-only centroid C32 added later as a comparator. Attention preflight64 training rows: require no OOM, peak<22GiB, median forward-time ratio<=2 against no-attention mode; otherwise omit attention for all full-run rows and record ruling before any dev scoring.
- Two heads: L2 logistic(C=.01, balanced, liblinear, max_iter2000); ExtraTrees32 trees, max_depth3, min_samples_leaf20, max_features=sqrt, balanced, n_jobs1, random_state0. Same head/configuration across full/random/PSO comparisons. First-logit C=.01 adaptation independent strong baseline.
- Compare last-only projected vectors, user-region-only vectors, their union, plus trajectories, plus optional attention/modules, F/C64, lengths-only, L1(C=.01), and linear/nonlinear all-feature controls. This is a projected high-dimensional group-selection pilot, not exhaustive neuron selection or original RCS/GraphShield replication.
- PSO/random five seeds0..4, <=100 distinct masks each run/head; BPSO10 particles x10 rounds. Fitness dev AUROC-.02*(selected scalar dimensions/total dimensions); no empty masks. Same pool/head/budget; record actual queries and incremental shared-cache timings. No post-test budget expansion.
- Freeze all candidates/heads/masks/hashes before calibration/test score computation; every seed retained. Threshold from safe calibration only, strict>; actual test FPR/TPR plus AUROC/AP, paired label-stratified group bootstrap500 and Wilson intervals. Old tests only exclusions or diagnostics, not untouched evaluation.
- Identify harmful-input labels versus toxicity versus jailbreak success; do not infer causal attention contributions, novelty, robustness or backbone speedup from this pilot. Raw cache is private; publish only IDs/hashes/scores/heads and sanitized logs.

## Review focus

1. Full tokenization alignment and causal/padding/empty-bin masks.
2. Consistent layer boundaries and author F equality, finite values and complete-prefix recovery with provenance.
3. Old-data/duplicate/official-source-role isolation and training-only feature fitting.
4. Interaction head capacity and matched mask-budget accounting; no test selection.
5. Fresh counts, archive byte hashes and calibration/uncertainty claims.

## Tasks

- [ ] TDD pool/mask/cache and acquisition/split contracts; create fresh locked manifest before extraction.
- [ ] Implement streaming features and64-train-row attention/resource pilot; freeze attention inclusion ruling.
- [ ] Extract bounded full cache, verify author8-row F and input/provenance integrity.
- [ ] TDD group indexing/head scoring; run information ablations, interaction controls and matched five-seed searches.
- [ ] Commit pre-test freeze; evaluate all candidates once, independently recompute public point metrics/hashes.
- [ ] Fresh whole-branch review, results analysis including negative findings, progress/web updates and server/GitHub branch synchronization.

## Ledger

- 2026-10-28: user opts out of routine questions and explicitly requests one autonomous analysis/experiment/conclusion round. Native execution selected; research design narrowed to a bounded pilot to separate information, interaction and optimizer contributions.
- 2026-10-28: training-label budget kept1600 and dev600 to match first pilot; calibration moves to unused train because old official validation has too few untouched safe examples. This is a changed calibration protocol, not a paired numeric claim against the old450/4800 results.
- 2026-10-28: before any dev scoring, add three unprojected4096-vector ridge controls (last/user/union), fixedlambda1 and training centering, normalized by sqrt(group count). They isolate position information without projection; compare within this ridge family, not as same-head comparisons against logistic/trees.
- 2026-10-28: resource pilot64 passed: median paired attention-forward ratio1.146 and max allocated15,251,129,344bytes. Attention included. JSONL Unicode separator issue fixed without changing frozen record bytes or split.
- 2026-10-28: review requires actual-byte seals for every cache array and rechecking actual locked model/tokenizer files before training. Freeze seals and verify before both fitting and final scoring. Add objective/new-fit/cache-hit/fit-seconds accounting per search run; timings remain incremental shared-cache, not standalone speed claims.
- 2026-10-28: serial training ended after six complete100-mask runs, before freeze or any heldout scoring; log has no exit reason and no running process remained. Local tool approval was separately blocked by usage limits; do not infer that caused the server exit. Preserve completed runs, resume with8 independent fit workers on16 available CPU cores, estimator threads remain1. Exact serial/batched PSO equivalence and concurrent estimator prediction checks pass. Head parameters, proposals, fitness, seeds and budgets unchanged; no post-test adaptation. Resumed timing is incremental with mixed serial/parallel phases and is not a standalone speed benchmark.
