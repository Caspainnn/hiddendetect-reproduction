# Signal Selection Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether refusal and two-class representation features complement each other, and whether BPSO selects useful sparse groups better than simple equal-budget alternatives.

**Architecture:** Freeze the existing LLaVA-v1.6-Vicuna-7B/float16 stack. Lock independent prompt-harmfulness training/search/calibration/test manifests before extraction; cache 32 last-position hidden vectors, 32 refusal signals and final next-token logits once. Fit references and heads on train only, select masks on development only, calibrate thresholds on calibration only, evaluate sealed test manifests after all candidates are frozen.

**Tech Stack:** Existing Python3.10, torch2.1.2, transformers4.37.2, numpy1.26.4, sklearn1.2.2. No additional optimization library, API judge, attack generation, base training or model download.

**Spec:** User-approved proposal in this conversation; protocol.md plus the additional prospective rules below. Branch experiment/signal-selection, based on 9a74f8e. Existing source and baseline results immutable.

## Global Constraints

- Text-only adapted comparison on one frozen model; not faithful whole-paper reproductions of First to Know or RCS, not a new multimodal benchmark.
- Existing XSTest450 and original12 are development diagnostics/exclusions, never fresh final tests.
- WildGuardMix is gated: no alternate mirror to bypass access conditions. Aegis2.0 is the primary prompt-harmfulness task; JBB is held-out external prompt-harmfulness; ToxicChat human-annotated toxicity is a separate task/proxy, never pooled as equivalent ground truth.
- Data sources/revisions, licenses, raw SHA256, omissions, long-input exclusions, normalized and near-duplicate filtering recorded. No raw user text, raw data, caches, weights or secrets in public Git.
- Target counts, subject to pre-score eligibility: Aegis train800/class, dev300/class, calibration500/class, official test500/class; JBB up to100/class, ToxicChat1123 official test200/class with human_annotation=true. No generation, truncate or reword prompts. Full templated inputs <=512 tokens; report short-prompt scope.
- Reject normalized duplicates across roles and the old450/12. Near-duplicate char-ngram cosine >=0.92 removed from later roles; this does not guarantee semantic independence. Lock final IDs and hashes before any model score; held-out source test files not read by optimizers.
- Features: F32; C32 = normalized hidden cosine to unsafe training centroid minus safe training centroid. C is RCS-inspired simplification, not original RCS. First-to-Know-style full final-logit logistic probe is an independent strong adapted baseline. Original RCS learned projection/KCD remains a named uncompleted stronger baseline, required before novelty claims.
- All logistic heads use training-only standardization, C=1, class_weight=balanced, deterministic liblinear, max_iter=1000. Score orientation unsafe=1. No tuning from final tests.
- Compare fixed original F integrals[16,29]/[17,29], F32/C32/all64 logistic, first-logit probe, 496 contiguous intervals (>=2 layers), L1 sparse head, forward greedy, random masks and BPSO masks. Interval enumeration is a fixed-score comparator, not a mask search.
- BPSO and random: at most500 distinct candidate masks per run, seeds[0,1,2,3,4]; BPSO20 particles x25 proposal rounds, duplicate proposals cached, empty masks rejected. Greedy capped at500 distinct evaluations; L1 fixed C grid[0.01,0.1,1,10]. Report actual evaluations/time, not only nominal iterations.
- Selection objective = dev AUROC -0.02*(selected_groups/64). First pilot uses one training source, so no invented cross-source variance penalty. Report dev TPR at descriptive ROC FPR5% as secondary; deployed thresholds calibrated separately from safe calibration scores (strict score>threshold, conservative order statistic).
- Freeze all masks/head choices and their hashes before test scoring. Evaluate every predeclared seed, not just the best test seed. Report realized FPR/TPR, AUROC, AP, group count, paired group bootstrap CIs and source-specific metrics. No post-test reselection.
- Success targets to assess, not guarantee: TPR +>=3pp at comparable realized FPR, or <=32/64 groups with TPR loss<=1pp, relative to full64. BPSO must also beat equal-budget random/L1 consistently; otherwise retain simple method and report negative evidence. Low test normal count => wide intervals; no reliable1%FPR claims.
- Conditional economy: if fusion fails to improve dev discrimination, still allow the bounded sparse search for compression, but do not claim detection quality gain. No adaptive expansion of search budget after test inspection.
- Progress source remains progress.md; rebuild existing webpage at each milestone. User permits execution without clarification. Quota reset cards may be used only through a supported operation when needed; ordinary usage currently permitted.

## Review Focus

1. Harmfulness, toxicity and refusal labels must not be conflated.
2. References/scalers/heads may not see dev/calibration/test labels during training; masks may not see test metrics.
3. Existing450/12, duplicate prompts and official split boundaries may not leak into final tests.
4. Cached F must reproduce the author's norm/head/cosine exactly, including final repeated norm.
5. Search evaluation accounting, probability boundary/empty masks, threshold ties and partial cache recovery must be correct.

## Tasks

### Task 1: prospective data and split contract
- [x] Write failing tests for label mapping, split exclusivity, duplicate priority, immutable manifests, threshold ties.
- [x] Implement download/split builder; acquire only ungated pinned public data; collect counts and licenses.
- [x] Verify class counts, hashes and no exact/near overlap; lock manifests and commit before extraction.

### Task 2: one-pass immutable feature cache
- [x] Reuse locked loader/template and add failure-first cache/row-shape checks.
- [x] Extract last-position H32, author F32 and final logits; save input token IDs hash/length and elapsed/peak resources, support resume with complete-row markers.
- [x] Verify author score equality and all cache indices/manifest hashes before supervised fitting.

### Task 3: heads and feature selection
- [x] Write failing tests for training-only reference fitting, objective/mask budget, deterministic random/BPSO, threshold calibration and metric orientation.
- [x] Implement simple heads and bounded selection; save pre-test freeze manifest including all masks and serialized head hash.
- [x] Run train/dev experiments and all declared seeds; retain negative findings and failures.

### Task 4: sealed evaluation and delivery
- [x] Evaluate pre-frozen candidates once; paired bootstrap by prompt group, report sample uncertainty and proxy label differences.
- [x] Independent fresh-context review of leakage/search accounting and artifacts; fix material implementation errors without selecting on test outcomes.
- [ ] Update progress/results/README and webpage, preserve experiment branch, synchronize local/server; user already authorized GitHub project publication, push this experimental branch with no merge into baseline main.

## Ledger

- 2026-10-28: isolated working branch chosen over an extra worktree because the tree is clean and there is no concurrent implementation; no duplicate model/cache copy.
- 2026-10-28: WildGuardMix access gate confirmed. Chose ungated primary and external sources without asking user or bypassing conditions. ToxicChat metrics remain separate.
- 2026-10-28: changed speculative cross-source-fitness penalty to a fixed sparse dev-AUROC objective for this single-training-source pilot. Cross-source claims depend on untouched external tests, not an invented training-source partition.
- 2026-10-28: original RCS full projection is explicitly pending, not renamed into a centroid baseline. This run tests information/selection feasibility and cannot alone establish superiority to RCS or novelty.

- 2026-10-28: manifest locked at472d8a9, extraction4800 rows in232.05s; author8-row F difference0; uninterrupted run resumed_at=0.
- 2026-10-28: all19 candidates frozen at3814579 before heldout evaluation. All five BPSO and five random runs each evaluated500 distinct masks; greedy500 and intervals496. No budget expansion after test inspection.
- 2026-10-28: independent review found partial-cache extractor/runtime identity missing. Added provenance with a failing-then-passing test; original complete cache remains attributable to recorded old extractor SHA and replay commitb38ba84. New extractor rejects old metadata rather than relabeling it.
- 2026-10-28: heldout evaluation complete; conditional paired bootstrap500 and finite-sample Wilson intervals retained. All point metrics/hash checks independently recomputed. Review found no unresolved material bug. Shared-cache search times are incremental, not standalone method benchmarks.
- 2026-10-28: fusion improved Aegis discrimination over supervised F32, but not JBB; BPSO did not stably outperform equal-budget random and external recall fell. Kept every seed and negative result. No originalRCS superiority, novelty or model-forward speedup claimed.
