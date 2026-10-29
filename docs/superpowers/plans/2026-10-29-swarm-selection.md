# Unified benchmark and population feature selection

> Execute inline with superpowers:executing-plans and one fresh-context final reviewer. User authorizes autonomous experiments without questions and GitHub experimental branch publication. Preserve all prior branches/results.

**Goal:** Test GA and threshold-adapted DE against PSO/random on the same strong-base anchored feature-selection problem, prioritizing low-FPR ranking, with a fixed public benchmark and a fresh confirmation if count audit permits.

**Architecture:** Reuse sealed highdim1600 train/600 dev, previous OOF B scores for training and full B for deployment. Always include B score and select any subset of259/19270 train-standardized feature groups. Fixed logistic C=.01 balanced liblinear; selection objective standardized dev partialAUROC(max_fpr=.1) minus .01 retained-coordinate fraction. Same five seeds0..4, same12 initial masks per seed (none/all plus mixed densities),120 unique masks per run for random/BPSO/GA/DE. All20 winners retained; no final-test selection.

**Tech:** Existing numpy/sklearn/torch only; no new packages, base-model training, attacks or paid APIs. Existing caches remain immutable. Branch experiment/swarm-selection.

## Design and fixed choices

- B-only zero-extra mask is valid; B coefficient is learned, not fixed. OOF/full expert-score shift remains standard-stacking limitation. Features/centroids/scalers exactly those fit on highdim training; do not fit any preprocessing on benchmark or confirmation.
- GA: tournament3, uniform crossover probability.9, independent bit mutation1/d, elitist population12.
- DE: DE/rand/1/bin F=.5,CR=.9 on latent[0,1], clip bounds, deterministic threshold>=.5 to masks; >=one donor coordinate. This is an explicitly described feature-selection adaptation, not a canonical binary-DE reproduction.
- BPSO: inertia.729,cognitive/social1.494, sigmoid positions, velocity clip[-6,6]. Identical initial masks to other methods. Duplicates cached; all methods receive exactly120 unique objective evaluations. After bounded duplicate stagnation, common fresh mixed-density proposals replenish diversity; log proposals/cache hits/restarts/time. Equal unique fits does not imply equal wall time.
- Random uses the same initialization and density mixture. Share fits across runs only as an execution cache; still charge each run its own120 unique masks. Thread pool4, underlying BLAS1 to avoid oversubscription. Save all masks, fitness curves, seed winners, sparsity, development AUROC/pAUC and times.
- Fixed benchmark: all non-probe roles from fusion manifest (Aegis prospective1000, official120, Toxic prospective400, official40, calibrationAegis800 / Toxic safe400). Already inspected historical benchmark, explicitly not untouched final evidence. No algorithm sees benchmark scores during search. Global/source thresholds exactly previous strict empirical95% order, source policy only separate toxicity proxy.
- Comparators: frozen original F integral, supervisedF32/F-C64, projected-last, raw last/positions, highdim all linear/tree/L1, old highdimPSO five seeds and random five seeds, existing full fusion/pairwise, anchoredB-only/all/L1. Same benchmark and same highdim training1600 where supervised; original rule uses fewer labels. Old searches differ in objective/budget and are historical controls, not fair algorithm-comparison arms.
- Fresh confirmation: count-only select200/class unused Aegis official-train prompts excluding all signal/highdim/fusion inputs with prior exact/char3-5 TFIDF>=.92 grouping; minimum100/class if availability requires count amendment before new inference. Append64 old-training probes only for feature consistency. No confirmation role enters development/calibration. No new source/unseen-domain claim. Feature extractor same audited implementation, only paths/role substitutions. Seal and compare probes before inference scoring; freeze all winners/code before confirmation scoring.
- Metrics: AUROC, standardizedpAUC(.1),AP,TPR/FPR, Wilson/1000 paired stratified bootstrap conditional on fixed fit/calibration. Primary algorithm comparison mean over5 seeds GA/DE minus matched random/PSO on main fixed benchmark pAUC; also individual seed results and parameter-selection stability. No multiplicity-adjusted superiority claim; five seeds pilot, not optimizer-family theorem. Separate public benchmark from fresh confirmation results. No retuning after either.
- Source bases: Storn/Price1997 https://groups.csail.mit.edu/EVO-DesignOpt/pb/uploads/Site/differentialEv.pdf; pymoo author documentation https://pymoo.org/customization/binary.html and https://pymoo.org/operators/crossover.html. Implement explicit minimal numpy variants, not pymoo package/reproduction.

## Tasks

- [x] Tests RED/GREEN: exact unique-budget, common initialization, deterministic masks, monotonically best fitness, GA/DE operator invariants, duplicate replenishment, partial-AUC and train-only transforms. Record immutable benchmark contract and source protocol before search.
- [x] Implement/train searches and fixed controls; preserve all20 winners and source/head hashes. Build count-only fresh confirmation manifest; resource pilot/extract/seal/probe compatibility. Commit freeze before scoring.
- [x] One fresh review; score same benchmark and untouched confirmation, compute paired distributions and independently verify all point metrics/roles/thresholds/budgets. No final-test tuning.
- [x] Report direction/method/results/conclusion, update R11 low-FPR anchored objective /R12GA/R13DE method boundaries and fixed benchmark policy, sync progress/page/GitHub/server.

## Ledger

- 2026-10-29: completed20 searches,120 unique masks each,2181 globally distinct L2 fits plus oneL1;1401.05s. All48 candidates committed3f8b721 before scoring. Unified benchmark and fresh400 confirmation scored84.01s;336 conditions/3160 rows independently verified on server and Windows, B bootstrap replayed via sklearn. Old8 heads ×2760 rows exactly reproduce prior scores. GA mean benchmarkpAUC .78590 vsB .79499; freshGA .74524 vsB .75079 with CI including0. DE/PSO/random return zero-extra; L1 onlyB. No confirmation-based reselection or tuning. Supplemental joint calibration uncertainty reported separately; no reset credit used. Full report updates prior fusion evidence honestly.

- 2026-10-29: before any new benchmark or confirmation scoring, add supplemental1000 joint safe-calibration/test bootstraps (seed20261102), shared resample indices across methods, to assess threshold uncertainty. This does not change fixed thresholds, primary objective, candidates or algorithm budget. Report separately from primary conditional intervals. Scalar-loop equivalence regression observed RED/GREEN.

- 2026-10-29: local and remote RED/GREEN regression checks passed. Count-only confirmation selected200/class plus64 old-training probes (464 rows), excluding11722 prior unique IDs. Full extraction/cache/model seals passed; all eight probe arrays have zero maximum difference. Confirmation has not been scored.
- 2026-10-29: initial training stopped saving its first model because259-bit filenames exceed Linux component limit. Fixed to SHA256 names and restarted unchanged search. Fresh review found probe import bootstrap and missing probe-success seal gate; both fixed with observed failing/passing checks before any scoring. No deferred findings.

- 2026-10-29: user explicitly notes PSO failure does not rule out other population methods; asks new optimization/validation on unified benchmark. Select GA+DE as two distinct mechanisms, retaining PSO/random and simple sparse controls. Existing observed benchmark is comparative only; fresh confirmation sought separately.
