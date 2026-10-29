# R23–R26: fixed full representation, matched detector-head comparison

User authorized three alternatives, autonomous verification and Git publication.
No base-model training, new attacks, paid APIs, or model downloads. Previous
opt-out of clarification remains in force. Use existing frozen features.

## Fixed representation and three alternatives

Full32-layer template-last raw decoder unit directions,4096 coordinates/layer,
concatenate and divide by sqrt32, fixed512 execution, no dimension truncation.
Train1600/dev600 and benchmarkv1 nonprobe2760 unchanged. Dev diagnostic only.

R23 is a fair tuned-ridge control, not a fourth alternative. R24 L2 logistic
regression changes the loss to binary classification. R25 standard linear SVM
(hinge loss, unpenalized intercept, precomputed linear kernel SVC) changes the
margin objective. R26 RBF kernel ridge introduces a nonlinear similarity.

Each family exactly9 candidates, train-only repeated stratified group5fold×2,
seeds20261115/16. Per-family winner maximizes mean standardized pAUC(.1), ties
resolve in listed order (stronger regularization first). Every family gets the
same folds and9-setting budget; runtime differs and is reported.

* Linear ridge lambda:[10000,1000,100,10,1,.1,.01,.001,.0001].
* L2 logistic C:[.0001,.001,.01,.1,1,10,100,1000,10000], lbfgs, max_iter5000,
  tol1e-9, no class reweighting. Use raw decision logits, not calibrated probability.
* Linear SVM same9 C values, SVC(kernel=precomputed), probability=False,
  tol1e-7, max_iter=-1; record support count. This is a linear decision function.
* RBF ridge lambda:[100,1,.01] × bandwidth factors:[.25,1,4];
  kernel exp(-factor*d_squared/median), median over fitting-fold pairwise squared
  distances excluding diagonal, floor1e-12. Thus9, not a wider hidden search.

Overall trainCV choice prefers tuned linear ridge within oneSE heuristic of best
family winner; overlapping-fold SE not a formal confidence interval. Publish four
family winners plus old lambda1 reference (five heads). Do not test-select a seed
or hyperparameter; all36×10 fold scores and OOF predictions archived.

## Exact computation, not feature selection

Precompute full raw Gram matrices from original coordinates in float64. Center
features/kernel using each fitting fold only. For logistic, reparameterize into
the complete fitting-sample span via eigendecomposition of centered Gram. Keep
all eigenvalues>max(max_eigenvalue,1)*1e-12, record numerical rank and relative
Gram reconstruction error<1e-9. This drops only numerical nullspace, not a64D
projection. Out-of-fold mapping uses fit-only eigenvectors/eigenvalues. Verify
against direct raw-coordinate logistic and linear SVM on a train-only pilot, and
synthetic ridge/SVM/logistic/kernel equivalence tests. All three linear winners
can be mapped back to full131072-coordinate coefficients for direct inference.
RBF winner stores dual coefficients plus references to sealed private train cache.

Reject convergence warnings before heldout evaluation; if max_iter needs increase,
record pre-scoring amendment applied uniformly, retain tolerance/objective/grid.
No per-coordinate scaling or fitted dimensionality reduction. Targets ridge±1,
logistic/SVM0/1, intercept unpenalized; each head has its own safe calibration.

## Fresh confirmation and bounded cache

Exclude all prior signal-selection/highdim/fusion/swarm/positions/routing/
processing records plus XSTest450 and author12. New Aegis train-partition seed
20261231,1100 candidates/class, exact plus char3–5 TFIDF cosine>=.92 exclusion,
<=512 tokens. Target500/class, count-only fallback floor min class count to25,
minimum200/class before inference. Same-source confirmation is not new-domain.

Reuse old4960 comparison rows' exact fixed template-last vectors. Extract only
fresh1000 full32×4096 float16 vectors (~.25GiB), cache cap.5GiB. Pilot32 old train
rows (16/class) must reproduce old vectors<=1e-5 and protected prefix exactly0.
Normalize full512 tensor before gathering, per R16; no previous shape shortcut.
Actual weights/tokenizer, cache bytes, IDs, source commit, environments and full
completion sealed. Never delete old caches. Resume requires exact metadata match.

## Freeze and evaluation

Commit manifest before inference, then commit complete cache, models, CV and
selection before evaluation. Check actual bytes against the frozen Git commit.
Independent review once before scoring, negative gate tests and unique analysis
module name to avoid the previous import collision.

Safe calibration400 threshold sorted[379], strict score>threshold; Aegis global
and Toxic source policies separate. Five test roles/seven conditions, five heads
=35 metric conditions. AUROC/AP/pAUC(.1)/TPR/FPR, Wilson and5000 stratified paired
bootstrap, seed20270109, conditional on trained heads and calibration.

Five prespecified fresh contrasts: tuned ridge−old lambda1, logistic−tuned ridge,
SVM−tuned ridge, RBF ridge−tuned ridge, overall CV choice−tuned ridge. Two-sided
Bonferroni quantiles .05/(2*5),1−.05/(2*5), per metric; AUROC and pAUC separate.
Other roles/comparisons descriptive. Keep original labels, no fitting to prior
false positives. Do not pool Toxic toxicity and Aegis harmfulness tasks.

Independently replay all360 fold metrics,36 settings, winner/oneSE rules,
IDs/folds,35 test conditions and bootstrap on host and portable score archive.
Update report, round ledger, progress and existing reproduction mind map/page.
