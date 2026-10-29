# R27 execution and completion ledger

Plan: docs/superpowers/plans/2026-10-29-training-stability.md. User authorized autonomous execution of proposed training stability study; existing independent experiment checkout retained, new branch experiment/training-stability.

- Task 1 complete: balanced 3200 pool, five deterministic draws of 1600; disjoint old dev/calibration/test and fresh confirmation. New inputs excluded 16122 historical IDs and near duplicates. Two initial isolation/control tests observed RED then GREEN; final four tests pass.
- Task 2 complete: extractor startup os-shadowing fault occurred before model loading; regression observed RED then GREEN. All 6400 rows extracted, 32 old training probes exactly match old B and summaries, protected prefix differences zero. Private cache 2726297856 bytes; no prior caches removed.
- Task 3 complete: 3600 fold fits, 40 final models, all ridge lambda1 and logistic C10 selected from train CV. Frozen at aa2113c1113b405302dd169fab71168d16926790 before scoring. No test-based selection.
- Task 4 complete: sole read-only branch reviewer found no blocking/important issues; warning retained about overlapping draws and non-ensemble averages. Long SSH stream disconnected while compute continued; checked existing PIDs/checkpoint before installing detached recovery manager. No data redraw or duplicate concurrent extractor/trainer. Server and local replayed 3600 fold metrics, 160 conditions and 5000 paired bootstrap draws with identical evaluation hash c6f6b8da9f6a8688c0431fa3d6aa287020e558d99cc0f6ce0e7138aef2bc25a1.
- Task 5: report, round record, progress, Markmap updated. Website build/browser checks and final Git delivery recorded in the task tool history.

Result: none of six prespecified fresh pAUC contrasts survived correction. Mean pool subsets are not independent replications; bootstrap intervals condition on these training draws and calibration. Dynamic-vs-random directions warrant retention as a future control, not replacement of the baseline. Multi-source prospective validation remains future work.
