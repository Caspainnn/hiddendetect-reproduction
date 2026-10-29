# Population-selection review before scoring

2026-10-29. Fresh whole-branch review found no additional material leakage, fair-budget or metric issue after inspecting initialization, search operators, train/dev separation, path-only extractor changes and vectorized bootstrap tests.

Two review findings were fixed before any benchmark or confirmation scoring: standalone probe imports and enforcement of a successful compatibility report bound to old/new cache seals. Regression checks were observed failing and then passing. An actual early run also exposed Linux's filename-component limit for259-bit mask names; SHA256 cache names fixed this, with a bounded/distinct-name regression check. The first attempt stopped while saving the initial candidate, before formal searches. Search restarted with unchanged mathematical model, proposal rules and budgets.

Tests cover common initial masks, exact unique budgets, duplicate handling, reproducibility, monotone best history, GA/DE operator invariants, vectorized dedup equivalence to the legacy union implementation, and vectorized AUROC/AP/pAUC bootstrap equality to sklearn including ties. No deferred minor findings. Runtime results are recorded separately after actual completion; this review is not itself a claim of algorithm effectiveness.
