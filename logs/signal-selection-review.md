# Fresh-context review, 2026-10-28

Reviewer: independent review_signal_selection agent; read-only branch review.

No unresolved material correctness bug found. Reviewer confirmed training-only references/scalers/heads, development-only selection, separated official calibration/test roles and old-prompt exclusions, author F norm/head semantics and exact smoke evidence, 500 unique nonempty masks for each of five random and five BPSO runs, greedy500, all19 candidates frozen and retained, training-code hash and test class counts.

Issue fixed after failure-first cache test: resume identity now also locks extractor/runtime/artifact manifest/LLaVA revision, preventing silent mixed-version partial caches. This original extraction resumed_at=0 and retains its original SHA and pinned replay version; no existing cache provenance was rewritten.

Nonblocking limits retained in report: shared fitted-head caching makes method timings incremental and order dependent; conditional bootstrap excludes retraining/search/calibration uncertainty; character dedup does not establish semantic independence; original RCS remains pending; no novelty or model-forward acceleration established.

Reviewer did not edit files, run GPU work, access SSH or credentials. Public result verification additionally recomputes hashes and every heldout point metric from saved scores.

Post-review delivery check caught Git newline normalization changing the acquisition-log bytes referenced by the frozen manifest. A specific -text attribute now preserves those original bytes; the verifier checks this outer hash too. Revalidation preserves the existing acquisition log and rejects changed file declarations, covered by an additional failing-then-passing lightweight test. No frozen candidate, threshold or test score changed.
