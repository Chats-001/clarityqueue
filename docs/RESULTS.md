# Benchmark results

## Evaluation design

The seeded generator produced 360 tickets across five routes and 15 issue families. One complete
issue family from each route was held out, leaving 120 evaluation tickets whose underlying pattern
did not occur in classifier training. The evidence index includes all 15 runbooks because retrieval
systems are expected to search the knowledge they serve.

## Findings

- Route accuracy was **85.0%** and macro-F1 was **85.2%**.
- Billing and integration generalized best; access and data-sync recall were lower on unseen issues.
- Priority macro-F1 was **72.8%**. Critical and high priority recall were both **100%**, while the
  intentionally under-specified low/medium cases were frequently confused.
- The expected runbook ranked first for all 120 in-domain held-out tickets. This reflects the small,
  controlled knowledge base and should not be generalized to a production corpus.
- Retrieval p95 latency was below 1 ms in the local benchmark.
- **74.2%** of cases crossed the route-confidence and evidence gates for the automated path. Both
  route and evidence were correct for that selected subset.
- The evidence threshold rejected 7 of 8 unrelated questions (**87.5%** OOD abstention). The one
  failure shows why a lexical score alone is not a complete out-of-domain detector.

## Recommendation

Use ClarityQueue as a decision-support queue, not an autonomous resolution system. Start with the
high-confidence subset, show the proposed route and cited runbook to an agent, and collect corrections.
The most valuable next experiment is threshold selection on real, time-separated tickets with a cost
for misrouting and a separate cost for human review.

All values above are generated from `artifacts/metrics.json` by `python -m scripts.train`.

