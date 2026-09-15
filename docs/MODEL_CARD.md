# Model card

## Intended use

Portfolio demonstration and prototyping for support-case routing, urgency estimation, evidence
retrieval, selective response, and operational analytics.

## Not intended for

Automatic customer communication, safety-critical incident handling, employee evaluation, or any
production deployment without representative data and human oversight.

## Components

- Word and bigram TF-IDF features.
- Balanced logistic-regression classifiers for route and priority.
- Cosine-similarity TF-IDF retrieval over authored runbooks.
- Extractive response composition from the top article.
- Evidence-score threshold with an abstention response.

## Evaluation

The benchmark holds out one complete issue family per route to reduce template leakage. Metrics
include accuracy, macro-F1, per-route precision/recall, confusion matrices, hit@1, hit@3, MRR,
coverage, selective accuracy, out-of-domain abstention, and retrieval latency.

## Limitations

- Synthetic tickets are smaller, cleaner, and less diverse than real conversations.
- The corpus has 15 English articles and no access-control model.
- The retrieval benchmark uses one expected article per ticket; real cases may have several valid sources.
- Lexical similarity is vulnerable to vocabulary shifts and keyword overlap.
- Probability outputs have not been calibrated on real traffic.
- Priority labels encode simple impact phrases rather than operational SLA history.
- Logged metadata still requires a retention and access policy in real use.

## Responsible operation

Keep citations visible, preserve abstention, monitor route-specific errors, sample automated cases for
review, and never treat synthetic benchmark quality as evidence of production readiness.

