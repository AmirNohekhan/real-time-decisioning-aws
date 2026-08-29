# ADR 0001: Multi-stage retrieval and ranking

Accepted. Separate retrieval from ranking so large catalogs can be reduced cheaply and rich contextual inference is limited to tens or hundreds of candidates. Use popularity plus collaborative SVD locally, with a hybrid union. Popularity handles cold start; SVD adds personalization. A tree ranker handles nonlinear mixed features and is operationally simpler than a neural ranker at this data scale. Native ANN and pairwise ranking remain compatible replacements.

# ADR 0002: Feature Store and event-time correctness

Accepted. Use a local interface for reproducibility and SageMaker Feature Store for managed online/offline parity in AWS. S3 remains the immutable source. Event-time feature views exclude the current/future event. This costs additional storage and operational discipline but makes freshness, lineage, and training-serving skew measurable.

# ADR 0003: Kinesis, DynamoDB, and orchestration

Accepted. Kinesis carries high-throughput behavior; S3 retains it. DynamoDB only holds conditional idempotency keys, stable assignments, and recent state needing single-digit-millisecond access. Step Functions coordinates bounded batch transitions and approval gates, not individual stream events.

# ADR 0004: Lambda versus SageMaker

Accepted. Lambda owns validation, orchestration, policy, and response shaping. SageMaker owns model inference. This prevents oversized Lambda packages and makes model scaling/versioning independent, at the cost of a network hop managed by strict latency budgets and fallback behavior.

# ADR 0005: Bounded exploration

Accepted. Epsilon-greedy is transparent, testable, and sufficient to demonstrate feedback acquisition. It runs only for assigned treatment traffic and logs propensities. Its opportunity cost and weak uncertainty model mean Thompson sampling or contextual bandits may be preferable after safe offline evaluation.

