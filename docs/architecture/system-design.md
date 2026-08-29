# System design

The serving SLO target should be set from product needs; a representative budget allocates 15 ms to the edge/API, 20 ms to state and features, 40 ms to candidate retrieval, 60 ms to SageMaker ranking, and 15 ms to rules/serialization. Timeouts fall back to cached popularity candidates with an explicit fallback model version.

Kinesis records use the versioned `InteractionEvent` contract. Producers retry with stable event IDs. Consumers conditionally claim IDs in DynamoDB, tolerate out-of-order delivery, watermark late data in batch, and write immutable date/hour partitions to S3. Poison events go to a dead-letter destination. At-least-once transport plus idempotent consumers gives effective exactly-once business handling.

Training uses event-time snapshots and temporal validation. A Step Functions state machine starts validation, feature processing, SageMaker Pipeline execution, and alarm notification; it does not poll or replace the streaming topology. Evaluation emits a signed metrics artifact. Registry packages contain code/image digest, feature schema, data interval, hyperparameters, source revision, and metrics.

Multi-AZ managed services, endpoint autoscaling, Kinesis capacity alarms, DynamoDB on-demand mode, API throttling, circuit breakers, and immutable artifacts define the failure strategy. Every response and event carries a request, model, schema, and experiment identity.

