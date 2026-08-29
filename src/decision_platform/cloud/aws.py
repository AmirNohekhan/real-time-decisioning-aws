import json
from typing import Any

import boto3

from decision_platform.contracts import InteractionEvent


class KinesisEventPublisher:
    def __init__(self, stream_name: str, region: str = "us-east-1", client: Any = None) -> None:
        self.stream_name = stream_name
        self.client = client or boto3.client("kinesis", region_name=region)

    def put(self, event: InteractionEvent) -> bool:
        self.client.put_record(
            StreamName=self.stream_name,
            PartitionKey=event.user_id,
            Data=event.model_dump_json().encode(),
        )
        return True


class SageMakerRanker:
    def __init__(self, endpoint: str, region: str = "us-east-1", client: Any = None) -> None:
        self.endpoint = endpoint
        self.client = client or boto3.client("sagemaker-runtime", region_name=region)

    def predict(self, rows: list[dict[str, float]]) -> list[float]:
        response = self.client.invoke_endpoint(
            EndpointName=self.endpoint,
            ContentType="application/json",
            Body=json.dumps({"instances": rows}),
        )
        return list(json.loads(response["Body"].read())["predictions"])


class DynamoIdempotencyStore:
    """Conditional writes provide durable event/request deduplication with TTL."""

    def __init__(self, table_name: str, region: str = "us-east-1", resource: Any = None) -> None:
        self.table = (resource or boto3.resource("dynamodb", region_name=region)).Table(table_name)

    def claim(self, key: str, expires_at: int) -> bool:
        try:
            self.table.put_item(
                Item={"pk": key, "expires_at": expires_at},
                ConditionExpression="attribute_not_exists(pk)",
            )
            return True
        except self.table.meta.client.exceptions.ConditionalCheckFailedException:
            return False
