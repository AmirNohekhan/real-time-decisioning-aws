from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DP_", env_file=".env", extra="ignore")

    env: str = "local"
    data_dir: Path = Path("data/generated")
    artifact_dir: Path = Path("artifacts")
    log_level: str = "INFO"
    model_version: str = "local-untrained"
    epsilon: float = 0.05
    random_seed: int = 42
    aws_region: str = "us-east-1"
    sagemaker_endpoint: str = ""
    kinesis_stream: str = ""
    dynamodb_table: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
