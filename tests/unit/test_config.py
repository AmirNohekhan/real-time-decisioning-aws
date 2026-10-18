from pathlib import Path

from decision_platform.config import Settings, get_settings


def test_defaults_are_local_and_deterministic(monkeypatch) -> None:
    for name in ("DP_ENV", "DP_EPSILON", "DP_RANDOM_SEED", "DP_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=None)
    assert settings.env == "local"
    assert settings.random_seed == 42
    assert 0 <= settings.epsilon <= 1
    assert settings.data_dir == Path("data/generated")


def test_environment_variables_override_defaults(monkeypatch) -> None:
    monkeypatch.setenv("DP_EPSILON", "0.2")
    monkeypatch.setenv("DP_KINESIS_STREAM", "events")
    settings = Settings(_env_file=None)
    assert settings.epsilon == 0.2
    assert settings.kinesis_stream == "events"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
