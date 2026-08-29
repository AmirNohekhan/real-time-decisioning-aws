import argparse
import json
import shutil

from decision_platform.config import get_settings
from decision_platform.contracts import Context, RecommendationRequest
from decision_platform.data.synthetic import generate_dataset
from decision_platform.logging import configure_logging
from decision_platform.policies.exploration import simulate_policy
from decision_platform.serving.dependencies import get_engine
from decision_platform.training.pipeline import evaluate, train


def generate() -> None:
    settings = get_settings()
    data = generate_dataset(seed=settings.random_seed)
    data.save(settings.data_dir)
    print(
        json.dumps(
            {
                "users": len(data.users),
                "items": len(data.items),
                "interactions": len(data.interactions),
            }
        )
    )


def train_command() -> None:
    settings = get_settings()
    print(
        json.dumps(train(settings.data_dir, settings.artifact_dir, settings.random_seed), indent=2)
    )


def demo() -> None:
    settings = get_settings()
    generate()
    train_command()
    get_engine.cache_clear()
    engine = get_engine()
    response = engine.recommend(
        RecommendationRequest(user_id="u00001", context=Context(device="mobile", hour=20), k=5)
    )
    simulation = simulate_policy(seed=settings.random_seed)
    print(response.model_dump_json(indent=2))
    print(json.dumps({"exploration_simulation": simulation}, indent=2))


def main() -> None:
    configure_logging(get_settings().log_level)
    parser = argparse.ArgumentParser(description="Decision platform workflows")
    parser.add_argument("command", choices=["generate-data", "train", "evaluate", "demo", "clean"])
    args = parser.parse_args()
    settings = get_settings()
    if args.command == "generate-data":
        generate()
    elif args.command == "train":
        train_command()
    elif args.command == "evaluate":
        print(json.dumps(evaluate(settings.artifact_dir), indent=2))
    elif args.command == "demo":
        demo()
    elif args.command == "clean":
        for path in [settings.data_dir, settings.artifact_dir]:
            if (
                path.resolve() in [settings.data_dir.resolve(), settings.artifact_dir.resolve()]
                and path.exists()
            ):
                shutil.rmtree(path)


if __name__ == "__main__":
    main()
