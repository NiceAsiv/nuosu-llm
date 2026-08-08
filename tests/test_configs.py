from pathlib import Path

from nuosu_llm.config import load_config


def test_all_training_configs_are_valid() -> None:
    root = Path(__file__).parents[1]
    paths = sorted((root / "configs").rglob("*.yaml"))
    paths.extend(sorted((root / "recipes").rglob("*.yaml")))
    assert paths
    for path in paths:
        load_config(path)
