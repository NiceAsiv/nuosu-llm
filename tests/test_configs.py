from pathlib import Path

from nuosu_llm.config import load_config


def test_all_training_configs_are_valid() -> None:
    config_dir = Path(__file__).parents[1] / "configs"
    paths = sorted(config_dir.glob("*.yaml"))
    assert paths
    for path in paths:
        load_config(path)
