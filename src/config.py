"""
src/config.py
Loads config/config.yaml and resolves environment-variable placeholders.

Every module imports settings from here; nothing reads config.yaml directly.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any
import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config" / "config.yaml"


def _resolve_env(value: Any) -> Any:
    """
    Replace ${VAR:default} placeholders with environment values.
    Works recursively through nested dicts / lists.
    """
    if isinstance(value, str):
        pattern = r"\$\{(\w+)(?::([^}]*))?\}"

        def replacer(m: re.Match) -> str:
            env_key = m.group(1)
            default = m.group(2) if m.group(2) is not None else ""
            return os.environ.get(env_key, default)

        return re.sub(pattern, replacer, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def load_config(path: Path = _CONFIG_PATH) -> dict:
    """Load and resolve the project configuration."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _resolve_env(raw)


# Load local .env if present
load_dotenv()

# Module-level singleton so all imports share the same dict
CONFIG: dict = load_config()

# Convenience shortcuts
DB_PASSWORD: str = os.environ.get("DB_PASSWORD", "")
DB_URL: str = (
    f"postgresql://{CONFIG['database']['user']}:"
    f"{DB_PASSWORD}@"
    f"{CONFIG['database']['host']}:{CONFIG['database']['port']}/"
    f"{CONFIG['database']['name']}"
)

ARTIFACTS_ROOT = _ROOT / CONFIG["artifacts"]["root"]
PREPROCESSOR_PATH = _ROOT / CONFIG["artifacts"]["preprocessor"]
MODEL_PATH = _ROOT / CONFIG["artifacts"]["final_model"]
FEATURE_NAMES_PATH = _ROOT / CONFIG["artifacts"]["feature_names"]
MODEL_THRESHOLD: float = float(CONFIG["model"]["threshold"])
MODEL_VERSION: str = CONFIG["project"]["version"]
MLFLOW_URI: str = CONFIG["mlflow"]["tracking_uri"]
MLFLOW_MODEL_NAME: str = CONFIG["mlflow"]["model_name"]
