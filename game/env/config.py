"""
Env Configuration Loader
========================

Single source of truth for every tunable in the env package. Values live
in `config.json` (next to this file) so they can be edited without
touching the .py files.

Usage:
    from game.env.config import config

    grid = config.GRID_SIZE
    wall_color = tuple(config.COLORS.WALL)
    levels = {int(k): v for k, v in config.LEVEL_CONFIG.items()}

The loader runs once at import time. If you need to reload (e.g., for
tests), call `load_config(force=True)` and rebind the `config` symbol.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any, Dict


_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.json"
)


def _to_namespace(obj: Any) -> Any:
    """Wrap a flat dict into a SimpleNamespace for attribute access.

    Nested dicts/lists are kept as plain Python containers so callers
    can still use `.items()`, `["key"]`, slicing, etc. on them.
    """
    if isinstance(obj, dict):
        return SimpleNamespace(**obj)
    return obj


def load_config(path: str = _CONFIG_PATH, force: bool = False) -> SimpleNamespace:
    """Load and return the config namespace.

    Args:
        path: Path to the JSON file. Defaults to the bundled config.json.
        force: Reserved for future caching-invalidation hooks.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)

    # Strip the leading "_comment" key (not a real setting).
    raw.pop("_comment", None)

    return _to_namespace(raw)


# Module-level singleton — imported by other env modules.
config: SimpleNamespace = load_config()
