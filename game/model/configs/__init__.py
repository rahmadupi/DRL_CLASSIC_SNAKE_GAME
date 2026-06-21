"""
JSON Configuration Loader
==========================

Shared loader for :mod:`game.train.dqn_trainer` and
:mod:`game.train.ppo_trainer` so the TUI launcher and CLI scripts can
read the same hyperparameters from one place.

Two JSON files live next to this module:

* ``dqn_config.json`` — DQN trainer defaults (12-bit + spatiotemporal).
* ``ppo_config.json`` — PPO trainer defaults (12-bit + spatiotemporal).

Each trainer exposes a ``from_json_dict(...)`` classmethod that merges
the JSON file with per-run overrides (obs_type, level,
total_timesteps, …). This keeps the JSON file as the single source of
truth for "knobs I rarely change" while still letting the TUI
override the user-facing parameters on every run.

Usage::

    from game.model.configs import load_config

    cfg = load_config("dqn")                  # -> dict
    cfg["learning_rate"] = 1e-4               # optional override

    from game.train.dqn_trainer import DQNTrainingConfig
    config = DQNTrainingConfig.from_json_dict(
        cfg,
        level=3,
        obs_type="spatiotemporal",
    )
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Module-level paths (anchored to this file, not CWD)
# ---------------------------------------------------------------------------
_CONFIGS_DIR = Path(__file__).resolve().parent

_DQN_CONFIG_PATH = _CONFIGS_DIR / "dqn_config.json"
_PPO_CONFIG_PATH = _CONFIGS_DIR / "ppo_config.json"

_ALGO_TO_PATH = {
    "dqn": _DQN_CONFIG_PATH,
    "ppo": _PPO_CONFIG_PATH,
}


# ---------------------------------------------------------------------------
# Mapping between long obs_type names and short filename tokens
# ---------------------------------------------------------------------------
# Trainers / env use the long names ("spatiotemporal", "12bit"). File
# names use short tokens ("sptmp", "12bit") so the stem stays compact
# and the TUI preview line still fits in the centered 50-char container.
#
# Token rules (used by auto_naming + the obs-type detection regex in
# game_launcher.py / input_controller.py):
#
#   "12bit"                 -> "12bit"
#   "spatiotemporal"        -> "sptmp"
#   "spatiotemporal_legacy" -> "sptmp_lgcy"
#
# Always add new obs types here in BOTH directions so the naming,
# detection, and training sides stay in lock-step.
OBS_TYPE_TO_TOKEN: Dict[str, str] = {
    "12bit": "12bit",
    "spatiotemporal": "sptmp",
    "spatiotemporal_legacy": "sptmp_lgcy",
}
TOKEN_TO_OBS_TYPE: Dict[str, str] = {v: k for k, v in OBS_TYPE_TO_TOKEN.items()}


def obs_type_token(obs_type: str) -> str:
    """
    Convert a long obs_type name to its short filename token.

    Raises ``ValueError`` for unknown obs types so callers fail loudly
    rather than silently emitting "unknown_<obs_type>" into filenames.
    """
    if obs_type not in OBS_TYPE_TO_TOKEN:
        raise ValueError(
            f"Unknown obs_type {obs_type!r}. "
            f"Known types: {sorted(OBS_TYPE_TO_TOKEN)}"
        )
    return OBS_TYPE_TO_TOKEN[obs_type]


def obs_type_from_token(token: str) -> Optional[str]:
    """
    Convert a filename token back to the long obs_type name.

    Returns ``None`` when the token is unknown so callers can decide
    whether to fall back to the SB3 metadata (``observation_space.shape``)
    or to raise.
    """
    return TOKEN_TO_OBS_TYPE.get(token.lower())


# ---------------------------------------------------------------------------
# JSON loader
# ---------------------------------------------------------------------------
def load_config(algo: str, path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Read the JSON config for ``algo`` and return the raw dict.

    Args:
        algo:  ``"dqn"`` or ``"ppo"``. Selects the default JSON file
               under :mod:`game.model.configs`.
        path:  Optional explicit path to a JSON file. Overrides
               ``algo`` when both are given.

    Returns:
        Dict of hyperparameters (the leading ``_comment`` key, if any,
        is stripped).

    Raises:
        FileNotFoundError: If the resolved path doesn't exist.
        ValueError:        If ``algo`` is not "dqn" / "ppo" and
                           ``path`` is None.
    """
    resolved: Optional[Path] = None
    if path is not None:
        resolved = Path(path)
    else:
        if algo not in _ALGO_TO_PATH:
            raise ValueError(
                f"Unknown algorithm {algo!r}; expected 'dqn' or 'ppo'."
            )
        resolved = _ALGO_TO_PATH[algo]

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Config file not found: {resolved}. "
            f"Make sure {resolved.name} exists under game/model/configs/."
        )

    with open(resolved, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # Strip optional leading comment so it doesn't leak into the trainer
    # dataclass (which would raise ``TypeError: unexpected keyword
    # argument '_comment'``).
    data.pop("_comment", None)
    return data
