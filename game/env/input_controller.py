import base64
import io
import json
import os
import pickle
import random
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym

import pygame

from game.env.config import config as _cfg


# ============================================================================
# Model introspection helpers
# ============================================================================
# Regex patterns for matching algorithm names in model filenames.
# Anchored on word boundaries (`_-. /`) so e.g. "abracadabra" doesn't match.
_ALGO_FILENAME_PATTERNS = {
    "ppo": re.compile(r"(?:^|[_\-\s/.])ppo(?:[_\-\s/.]|$)", re.IGNORECASE),
    "dqn": re.compile(r"(?:^|[_\-\s/.])dqn(?:[_\-\s/.]|$)", re.IGNORECASE),
}

# Regex patterns for matching obs_type tokens in model filenames.
# The model's filename embeds one of these short tokens right after
# the algorithm token — see game/model/configs/__init__.py for the
# full list of supported obs types and their token mappings. The
# patterns accept both the canonical token ("12bit", "sptmp",
# "sptmp_lgcy") and a few human-friendly aliases that show up in
# hand-named files (e.g. someone named their model `…_spatiotemporal_…`
# instead of `…_sptmp_…`).
_OBS_TYPE_FILENAME_PATTERNS = {
    "12bit": re.compile(
        r"(?:^|[_\-\s/.])12bit(?:[_\-\s/.]|$)",
        re.IGNORECASE,
    ),
    "spatiotemporal": re.compile(
        r"(?:^|[_\-\s/.])(?:sptmp|spatiotemporal)(?:[_\-\s/.]|$)",
        re.IGNORECASE,
    ),
    "spatiotemporal_legacy": re.compile(
        r"(?:^|[_\-\s/.])(?:sptmp_lgcy|sptmp_legacy|spatiotemporal_legacy|spatiotemporal_v1|legacy_sptmp)(?:[_\-\s/.]|$)",
        re.IGNORECASE,
    ),
}

# Stable-Baselines3 writes policy classes as pickled references wrapped in
# ``{':type:': '<class-string>', ':serialized:': '<base64-pickle>'}``.
# When unpickled, the class's ``__module__`` tells us which algorithm it
# belongs to:
#   - ``stable_baselines3.common.policies`` → ActorCriticPolicy → PPO
#   - ``stable_baselines3.dqn.policies``    → DQNPolicy         → DQN
# We keep the legacy plain-string mapping as a fallback for models saved
# by older SB3 versions that wrote the class name as a JSON string.
_POLICY_MODULE_TO_ALGO = {
    "stable_baselines3.common.policies": "ppo",
    "stable_baselines3.dqn.policies": "dqn",
}
_POLICY_CLASS_TO_ALGO = {
    "ActorCriticPolicy": "ppo",
    "DQNPolicy": "dqn",
}


def _decode_sb3_serialized(blob: str) -> Any:
    """Decode SB3's ``{':serialized:': '<base64>'}`` payload.

    Returns the unpickled Python object. Raises ``Exception`` on any
    decode error — caller decides how to handle it.
    """
    return pickle.loads(base64.b64decode(blob))


def _infer_obs_type_from_shape(shape) -> Optional[str]:
    """Map an SB3 ``observation_space.shape`` to our env's ``obs_type``.

    * ``(8, H, W)``  → ``"spatiotemporal"``        (v2 — current default)
    * ``(4, H, W)``  → ``"spatiotemporal_legacy"`` (v1 — backward-compat)
    * ``(12,)``      → ``"12bit"``
    * else           → ``None`` (unknown)
    """
    tup = tuple(int(s) for s in shape)
    if len(tup) == 3 and tup[0] == 8:
        return "spatiotemporal"
    if len(tup) == 3 and tup[0] == 4:
        return "spatiotemporal_legacy"
    if tup == (12,):
        return "12bit"
    return None


def _infer_obs_type_from_filename(zip_path: str) -> Optional[str]:
    """Map a saved-model filename to the env's ``obs_type`` by regex.

    The filename convention enforced by
    :func:`game.train.utility.auto_naming` embeds an obs-type token
    right after the algorithm token::

        <prefix>_<algo>[_<obstype>]_level<L>[_<n>].zip
                                                │
                                                └─ obs_type token here
                                                   (12bit, sptmp, sptmp_lgcy)

    Falls back to ``None`` when the filename doesn't advertise an obs
    type. This helper is the second-pass fallback used after the SB3
    ``observation_space.shape`` lookup fails — see
    :func:`_read_sb3_metadata` for the priority order.

    Note that we deliberately iterate the patterns in a fixed order so
    that ``spatiotemporal_legacy`` is matched BEFORE ``spatiotemporal``
    (the longer token is a strict superset of the shorter prefix, so
    first-match-wins is the only correct ordering).
    """
    fname = Path(zip_path).name
    # Order matters: check the more specific token first so the legacy
    # variant doesn't get swallowed by the generic spatiotemporal match.
    for obs_type in ("spatiotemporal_legacy", "12bit", "spatiotemporal"):
        if _OBS_TYPE_FILENAME_PATTERNS[obs_type].search(fname):
            return obs_type
    return None


def _read_sb3_metadata(zip_path: str) -> Dict[str, Any]:
    """Read the SB3 ``data`` JSON from a saved model without loading PyTorch.

    Returns dict with keys:
        ``policy_class``  — SB3 class name (e.g. ``"ActorCriticPolicy"``)
        ``obs_shape``     — tuple from ``observation_space["shape"]``
        ``obs_type``      — mapped via :func:`_infer_obs_type_from_shape`
                            (preferred), falling back to
                            :func:`_infer_obs_type_from_filename` when the
                            metadata obs space is missing/undecodable.
        ``algo``          — ``"ppo"`` or ``"dqn"`` or ``None``
    """
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path!r} is not a valid zip file.")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if "data" not in names:
            raise ValueError(
                f"{zip_path!r} has no 'data' entry — not a Stable-Baselines3 model."
            )
        with z.open("data") as f:
            raw = f.read().decode("utf-8")

    meta = json.loads(raw)

    # ---------- Algorithm -------------------------------------------------
    # SB3 ≥1.6 stores ``policy_class`` as
    # ``{"type": "...", "serialized": "<base64-pickle>"}``. Older
    # versions wrote a plain string. Handle both.
    algo: Optional[str] = None
    policy_module = ""
    policy_field = meta.get("policy_class")
    if isinstance(policy_field, str) and policy_field:
        # Legacy: bare class name.
        algo = _POLICY_CLASS_TO_ALGO.get(policy_field)
        policy_module = policy_field
    elif isinstance(policy_field, dict):
        serialized = policy_field.get(":serialized:")
        if isinstance(serialized, str) and serialized:
            try:
                cls_obj = _decode_sb3_serialized(serialized)
                policy_module = getattr(cls_obj, "__module__", "") or ""
                algo = _POLICY_MODULE_TO_ALGO.get(policy_module)
                if algo is None:
                    # Fallback: try the class name itself.
                    algo = _POLICY_CLASS_TO_ALGO.get(
                        getattr(cls_obj, "__name__", "")
                    )
            except Exception:
                # Pickle may fail if the model was saved on a different
                # machine or with a different SB3 version. Continue to
                # filename-regex fallback below.
                pass

    # Final fallback: regex on the filename.
    if algo is None:
        fname = Path(zip_path).name.lower()
        if _ALGO_FILENAME_PATTERNS["ppo"].search(fname):
            algo = "ppo"
        elif _ALGO_FILENAME_PATTERNS["dqn"].search(fname):
            algo = "dqn"

    # ---------- Observation space -----------------------------------------
    # Same ``{:type:, :serialized:}`` envelope. Unpickle and read ``.shape``.
    obs_shape: Optional[Tuple[int, ...]] = None
    obs_field = meta.get("observation_space")
    if isinstance(obs_field, dict):
        serialized = obs_field.get(":serialized:")
        if isinstance(serialized, str) and serialized:
            try:
                space_obj = _decode_sb3_serialized(serialized)
                if hasattr(space_obj, "shape"):
                    obs_shape = tuple(int(s) for s in space_obj.shape)
            except Exception:
                pass

    # obs_type resolution: prefer shape-based inference (most reliable),
    # fall back to filename regex when the shape lookup fails. This
    # covers the case where the model's ``observation_space`` field is
    # undecodable but the filename still embeds the obs-type token (see
    # game/train/utility.py::auto_naming).
    obs_type: Optional[str] = None
    if obs_shape is not None:
        obs_type = _infer_obs_type_from_shape(obs_shape)
    if obs_type is None:
        obs_type = _infer_obs_type_from_filename(zip_path)

    return {
        "algo": algo,
        "policy_class": policy_module,
        "obs_shape": obs_shape,
        "obs_type": obs_type,
    }


# ============================================================================
# Input Controller (HUMAN + AI PLACEHOLDER)
# ============================================================================

class input_controller:
    """
    Handles all input logic for the game:
        - Human input via arrow keys (with 180° anti-reverse buffering)
        - AI input placeholder (will load trained model later)

    State (for human mode):
        - current_action: last applied direction (None = waiting for first input)
        - queued_action: next direction to apply (buffered)
        - game_started: False until first key pressed
    """

    # Action constants (sourced from config.json — single source of truth)
    ACTION_UP = _cfg.ACTION_UP
    ACTION_RIGHT = _cfg.ACTION_RIGHT
    ACTION_DOWN = _cfg.ACTION_DOWN
    ACTION_LEFT = _cfg.ACTION_LEFT

    # Map pygame keys to action indices
    KEY_TO_ACTION = {
        pygame.K_UP: ACTION_UP,
        pygame.K_DOWN: ACTION_DOWN,
        pygame.K_LEFT: ACTION_LEFT,
        pygame.K_RIGHT: ACTION_RIGHT,
    }

    def __init__(self):
        # Human input state
        self.current_action: Optional[int] = None
        self.queued_action: Optional[int] = None
        self.game_started: bool = False

        # AI state (placeholder)
        self.ai_model = None  # TODO: Load trained model here

        # Pause / return-to-menu flag. Set when the player presses
        # ``R`` mid-game (currently in human mode); the launcher reads
        # it each frame and pops the pause modal when it sees True.
        # Cleared by ``reset_state()`` so it doesn't leak across
        # retries or new games.
        self.return_requested: bool = False

    # ----------------------------------------------------------------
    # Human Input
    # ----------------------------------------------------------------
    def process_human_input(self) -> Tuple[Optional[int], bool]:
        """
        Read pygame events and update input state.

        Returns:
            action: Action to step the env with (None = no action / reset / quit)
            keep_running: False if user wants to quit, True otherwise
        """
        keep_running = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keep_running = False
                action = None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    keep_running = False
                    action = None
                elif event.key == pygame.K_r:
                    # Pause / return-to-menu request. Just raise a
                    # flag — the launcher (``_run_playing``) detects
                    # it after ``process_human_input`` returns and
                    # transitions to the RETURN_MODAL state. We do
                    # NOT reset state here: the env must remain
                    # frozen so "Continue" can resume the same game.
                    self.return_requested = True
                    action = None
                elif event.key in self.KEY_TO_ACTION:
                    action = self.KEY_TO_ACTION[event.key]
                    # First key starts the game
                    if not self.game_started:
                        self.current_action = action
                        self.game_started = True
                    # Buffer subsequent input
                    self.queued_action = action
                else:
                    action = None
            else:
                action = None

        # Also poll for held keys (more responsive for continuous press)
        keys = pygame.key.get_pressed()
        for key, action in self.KEY_TO_ACTION.items():
            if keys[key]:
                if not self.game_started:
                    self.current_action = action
                    self.game_started = True
                self.queued_action = action
                break  # Only one direction at a time

        # Apply queued action (with anti-reverse protection)
        if self.game_started and self.queued_action is not None:
            # Check 180° reverse: prevent immediate reversal
            # This requires the env's current direction_idx — handled by env.step()

            # If you want pre-check (optional, since env also checks):
            action = self.queued_action
            self.queued_action = None
            self.current_action = action   # Track the last applied direction
            return action, keep_running

        # No new input this frame
        if self.game_started and self.current_action is not None:
            return self.current_action, keep_running

        return None, keep_running

    # ----------------------------------------------------------------
    # AI Input (Placeholder)
    # ----------------------------------------------------------------
    def process_ai_input(self, obs=None) -> int:
        """
        Get the next action for the AI agent.

        If a model has been loaded via `load_model(...)` and `obs` is
        provided, the model's prediction is used. Otherwise (or if the
        model raises during prediction) this falls back to a random
        action that respects the 180° anti-reverse rule, so the game
        never enters an impossible state.

        Args:
            obs: Current observation from the env. Shape/type must
                 match the obs space the loaded model was trained on
                 (e.g., 8×20×20 for "spatiotemporal", 4×20×20 for
                 "spatiotemporal_legacy", 12-dim for "12bit"). If None,
                 falls back to a random action.

        Returns:
            action: Integer 0-3 (UP, RIGHT, DOWN, LEFT)
        """
        if not self.game_started or self.current_action is None:
            action = self._random_ai_action()
        # --- 1. Try the trained model ---------------------------------
        if self.ai_model is not None and obs is not None:
            try:
                action, _ = self.ai_model.predict(obs, deterministic=True)
                action = int(action)
            except Exception as exc:
                # Wrong obs shape, mismatched model, etc.
                print(
                    f"[input_controller] Model predict() failed: {exc!r}. "
                    f"Falling back to random action."
                )
                action = self._random_ai_action()
        else:
            # --- 2. Fallback: random anti-reverse policy --------------
            action = self._random_ai_action()

        # Track the last chosen action (mirrors the human-input path)
        self.current_action = action
        self.game_started = True
        return action

    def _random_ai_action(self) -> int:
        """
        Random action with 180° anti-reverse protection.

        90% of the time the snake will not pick the immediate reverse
        direction; 10% it does, so the snake still has a chance of
        recovering from a dead-end during early training / fallback.
        """
        if not self.game_started or self.current_action is None:
            # First move: any direction is valid
            return 0

        if random.random() < 0.9:
            opposite = (self.current_action + 2) % 4
            safe = [a for a in range(4) if a != opposite and a != self.current_action]
            return random.choice(safe)
        return random.randint(0, 3)

    def load_model(self, model_path: str) -> Dict[str, Any]:
        """
        Load a trained PPO or DQN model from disk.

        Inspects the SB3 ``.zip`` metadata (regex on filename + policy
        class lookup) to determine the algorithm and the required env
        ``obs_type``, then dispatches to the correct loader.

        Returns a dict::

            {
                "algo":         "ppo" | "dqn",
                "obs_type":     "spatiotemporal" | "12bit",
                "obs_shape":    tuple,           # raw shape from metadata
                "policy_class": str,             # SB3 class name
                "model":        BaseAlgorithm,   # the loaded model
            }

        The caller MUST build an env whose ``obs_type`` equals the
        returned ``obs_type`` — otherwise ``predict()`` will receive a
        tensor of the wrong shape and silently produce garbage actions.
        The legacy 4-channel obs (``"spatiotemporal_legacy"``) is kept
        so any model trained on the old 4-channel format keeps loading
        alongside newer 10-channel models.

        Raises:
            RuntimeError: if the algorithm cannot be determined, the
            observation space is unknown, or the model file is corrupt.

        Usage::

            info = controller.load_model("saved_models/dqn_baseline.zip")
            env = game_environment(level=2, obs_type=info["obs_type"])
            # env is now guaranteed to match the model's obs space.
        """
        # 1) Inspect the zip metadata — cheap, no PyTorch involved.
        try:
            meta = _read_sb3_metadata(model_path)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read SB3 metadata from {model_path!r}: {exc!r}"
            ) from exc

        algo = meta["algo"]
        obs_type = meta["obs_type"]
        policy_class = meta["policy_class"]

        if algo is None:
            raise RuntimeError(
                f"Could not determine algorithm for {model_path!r}. "
                f"Filename should contain 'ppo' or 'dqn' (got policy "
                f"class {policy_class!r})."
            )
        if obs_type is None:
            raise RuntimeError(
                f"Unknown observation space shape {meta['obs_shape']!r} in "
                f"{model_path!r}. Expected spatiotemporal (8,H,W), "
                f"spatiotemporal_legacy (4,H,W) or 12bit (12,)."
            )

        # 2) Dispatch to the correct SB3 loader.
        # Imported lazily so non-AI users don't need stable_baselines3.
        from stable_baselines3 import DQN, PPO

        if algo == "ppo":
            loader, loader_name = PPO, "PPO"
        else:  # algo == "dqn"
            loader, loader_name = DQN, "DQN"

        try:
            self.ai_model = loader.load(model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {loader_name} model from {model_path!r}: {exc!r}"
            ) from exc

        print(
            f"[input_controller] Loaded {loader_name} model "
            f"(obs_shape={meta['obs_shape']!r}, obs_type={obs_type!r}): "
            f"{model_path}"
        )

        return {
            "algo": algo,
            "obs_type": obs_type,
            "obs_shape": meta["obs_shape"],
            "policy_class": policy_class,
            "model": self.ai_model,
        }

    def reset_state(self):
        """Reset input state (call when env resets)."""
        self.current_action = None
        self.queued_action = None
        self.game_started = False
        # Clear any stale pause/return request so the new game
        # doesn't immediately re-pop the modal.
        self.return_requested = False
