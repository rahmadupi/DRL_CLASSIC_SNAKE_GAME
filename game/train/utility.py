"""
Training Utilities
==================

Shared infrastructure for the PPO and DQN trainers:

* :func:`make_vec_env`   — builds a vectorised env (DummyVecEnv or SubprocVecEnv).
* :func:`auto_naming`    — produces a unique, collision-free output `.zip` path.
* :func:`resolve_logger_dir` — TensorBoard log directory per run.
* :func:`detect_device`  — auto-select CUDA when available, else CPU.
* :func:`get_cpu_count`  — capped number of worker processes to spawn.

Why both VecEnv types?
----------------------
* ``DummyVecEnv`` runs every env in the **same** Python process. It is the
  only safe choice when ``n_envs == 1`` AND a renderer / GUI is attached
  (Pygame display objects cannot cross process boundaries).
* ``SubprocVecEnv`` runs each env in a **separate** worker process. True
  parallelism — CPU-bound env stepping scales linearly with the core count.
  Forced ``headless`` because each worker has no access to the parent's
  display server.

The trainer (or TUI launcher) decides which one to use based on the
parallelisation setting chosen by the user.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn, get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from tqdm.auto import tqdm

from game.env.config import config as _cfg

from game.env.game_environment import game_environment


# ---------------------------------------------------------------------------
# Learning-rate schedule helper (shared by PPO and DQN trainers)
# ---------------------------------------------------------------------------
def build_lr_schedule(
    learning_rate: float,
    use_linear: bool = True,
    end_fraction: float = 0.0,
) -> Callable[[float], float]:
    """
    Build a SB3-compatible learning-rate schedule callable.

    Returns a callable ``f(progress_remaining: float) -> lr`` that
    SB3 invokes every step to look up the current LR. ``progress_remaining``
    walks from ``1.0`` (training start) to ``0.0`` (training end).

    Args:
        learning_rate:  Initial LR (at ``progress_remaining = 1.0``).
        use_linear:     When ``True`` (default — the original PPO recipe),
                        LR decays linearly from ``learning_rate`` down to
                        ``learning_rate * end_fraction`` over the course
                        of training. When ``False``, LR stays constant.
        end_fraction:   Final LR as a **fraction of the initial LR**.
                        ``0.0`` (default) → decay all the way to zero.
                        ``1.0``           → no decay (equivalent to
                                             ``use_linear=False``).

    Notes
    -----
    This helper is shared between the PPO and DQN trainers so both
    algorithms get consistent late-training behaviour. A constant LR
    on a mature policy causes oscillation around the local optimum
    — the linear decay is what stabilises the resumed runs.
    """
    if use_linear:
        return get_linear_fn(learning_rate, learning_rate * end_fraction, 1.0)
    return get_schedule_fn(learning_rate)


# ---------------------------------------------------------------------------
# Project paths (anchored to the repo root, not CWD)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
SAVED_MODELS_DIR = _ROOT / "saved_models"
LOG_ROOT = _ROOT / "logs" / "tb_logs"

SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------
class ConfigurableMonitor(Monitor):
    """Monitor wrapper that forwards custom ``info_keywords`` to SB3.

    SB3's default :class:`~stable_baselines3.common.monitor.Monitor`
    accepts an ``info_keywords`` tuple, but the env factory in
    :func:`make_vec_env` constructs it with the default empty tuple —
    which silently drops the env's terminal info (e.g.
    ``snake_length``) from the ``info["episode"]`` dict. This subclass
    exists purely so the factory can opt into custom keywords without
    repeating the kwarg plumbing at every call site.

    Note on SB3 2.2.1 API: SB3 2.x removed the legacy
    ``self.ep_info_buffer`` deque. Episode info is delivered through
    ``info["episode"]`` on the terminated/truncated step. The training
    callbacks in this module read from ``self.locals["infos"]`` (which
    SB3 populates every step inside the train loop) and accumulate
    their own rolling window — see :class:`_EpisodeHistory`.
    """

    # Marker subclass; no extra behaviour needed beyond being a distinct
    # type for ``isinstance`` checks (used in unit tests).


def _make_env_fn(
    level: int,
    obs_type: str,
    seed: Optional[int] = None,
) -> Callable[[], game_environment]:
    """
    Build a thunk that returns a *fresh* ``game_environment`` instance.

    A thunk is required because ``SubprocVecEnv`` pickles the callable and
    re-invokes it in each worker process — passing the env instance itself
    would fail to serialise the unpicklable parts (random state, etc.).
    """
    def _thunk() -> game_environment:
        env = game_environment(level=level, obs_type=obs_type)
        # Wrap with ConfigurableMonitor so the env's terminal info
        # (e.g. ``snake_length``) is merged into the per-episode dict
        # that SB3 surfaces as ``info["episode"]``. SB3's default
        # ``info_keywords=()`` would silently drop everything.
        env = ConfigurableMonitor(
            env,
            info_keywords=("snake_length",),
        )
        if seed is not None:
            env.reset(seed=seed)
        return env

    return _thunk


def make_vec_env(
    level: int,
    obs_type: str = "spatiotemporal",
    n_envs: int = 1,
    seed: Optional[int] = None,
    headless: Optional[bool] = None,
) -> VecEnv:
    """
    Build a vectorised environment.

    Args:
        level:     Curriculum level (1-5, see ``game_environment.LEVEL_CONFIG``).
        obs_type:  ``"spatiotemporal"`` for PPO, ``"12bit"`` for DQN.
        n_envs:    Number of parallel envs. ``1`` → ``DummyVecEnv``; ``>1`` → ``SubprocVecEnv``.
        seed:      Optional base seed (each worker offsets it).
        headless:  Force-disable rendering. If ``None`` we auto-decide:
                   * ``n_envs == 1`` → ``headless=False`` (Demo Mode possible)
                   * ``n_envs  > 1`` → ``headless=True``  (SubprocVecEnv cannot render anyway)

    Returns:
        A SB3 ``VecEnv`` instance.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")

    fns = [_make_env_fn(level=level, obs_type=obs_type, seed=(seed + i) if seed is not None else None)
           for i in range(n_envs)]

    if n_envs == 1:
        # Single in-process env — the only mode that can drive a renderer.
        return DummyVecEnv(fns)

    # Multi-process: must be headless; SubprocVecEnv pickles env thunks.
    if headless is False:
        raise ValueError(
            "Cannot use headless=False with n_envs > 1 — SubprocVecEnv workers "
            "have no access to the parent's display server."
        )
    return SubprocVecEnv(fns, start_method="spawn")


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------
_INVALID_FS_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitise_prefix(prefix: str) -> str:
    """Strip filesystem-unfriendly characters from the user-typed prefix."""
    cleaned = _INVALID_FS_CHARS.sub("-", prefix.strip())
    return cleaned.strip("-") or "model"


def auto_naming(
    prefix: str,
    algo: str,
    level: int,
    base_dir: Optional[os.PathLike] = SAVED_MODELS_DIR,
) -> Path:
    """
    Return a unique output path of the form::

        <base_dir>/<prefix>_<algo>_level<L>[_<n>].zip

    If that filename already exists, append ``_<n>`` (incrementing ``n``
    from 1) until a free name is found. This avoids silently overwriting
    earlier checkpoints.

    Args:
        prefix:   User-provided label (will be sanitised).
        algo:     ``"ppo"`` or ``"dqn"``.
        level:    Curriculum level (1-5).
        base_dir: Destination directory. Defaults to ``saved_models/``.

    Returns:
        An unused ``Path`` whose suffix is ``.zip``.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    safe_prefix = _sanitise_prefix(prefix)
    algo = algo.lower()
    stem = f"{safe_prefix}_{algo}_level{level}"

    candidate = base_dir / f"{stem}.zip"
    if not candidate.exists():
        return candidate

    # Auto-increment: foo_ppo_level3.zip → foo_ppo_level3_1.zip → _2 ...
    n = 1
    while True:
        candidate = base_dir / f"{stem}_{n}.zip"
        if not candidate.exists():
            return candidate
        n += 1


def resolve_logger_dir(
    prefix: str,
    algo: str,
    level: int,
    root: Optional[os.PathLike] = LOG_ROOT,
) -> Path:
    """
    Build the TensorBoard log directory for a run.

    Mirrors the model-naming scheme so logs and weights are easy to pair::

        logs/tb_logs/<prefix>_<algo>_level<L>[_<n>]/
    """
    root = Path(root)
    safe_prefix = _sanitise_prefix(prefix)
    stem = f"{safe_prefix}_{algo.lower()}_level{level}"

    candidate = root / stem
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    # If a previous run used the same prefix, branch off a sibling folder.
    n = 1
    while True:
        candidate = root / f"{stem}_{n}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        n += 1


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------
def detect_device() -> str:
    """Return ``"cuda"`` if a GPU is available, else ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_cpu_count() -> int:
    """
    Return the number of CPUs available to *this* process.

    SB3's ``SubprocVecEnv`` will fork one worker per env, so this is the
    upper bound the TUI launcher should expose to the user.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # macOS / Windows fallback
        return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# Convenience for tests / scripts
# ---------------------------------------------------------------------------
def preview_name(prefix: str, algo: str, level: int) -> Tuple[str, str]:
    """
    Return ``(display_name, full_path)`` for UI previews.

    The display name is the stem only (no directory, no ``.zip``), which
    is what the TUI launcher shows in its preview row.
    """
    path = auto_naming(prefix, algo, level)
    return path.stem, str(path)


# ---------------------------------------------------------------------------
# Shared episode-history tracker
# ---------------------------------------------------------------------------
class _EpisodeHistory:
    """
    Rolling deque of completed episode dicts.

    SB3 ≤ 1.x exposed the latest episodes via ``model.ep_info_buffer``
    (a ``deque(maxlen=100)``). SB3 2.2.1 dropped that attribute — the
    per-episode info dict now rides in ``info["episode"]`` on the
    terminated/truncated step, and the train loop forwards it through
    ``self.locals["infos"]`` on every callback ``_on_step``. We
    accumulate those dicts here so the progress bar and Pygame curve
    panel have a stable, shared source of truth.

    Capacity defaults to **200** (vs SB3's old hardcoded 100) so the
    rolling reward mean isn't dominated by single ±10 spikes from food
    or collision events — Snake's per-episode reward swings are large.

    Two callbacks share this:

    * :class:`RewardProgressBarCallback` — renders the history as a
      Unicode sparkline in the tqdm postfix (works in headless mode).
    * :class:`TrainingRenderCallback` — renders it as a Pygame line
      graph in the training window.

    Sharing the logic means the two views stay in lock-step — if you
    see the curve go up in one, the other reflects the same data.
    """

    def __init__(self, capacity: int = 200) -> None:
        self._data: deque = deque(maxlen=capacity)

    def add(self, ep_info) -> None:
        """Append one completed-episode dict (from ``info["episode"]``)."""
        if ep_info is not None:
            self._data.append(ep_info)

    @property
    def data(self) -> deque:
        return self._data

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# Progress bar callback
# ---------------------------------------------------------------------------
class RewardProgressBarCallback(BaseCallback):
    """
    tqdm progress bar that overlays mean episode reward + a live sparkline.

    Replaces SB3's built-in ``progress_bar`` so the user sees reward-derived
    metrics alongside the step counter. Reads from
    ``self.model.ep_info_buffer`` (a ``deque(maxlen=100)`` populated by
    :class:`~stable_baselines3.common.monitor.Monitor`, which
    :func:`make_vec_env` now installs automatically).

    Postfix keys
    ------------
    * ``rew``   — rolling mean episode reward over the last 100 episodes.
    * ``len``   — rolling mean snake **body length** at episode end
                  (i.e. ``INITIAL_SNAKE_LENGTH + foods_eaten``). Read
                  from ``ep_info_buffer[ep]["snake_length"]`` which the
                  env stores in its terminal info dict. Older episodes
                  (before this key was added) report 0 and are skipped
                  in the average.
    * ``eps``   — total episodes finished since training start.
    * ``envs``  — only shown when ``n_envs > 1`` (multi-env runs).
    * ``graph`` — Unicode sparkline of recent per-episode rewards.
                  Uses ``▁▂▃▄▅▆▇█`` block characters (8 vertical
                  levels). Auto-scales to the local min/max so even
                  small reward changes are visible. Works in any
                  terminal supporting Unicode (essentially all modern
                  terminals), including headless SSH sessions where
                  no display server is available.

    Usage::

        cb = RewardProgressBarCallback(total_timesteps=N, desc="PPO")
        model.learn(total_timesteps=N, callback=[cb], progress_bar=False)

    Why we don't extend SB3's built-in bar
    --------------------------------------
    SB3's internal ``_ProgressBarCallback`` is private API and doesn't
    expose a stable hook to inject custom postfix values. Owning the
    ``tqdm`` instance directly is cleaner and lets us use a custom
    ``bar_format`` for readability.
    """

    # 8 vertical levels of Unicode block characters (low → high).
    # These are rendered as a single column of varying height, which
    # gives the classic "sparkline" look inside the tqdm postfix.
    SPARK_CHARS = "▁▂▃▄▅▆▇█"

    def __init__(
        self,
        total_timesteps: int,
        desc: str = "Training",
        graph_width: int = 20,
        show_graph: bool = True,
        total_episodes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._total = int(total_timesteps)
        self._desc = desc
        self._graph_width = max(2, int(graph_width))
        self._show_graph = show_graph
        # When the user enters a target episode count in the launcher,
        # we surface it as ``eps/total_episodes`` in the postfix so the
        # bar shows the metric they were thinking in, not just the
        # translated timestep budget underneath.
        self._total_episodes = (
            int(total_episodes) if total_episodes is not None and total_episodes > 0 else None
        )
        self.pbar = None
        # History capacity = 4× graph width so downsampling has
        # enough resolution to preserve peaks and troughs.
        self._history = _EpisodeHistory(capacity=self._graph_width * 4)

    # ---- lifecycle hooks -------------------------------------------------
    def _on_training_start(self) -> None:
        # tqdm.auto picks the right backend for the host (notebook vs tty).
        self.pbar = tqdm(
            total=self._total,
            desc=self._desc,
            unit="step",
            dynamic_ncols=True,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
        )
        self.pbar.set_postfix(rew="–", len="–", eps=0)

    def _on_step(self) -> bool:
        if self.pbar is None:
            return True
        # Sync the bar with SB3's cumulative env-step counter. For vec
        # envs with N workers each step advances the counter by N.
        self.pbar.n = self.model.num_timesteps

        # Capture completed episodes from SB3 2.2.1's per-step locals.
        # On the step that flips ``dones[i]`` to True, Monitor has
        # stuffed ``info["episode"]`` (with r, l, t, plus any
        # ``info_keywords`` like ``snake_length``) into ``infos[i]``.
        #
        # ``infos`` is a Python list (empty ``[]`` when n_envs==0), so
        # ``or []`` is safe. ``dones`` is a numpy array of bools though,
        # and evaluating its truthiness raises
        # "The truth value of an array with more than one element is
        # ambiguous" — use an explicit ``None`` check instead.
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        if dones is None:
            dones = []
        for done, info in zip(dones, infos):
            if done and isinstance(info, dict) and "episode" in info:
                self._history.add(info["episode"])

        eps = len(self._history)
        # Build postfix as a dict so we can mix numeric and string
        # values (older code used a one-element set when ``total_episodes``
        # was set, which broke subsequent ``postfix[k] = v`` assignments).
        postfix: dict = {}
        # ``eps/total_eps`` mirrors what the user typed in the launcher's
        # Episode field. When total_eps is unknown (default-budget run),
        # we just show the running episode count.
        if self._total_episodes is not None:
            postfix["eps"] = f"{eps}/{self._total_episodes}"
        else:
            postfix["eps"] = eps
        if eps > 0:
            mean_rew = sum(ep["r"] for ep in self._history.data) / eps
            postfix["rew"] = f"{mean_rew:+.2f}"
            # Mean snake body length at episode end. Episodes that
            # pre-date the ``snake_length`` info key default to 0;
            # we filter them out so the rolling average reflects only
            # episodes where the env actually reported a length.
            length_episodes = [ep for ep in self._history.data if ep.get("snake_length")]
            if length_episodes:
                mean_len = sum(ep["snake_length"] for ep in length_episodes) / len(length_episodes)
                postfix["len"] = f"{mean_len:.1f}"
        n_envs = getattr(self.model, "n_envs", 1)
        if n_envs > 1:
            postfix["envs"] = n_envs

        # Live reward sparkline (headless-friendly — no display needed).
        if self._show_graph:
            sparkline = self._render_sparkline()
            if sparkline:
                postfix["graph"] = sparkline

        self.pbar.set_postfix(postfix)
        return True

    # ---- sparkline rendering ---------------------------------------------
    def _render_sparkline(self) -> str:
        """
        Downsample the history to ``self._graph_width`` points and map
        each to one of 8 vertical levels of Unicode block characters.

        Auto-scales to local min/max so even small reward swings are
        visible. When all values are equal, returns a flat mid-level
        bar so the user always sees something.
        """
        rewards = [ep["r"] for ep in self._history.data]
        n = len(rewards)
        if n < 2:
            return ""

        width = min(self._graph_width, n)
        if width < 2:
            return ""

        # Evenly-spaced downsampling. Using integer-index sampling (not
        # averaging) preserves peaks/troughs so the sparkline shows
        # the actual shape of the curve rather than smoothing it away.
        sampled: list[float] = []
        for i in range(width):
            idx = int(round(i * (n - 1) / (width - 1)))
            sampled.append(rewards[min(idx, n - 1)])

        y_min = min(sampled)
        y_max = max(sampled)
        if abs(y_max - y_min) < 1e-9:
            # All values identical — show a flat mid bar.
            return self.SPARK_CHARS[3] * width

        # Map each value to a level 0..7 based on its position in [y_min, y_max].
        levels: list[str] = []
        for v in sampled:
            lvl = int(round((v - y_min) / (y_max - y_min) * 7))
            lvl = max(0, min(7, lvl))
            levels.append(self.SPARK_CHARS[lvl])
        return "".join(levels)

        self.pbar.set_postfix(postfix)
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

    # Failsafe: ensure no orphaned tqdm line if training crashes between
    # `_on_training_start` and `_on_training_end`.
    def __del__(self) -> None:
        if self.pbar is not None:
            try:
                self.pbar.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Pygame render callback
# ---------------------------------------------------------------------------
class TrainingRenderCallback(BaseCallback):
    """
    Opens a Pygame window that mirrors the env as the agent trains.

    Designed for the **Demo Mode** mentioned in the PRD — when
    ``n_envs == 1`` the env runs in-process and we can attach a renderer.
    With ``SubprocVecEnv`` (``n_envs > 1``) workers have no access to
    the parent's display server, so this callback silently no-ops.

    The window mirrors ``n_envs[0]`` (the only env when single-env).
    Rendering is throttled to ``fps`` frames per second so the display
    loop doesn't bottleneck SB3's inner step loop (which can run at
    thousands of steps/sec).

    User interaction
    ----------------
    * Closing the window **or** pressing **Esc** aborts training
      (returns ``False`` from ``_on_step``).
    * No other keys are intercepted — SB3 owns the action selection.

    Graceful degradation
    --------------------
    Skips silently (no exception) when any of these are true:

    * ``pygame`` is not installed.
    * No display is available (``pygame.display.set_mode`` raises).
    * ``self.model.get_env()`` is not a ``DummyVecEnv`` (e.g. SubprocVecEnv).
    * The env was closed mid-training.

    Args:
        fps:           Target render FPS. Lower → faster training.
        window_title:  Title for the Pygame window.
        overlay:       When ``True`` (default), draw reward / episode
                       stats AND a live reward curve on top of the frame.
        curve_capacity: How many recent episodes to keep in the rolling
                       history for the live curve.
    """

    def __init__(
        self,
        fps: int = _cfg.DEFAULT_FPS,
        window_title: str = "Snake — Training",
        overlay: bool = True,
        curve_capacity: int = 200,
    ) -> None:
        super().__init__()
        self._fps = max(1, int(fps))
        self._title = window_title
        self._overlay = overlay
        self._pygame = None
        self._renderer = None
        self._screen = None
        self._clock = None
        self._last_render = 0.0
        self._user_closed = False

        # Live reward curve state. The shared ``_EpisodeHistory`` is
        # also used by :class:`RewardProgressBarCallback` so the two
        # views stay in lock-step (the sparkline in tqdm and the Pygame
        # curve panel always show the same data).
        self._history = _EpisodeHistory(capacity=curve_capacity)

    # ---- lifecycle hooks -------------------------------------------------
    def _on_training_start(self) -> None:
        env = self._unwrap_env()
        if env is None:
            # SubprocVecEnv / closed env — silently skip.
            return
        try:
            import pygame  # type: ignore
            from game.env.game_renderer import game_renderer  # type: ignore
        except Exception:
            # pygame not installed — skip without raising.
            return

        self._pygame = pygame
        try:
            self._renderer = game_renderer()
            self._screen = self._renderer.create_window(self._title)
            self._clock = pygame.time.Clock()
        except pygame.error:
            # No DISPLAY available (headless / SSH) — skip.
            self._screen = None
            self._renderer = None

    def _on_step(self) -> bool:
        if self._user_closed:
            return False  # user closed window → abort training
        if self._screen is None:
            return True   # rendering disabled / not available

        # Pump events so the window stays responsive.
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self._user_closed = True
                return False
            if (
                event.type == self._pygame.KEYDOWN
                and event.key == self._pygame.K_ESCAPE
            ):
                self._user_closed = True
                return False

        # Throttle: only redraw when the FPS budget has elapsed so we
        # don't bottleneck the inner SB3 step loop.
        now = time.monotonic()
        if now - self._last_render < 1.0 / self._fps:
            return True
        self._last_render = now

        env = self._unwrap_env()
        if env is None:
            return True

        try:
            self._renderer.draw_frame(self._screen, env)
            if self._overlay:
                self._draw_overlay()
            self._pygame.display.flip()
            self._clock.tick(self._fps)
        except Exception:
            # Window closed externally (X11 crash, etc.) — disable
            # further rendering but keep training alive.
            self._screen = None
            self._renderer = None
        return True

    def _on_training_end(self) -> None:
        self._teardown()

    def __del__(self) -> None:
        # Failsafe for crashes between start and end hooks.
        self._teardown()

    def _teardown(self) -> None:
        if self._screen is not None and self._pygame is not None:
            try:
                self._pygame.display.quit()
            except Exception:
                pass
        self._screen = None
        self._renderer = None
        self._clock = None

    # ---- helpers ---------------------------------------------------------
    def _unwrap_env(self):
        """
        Drill through ``DummyVecEnv`` → ``Monitor`` → ``game_environment``.

        Returns the inner ``game_environment`` instance, or ``None`` if
        the env is not a single in-process vec env (e.g. SubprocVecEnv
        or already closed).
        """
        env = self.model.get_env()
        if env is None:
            return None
        # DummyVecEnv exposes `.envs` (a list of sub-envs).
        envs_attr = getattr(env, "envs", None)
        if not envs_attr:
            return None
        inner = envs_attr[0]
        # Walk wrapper chain (Monitor wraps game_environment). Stop if
        # we reach something that doesn't have an `.env` attribute.
        while hasattr(inner, "env"):
            inner = inner.env
        return inner

    def _draw_overlay(self) -> None:
        """Draw reward + episode info on top of the frame, plus the live curve."""
        from game.env.game_renderer import render_text_pil

        # Capture any episodes that completed this step (SB3 2.2.1
        # forwards them through ``locals["infos"][i]["episode"]``).
        #
        # ``infos`` is a Python list (empty ``[]`` when n_envs==0), so
        # ``or []`` is safe. ``dones`` is a numpy array of bools though,
        # and evaluating its truthiness raises
        # "The truth value of an array with more than one element is
        # ambiguous" — use an explicit ``None`` check instead.
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        if dones is None:
            dones = []
        for done, info in zip(dones, infos):
            if done and isinstance(info, dict) and "episode" in info:
                self._history.add(info["episode"])

        eps = len(self._history)
        rew_str = "—"
        if eps > 0:
            mean_rew = sum(ep["r"] for ep in self._history.data) / eps
            rew_str = f"{mean_rew:+.2f}"

        line1 = (
            f"step {self.model.num_timesteps:,}   eps {eps}   "
            f"rew(avg{len(self._history)}) {rew_str}"
        )
        surface = render_text_pil(line1, font_size=24, color=(40, 30, 20))

        # Anchor: bottom-left of the play area (just inside the wall).
        x = self._renderer.offset + 15
        y = self._renderer.window_h - self._renderer.wall_thickness - 40
        self._screen.blit(surface, (x, y))

        # Live reward curve (bottom-right corner).
        self._draw_reward_curve()

    # ---- live reward curve -----------------------------------------------
    def _draw_reward_curve(self) -> None:
        """Draw the rolling reward curve in the bottom-right corner."""
        if len(self._history) < 2:
            return  # need at least 2 points to draw a line
        from game.env.game_renderer import render_text_pil

        pygame = self._pygame
        r = self._renderer

        # ---- Panel geometry ----
        panel_w = 290
        panel_h = 120
        margin = 12
        panel_x = r.window_w - r.wall_thickness - panel_w - margin
        panel_y = r.window_h - r.wall_thickness - panel_h - margin

        # Background + border
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        pygame.draw.rect(self._screen, (255, 247, 217), panel_rect)  # BG_CREAM
        pygame.draw.rect(self._screen, (40, 30, 20), panel_rect, 1)  # dark border

        # Title (top-left of panel)
        title = render_text_pil("Reward (per episode)", font_size=18, color=(40, 30, 20))
        self._screen.blit(title, (panel_x + 8, panel_y + 4))

        # Latest value (top-right of panel) — what just finished
        latest_rew = self._history.data[-1]["r"]
        latest_surf = render_text_pil(
            f"{latest_rew:+.1f}", font_size=18, color=(20, 60, 120)
        )
        self._screen.blit(
            latest_surf,
            (panel_x + panel_w - latest_surf.get_width() - 8, panel_y + 4),
        )

        # ---- Plot area ----
        plot_x = panel_x + 36   # leave room for y-axis labels
        plot_y = panel_y + 28
        plot_w = panel_w - 44
        plot_h = panel_h - 44

        # Y range (auto-scale with a minimum span so the curve doesn't
        # jitter when rewards are nearly identical).
        rewards = [ep["r"] for ep in self._history.data]
        y_min, y_max = min(rewards), max(rewards)
        if y_max - y_min < 1.0:
            center = (y_max + y_min) / 2
            y_min, y_max = center - 0.5, center + 0.5
        span = y_max - y_min
        y_min -= span * 0.10
        y_max += span * 0.10
        y_range = y_max - y_min

        # X range (sliding window of the last `curve_capacity` episodes).
        # We don't have absolute indices, so we use the buffer position
        # (0 = oldest in window, N-1 = newest) as the x coordinate.
        x_min, x_max = 0, len(self._history.data) - 1
        if x_max == x_min:
            x_max = x_min + 1
        x_range = x_max - x_min

        def to_px(eps_idx: int, rew: float):
            px = plot_x + int((eps_idx - x_min) / x_range * plot_w)
            py = plot_y + plot_h - int((rew - y_min) / y_range * plot_h)
            # Clamp to plot area so the curve doesn't bleed out.
            px = max(plot_x, min(plot_x + plot_w, px))
            py = max(plot_y, min(plot_y + plot_h, py))
            return px, py

        # Zero line (light gray, dashed via short segments)
        if y_min < 0 < y_max:
            zero_y = plot_y + plot_h - int((0 - y_min) / y_range * plot_h)
            for sx in range(plot_x, plot_x + plot_w, 8):
                pygame.draw.line(
                    self._screen, (180, 180, 180),
                    (sx, zero_y), (min(sx + 4, plot_x + plot_w), zero_y), 1,
                )

        # Y-axis labels (min / max)
        y_min_surf = render_text_pil(f"{y_min:+.0f}", font_size=14, color=(80, 80, 80))
        y_max_surf = render_text_pil(f"{y_max:+.0f}", font_size=14, color=(80, 80, 80))
        self._screen.blit(y_min_surf, (panel_x + 4, plot_y + plot_h - 12))
        self._screen.blit(y_max_surf, (panel_x + 4, plot_y - 2))

        # X-axis label — sliding window over the buffer's most recent
        # episodes. We don't have absolute indices; the buffer's
        # position-in-window is what we display.
        x_label = render_text_pil(
            f"oldest ← {len(self._history)} recent eps → newest",
            font_size=14,
            color=(80, 80, 80),
        )
        self._screen.blit(
            x_label,
            (
                plot_x + plot_w - x_label.get_width(),
                plot_y + plot_h + 2,
            ),
        )

        # The curve itself (smooth line in deep blue)
        points = [to_px(i, ep["r"]) for i, ep in enumerate(self._history.data)]
        if len(points) >= 2:
            pygame.draw.lines(self._screen, (20, 60, 120), False, points, 2)

        # Latest-point marker (filled circle in matching blue)
        lx, ly = points[-1]
        pygame.draw.circle(self._screen, (20, 60, 120), (lx, ly), 3)