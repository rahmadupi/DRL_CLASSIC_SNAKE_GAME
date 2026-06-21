"""
DQN Trainer
===========

DQN agent supporting BOTH the 12-bit flat-vector baseline AND the
spatiotemporal (8×20×20) CNN+Transformer encoder. Mirrors
:mod:`game.train.ppo_trainer` so the PPO vs DQN comparison is
apples-to-apples — same env, same curriculum, same logger cadence.

Notable differences from the PPO trainer:

* ``obs_type`` is selectable via :class:`DQNTrainingConfig`:
    - ``"12bit"`` — the flat 12-dim vector, used with the MLP
      :class:`game.model.dqn_12bit.DQN12BitExtractor`.
    - ``"spatiotemporal"`` / ``"spatiotemporal_legacy"`` — the 8- or
      4-channel 20×20 tensor, used with the CNN+attention
      :class:`game.model.ppo_spatiotemporal.SpatiotemporalExtractor`
      (the same encoder used by PPO so the spatial feature learning
      is directly comparable).
* ``n_envs`` is **capped** (``max_n_envs``, default 4) — DQN's
  replay buffer samples trajectories from a single env; too many
  parallel envs introduce off-policy bias.
* Uses :class:`stable_baselines3.DQN` with an MLP policy +
  epsilon-greedy exploration schedule.

Hyperparameter defaults live in
``game/model/configs/dqn_config.json`` and are loaded by
:meth:`DQNTrainingConfig.from_json_dict`. The TUI launcher passes
per-run overrides (level, obs_type, total_timesteps, …).

Usage::

    from game.train.dqn_trainer import train_dqn, DQNTrainingConfig

    config = DQNTrainingConfig.from_json_dict(
        level=1,
        obs_type="spatiotemporal",
    )
    model, saved_path = train_dqn(config)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv

from game.model.configs import load_config as load_dqn_config
from game.model.dqn_12bit import make_dqn_policy_kwargs as make_dqn_12bit_kwargs
from game.model.dqn_spatiotemporal import make_dqn_policy_kwargs as make_dqn_sptmp_kwargs
from game.train.utility import (
    RewardProgressBarCallback,
    RolloutMetricsCallback,
    TrainingRenderCallback,
    auto_naming,
    build_lr_schedule,
    detect_device,
    make_vec_env,
    resolve_logger_dir,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DQNTrainingConfig:
    """All knobs the DQN trainer cares about."""

    # Curriculum
    level: int = 1
    obs_type: str = "12bit"
    # Accepted values:
    #   "12bit"                 — flat 12-dim vector  (MLP encoder)
    #   "spatiotemporal"        — 8×20×20 tensor       (CNN+Attention)
    #   "spatiotemporal_legacy" — 4×20×20 tensor       (CNN+Attention)

    # Parallelism
    # DQN is off-policy (replay buffer), so multi-env helps wall-clock
    # throughput but each env is a separate MDP — too many workers
    # introduce off-policy bias. We cap at 4 as a sane default; set
    # higher explicitly if you know what you're doing.
    n_envs: int = 1
    max_n_envs: int = 4                 # hard upper bound (PPO has none)

    # Schedule
    total_timesteps: int = 200_000
    # Optional user-facing episode target. When set, the progress bar
    # shows ``eps/total_episodes`` instead of just the running count.
    total_episodes: Optional[int] = None
    learning_rate: float = 5e-4
    buffer_size: int = 100_000          # replay buffer capacity
    learning_starts: int = 1_000        # random exploration before training starts
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 1.0                    # soft target update coefficient
    train_freq: int = 4                 # gradient steps every N env steps
    gradient_steps: int = 1             # how many GD steps per train_freq
    target_update_interval: int = 1_000

    # Learning-rate schedule.
    # When ``use_linear_schedule=True`` (default — recommended) the
    # learning rate decays linearly from ``learning_rate`` down to
    # ``learning_rate * lr_end_fraction`` over the course of training.
    # This mirrors what we did for PPO and fixes the same late-training
    # oscillation you saw on resumed runs.
    use_linear_schedule: bool = True
    lr_end_fraction: float = 0.0        # 0.0 → decay LR all the way to 0

    # Exploration (linear decay from start -> end over fraction of training)
    # SB3's epsilon schedule is already linear-by-default; we only override
    # it on resume (see ``build_dqn``) so a resumed run doesn't re-explore.
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05

    # Architecture — 12-bit MLP path
    hidden_dim: int = 64
    features_dim: int = 64

    # Architecture — spatiotemporal (CNN+attention) path
    cnn_channels: int = 32
    d_model: int = 64
    n_heads: int = 8
    dropout: float = 0.0
    use_attention: bool = True

    # Output
    output_prefix: str = "snake"
    checkpoint_freq: int = 50_000
    seed: Optional[int] = None
    device: str = field(default_factory=detect_device)

    # Resume
    load_path: Optional[str] = None

    # ------------------------------------------------------------------
    # JSON loader
    # ------------------------------------------------------------------
    @classmethod
    def from_json_dict(
        cls,
        json_or_algo: Any = None,
        **overrides: Any,
    ) -> "DQNTrainingConfig":
        """
        Build a config from the JSON file + per-run overrides.

        Two call styles are supported for convenience:

        1. ``DQNTrainingConfig.from_json_dict()``
           — loads ``game/model/configs/dqn_config.json`` with no overrides.
        2. ``DQNTrainingConfig.from_json_dict("dqn", level=3, obs_type="12bit")``
           — loads the same JSON and applies the overrides on top.
        3. ``DQNTrainingConfig.from_json_dict({...})``
           — treats the first arg as a pre-loaded config dict; useful for
           unit tests that want to skip the file read.

        Any kwargs override matching dataclass fields win over the JSON
        value (so the TUI can override ``level`` / ``obs_type`` /
        ``total_timesteps`` per-run without editing the JSON).
        """
        if isinstance(json_or_algo, dict):
            base: Dict[str, Any] = dict(json_or_algo)
        elif json_or_algo is None or json_or_algo == "dqn":
            base = load_dqn_config("dqn")
        elif isinstance(json_or_algo, (str, os.PathLike)):
            base = load_dqn_config("dqn", path=json_or_algo)
        else:
            raise TypeError(
                f"Unsupported first argument: {type(json_or_algo).__name__}. "
                f"Expected dict, 'dqn', or a path."
            )
        base.update(overrides)
        return cls(**base)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def build_dqn(
    env: VecEnv,
    config: DQNTrainingConfig,
    tensorboard_log: Optional[os.PathLike] = None,
) -> DQN:
    """Construct (but do not train) a DQN agent."""
    # Pick the right policy-kwargs factory based on the chosen obs_type.
    # The 12-bit path uses an MLP; both spatiotemporal variants reuse
    # the CNN+attention encoder (which auto-detects the channel count
    # from the obs space shape, so 4- and 8-channel layouts both work).
    if config.obs_type == "12bit":
        policy_kwargs = make_dqn_12bit_kwargs(
            hidden_dim=config.hidden_dim,
            features_dim=config.features_dim,
        )
    else:
        # "spatiotemporal" or "spatiotemporal_legacy"
        policy_kwargs = make_dqn_sptmp_kwargs(
            cnn_channels=config.cnn_channels,
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_attention=config.use_attention,
        )

    # Build the LR schedule (constant or linear) using the shared helper
    # from :mod:`game.train.utility`. Keeps PPO and DQN behaviour in
    # lock-step — both algorithms get the same late-training decay.
    lr_schedule = build_lr_schedule(
        learning_rate=config.learning_rate,
        use_linear=config.use_linear_schedule,
        end_fraction=config.lr_end_fraction,
    )

    if config.load_path:
        # Resume: SB3 restores policy + optimizer + replay buffer.
        model = DQN.load(
            config.load_path,
            env=env,
            device=config.device,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
            # ``custom_objects`` overrides the saved learning_rate float
            # so unpickling works; we then install our own schedule below.
            custom_objects={"learning_rate": config.learning_rate},
        )
        # SB3 stores the LR as ``model.lr_schedule`` (a callable queried
        # every gradient step). Override it here so resumed runs use
        # the same linear-decay schedule as fresh runs — this is what
        # stops the late-training oscillation you saw around 500k steps.
        model.lr_schedule = lr_schedule
        return model

    return DQN(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        tau=config.tau,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_initial_eps=config.exploration_initial_eps,
        exploration_final_eps=config.exploration_final_eps,
        verbose=1,
        seed=config.seed,
        device=config.device,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def train_dqn(
    config: DQNTrainingConfig,
    progress_bar: bool = True,
) -> Tuple[DQN, Path]:
    """
    Full DQN training pipeline:

    1. Build a vec env (``DummyVecEnv`` if ``n_envs==1`` else ``SubprocVecEnv``).
    2. Optionally resume from ``config.load_path``.
    3. Wire up a :class:`CheckpointCallback`.
    4. ``model.learn(...)`` for ``config.total_timesteps`` steps.
    5. Save the final model to a uniquely-named ``.zip`` whose name
       embeds the obs_type token (``12bit``, ``sptmp``, ``sptmp_lgcy``).
    6. Close the env and return ``(model, final_path)``.
    """
    # --- 1. Env -----------------------------------------------------------
    # DQN can use multi-env (capped at config.max_n_envs for off-policy
    # stability); the replay buffer shuffles transitions across workers.
    n_envs = max(1, min(config.n_envs, config.max_n_envs))
    env = make_vec_env(
        level=config.level,
        obs_type=config.obs_type,
        n_envs=n_envs,
        seed=config.seed,
    )

    # --- 2. Logger directory ---------------------------------------------
    # The logger dir embeds the obs_type token too, so TensorBoard runs
    # for 12-bit and spatiotemporal variants never collide.
    tb_dir = resolve_logger_dir(
        prefix=config.output_prefix,
        algo="dqn",
        level=config.level,
        obs_type=config.obs_type,
    )

    # --- 3. Model ---------------------------------------------------------
    model = build_dqn(env=env, config=config, tensorboard_log=tb_dir)

    # --- 4. Checkpoint callback -----------------------------------------
    checkpoint_dir = Path(tb_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, config.checkpoint_freq),
        save_path=str(checkpoint_dir),
        name_prefix=f"{config.output_prefix}_dqn_level{config.level}",
        save_replay_buffer=True,    # DQN: keep replay buffer so resume works
        save_vecnormalize=False,
    )

    # --- 5. Train --------------------------------------------------------
    # Custom tqdm bar (see RewardProgressBarCallback) shows mean episode
    # reward read from `model.ep_info_buffer`. Disable SB3's built-in bar
    # so we don't render two progress lines at once.
    callbacks = [checkpoint_cb]

    # Snake-specific rollout metrics — see ``RolloutMetricsCallback``.
    # Always on; DQN's off-policy loop flushes these at the same
    # cadence as SB3's built-in ``ep_rew_mean`` (gated by
    # ``log_interval``, default every 4 episodes).
    callbacks.append(RolloutMetricsCallback())

    if progress_bar:
        callbacks.append(
            RewardProgressBarCallback(
                total_timesteps=config.total_timesteps,
                desc=f"DQN {config.obs_type} level{config.level}",
                total_episodes=config.total_episodes,
            )
        )

    # Demo Mode (PRD §"Mode Pelatihan"): when n_envs == 1 the env runs
    # in-process, so we can attach a Pygame window to watch the agent
    # play in real time. SubprocVecEnv (n_envs > 1) can't render from
    # the parent — the callback silently no-ops in that case, but we
    # also gate it here to keep the callback list lean.
    if n_envs == 1:
        callbacks.append(
            TrainingRenderCallback(
                fps=30,
                window_title=f"DQN {config.obs_type} level{config.level} — Training",
            )
        )

    try:
        # Linear LR schedule needs ``progress_remaining`` to walk
        # 1.0 → 0.0 over the new run, which requires
        # ``reset_num_timesteps=True`` on resume. Same rationale as PPO
        # (see ``train_ppo``). The exploration schedule's
        # ``initial_eps=1.0`` is safely clamped by SB3's formula at the
        # resume boundary — epsilon stays at ``final_eps`` for the
        # entire resumed run, so we don't need to override that too.
        reset_timesteps = (config.load_path is None) or config.use_linear_schedule
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            tb_log_name="dqn_run",
            reset_num_timesteps=reset_timesteps,
            progress_bar=False,
        )
    finally:
        env.close()

    # --- 6. Final save --------------------------------------------------
    final_path = auto_naming(
        prefix=config.output_prefix,
        algo="dqn",
        level=config.level,
        obs_type=config.obs_type,
    )
    model.save(str(final_path))

    return model, final_path
