"""
PPO Trainer
===========

Builds a :class:`stable_baselines3.PPO` agent wired with the
:class:`SpatiotemporalExtractor`, the vectorised env, the
:class:`CheckpointCallback`, and the TensorBoard logger — all from a
single ``train_ppo(...)`` call.

Usage::

    from game.train.ppo_trainer import train_ppo, PPOTrainingConfig

    config = PPOTrainingConfig(
        level=1,
        n_envs=4,
        total_timesteps=500_000,
        learning_rate=1e-4,        # 1e-4 (conservative) vs 5e-4 (aggressive)
        checkpoint_freq=50_000,
    )
    model, saved_path = train_ppo(config)

    # Continue learning (curriculum)
    config2 = PPOTrainingConfig(
        level=2,
        n_envs=4,
        total_timesteps=500_000,
        load_path=saved_path,
    )
    model2, _ = train_ppo(config2)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv

from game.env.game_environment import game_environment  # noqa: F401  (re-exported)
from game.model.ppo_spatiotemporal import (
    SpatiotemporalExtractor,
    make_ppo_policy_kwargs,
)
from game.train.utility import (
    RewardProgressBarCallback,
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
class PPOTrainingConfig:
    """All knobs the PPO trainer cares about."""

    # Curriculum
    level: int = 1
    obs_type: str = "spatiotemporal"   # PPO uses the 8-channel v2 tensor
                                        # (set to "spatiotemporal_legacy" to
                                        #  load a model trained on the old
                                        #  4-channel v1 layout)

    # Parallelism
    n_envs: int = 1

    # Schedule
    total_timesteps: int = 500_000
    # Optional user-facing episode target. When set, the progress bar
    # shows ``eps/total_episodes`` instead of just the running count.
    total_episodes: Optional[int] = None
    learning_rate: float = 7e-4       # conservative default; 1e-3 1e-4
    n_steps: int = 4096 #2048
    batch_size: int = 128 #64
    n_epochs: int = 5 # 7
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.03
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Learning-rate / clip-range schedule.
    # When ``use_linear_schedule=True`` (default — recommended by the
    # original PPO paper) both ``learning_rate`` and ``clip_range``
    # decay linearly from their initial value down to
    # ``<param> * <param>_end_fraction`` over the course of training.
    # This is what stabilises the late-training oscillation you saw
    # when resuming with a constant schedule (5e-4 stayed fixed and
    # kept knocking the policy off the good region around 500k steps).
    #
    # Effect on resume:
    # * Linear schedules need ``reset_num_timesteps=True`` so that
    #   ``progress_remaining`` walks from 1.0 → 0.0 over the resumed
    #   run. ``train_ppo`` does this automatically — the TensorBoard
    #   X-axis for the new run restarts at 0, but the old run's logs
    #   remain available in their original directory.
    use_linear_schedule: bool = True
    lr_end_fraction: float = 0.1        # 0.0 → decay LR all the way to 0
    clip_end_fraction: float = 0.0      # 0.0 → decay clip all the way to 0

    # Architecture (forwarded to SpatiotemporalExtractor)
    # Defaults bumped to (64, 128, 4) — head_dim = 128/4 = 32, healthy.
    # Total params: ~280k (≈4× the previous 32/64/4 ≈ 75k).
    # Dropout=0.1 added as regularization for the larger capacity.
    cnn_channels: int = 64 # 32
    d_model: int = 128 # 64
    n_heads: int = 8
    dropout: float = 0.2
    use_attention: bool = True         # toggle for the ablation experiment

    # Output
    output_prefix: str = "snake"
    checkpoint_freq: int = 50_000
    seed: Optional[int] = None
    device: str = field(default_factory=detect_device)

    # Resume from an existing checkpoint (for curriculum continuous-learning)
    load_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def build_ppo(
    env: VecEnv,
    config: PPOTrainingConfig,
    tensorboard_log: Optional[os.PathLike] = None,
) -> PPO:
    """
    Construct (but do not train) a PPO agent.

    Exposed separately so unit tests can introspect the model without
    paying the cost of ``learn()``.
    """
    policy_kwargs = make_ppo_policy_kwargs(
        cnn_channels=config.cnn_channels,
        d_model=config.d_model,
        n_heads=config.n_heads,
        dropout=config.dropout,
        use_attention=config.use_attention,
    )

    # Build learning-rate and clip-range schedules (constant or linear).
    # Both branches are SB3-compatible callables that map
    # ``progress_remaining`` (1.0 → 0.0) to a scalar.
    lr_schedule = build_lr_schedule(
        learning_rate=config.learning_rate,
        use_linear=config.use_linear_schedule,
        end_fraction=config.lr_end_fraction,
    )
    clip_schedule = _build_clip_schedule(config)

    if config.load_path:
        # Resume: load the full agent (policy + optimizer + scaler state).
        # The env must still match the obs space the model was trained on.
        model = PPO.load(
            config.load_path,
            env=env,
            device=config.device,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
            # When resuming, SB3 re-uses the saved lr_schedule. Force the
            # new schedule so curriculum runs can change the lr per stage.
            custom_objects={"learning_rate": config.learning_rate},
        )
        # Override BOTH schedules on the loaded model. SB3 reads
        # ``model.lr_schedule`` and ``model.clip_range`` on every step
        # via ``_update_learning_rate`` / ``_setup_model``, so swapping
        # them here replaces the previously-saved schedule cleanly.
        model.lr_schedule = lr_schedule
        model.clip_range = clip_schedule
        return model

    return PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=clip_schedule,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        verbose=1,
        seed=config.seed,
        device=config.device,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
    )


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------
def _build_clip_schedule(config: "PPOTrainingConfig"):
    """Return a SB3-compatible clip-range schedule callable.

    Linear decay of ``clip_range`` alongside LR is the original PPO
    recipe — it keeps updates small as the policy matures, which is
    what stops the late-stage oscillation. The shared
    :func:`game.train.utility.build_lr_schedule` handles the LR side;
    this helper is PPO-only because ``clip_range`` doesn't exist on DQN.
    """
    from stable_baselines3.common.utils import get_schedule_fn, get_linear_fn
    if config.use_linear_schedule:
        return get_linear_fn(
            config.clip_range,
            config.clip_range * config.clip_end_fraction,
            1.0,
        )
    return get_schedule_fn(config.clip_range)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def train_ppo(
    config: PPOTrainingConfig,
    progress_bar: bool = True,
) -> Tuple[PPO, Path]:
    """
    Full PPO training pipeline:

    1. Build the vectorised env (``DummyVecEnv`` if ``n_envs==1`` else ``SubprocVecEnv``).
    2. Optionally resume from ``config.load_path``.
    3. Wire up a :class:`CheckpointCallback` that saves every
       ``checkpoint_freq`` timesteps into ``saved_models/``.
    4. ``model.learn(...)`` for ``config.total_timesteps`` steps.
    5. Save the final model to a uniquely-named ``.zip`` via :func:`auto_naming`.
    6. Close the env and return ``(model, final_path)``.
    """
    # --- 1. Env -----------------------------------------------------------
    env = make_vec_env(
        level=config.level,
        obs_type=config.obs_type,
        n_envs=config.n_envs,
        seed=config.seed,
    )

    # --- 2. Logger directory ---------------------------------------------
    tb_dir = resolve_logger_dir(
        prefix=config.output_prefix,
        algo="ppo",
        level=config.level,
    )

    # --- 3. Model ---------------------------------------------------------
    model = build_ppo(env=env, config=config, tensorboard_log=tb_dir)

    # --- 4. Checkpoint callback -----------------------------------------
    checkpoint_dir = Path(tb_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, config.checkpoint_freq // max(1, config.n_envs)),
        save_path=str(checkpoint_dir),
        name_prefix=f"{config.output_prefix}_ppo_level{config.level}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # --- 5. Train --------------------------------------------------------
    # Custom tqdm bar (see RewardProgressBarCallback) shows mean episode
    # reward read from `model.ep_info_buffer`. Disable SB3's built-in bar
    # so we don't render two progress lines at once.
    callbacks = [checkpoint_cb]
    if progress_bar:
        callbacks.append(
            RewardProgressBarCallback(
                total_timesteps=config.total_timesteps,
                desc=f"PPO level{config.level}",
                total_episodes=config.total_episodes,
            )
        )

    # Demo Mode (PRD §"Mode Pelatihan"): when n_envs == 1 the env runs
    # in-process, so we can attach a Pygame window to watch the agent
    # play in real time. SubprocVecEnv (n_envs > 1) can't render from
    # the parent — the callback silently no-ops in that case, but we
    # also gate it here to keep the callback list lean.
    if config.n_envs == 1:
        callbacks.append(
            TrainingRenderCallback(
                fps=30,
                window_title=f"PPO level{config.level} — Training",
            )
        )

    try:
        # Linear LR/clip schedules rely on ``progress_remaining`` walking
        # 1.0 → 0.0 over the new run. When resuming with a constant
        # schedule we keep the existing behaviour (cumulative step count
        # continues — nicer TensorBoard X-axis), but a linear schedule
        # needs ``reset_num_timesteps=True`` so ``num_timesteps`` starts
        # at 0 and the schedule's input is meaningful.
        reset_timesteps = (config.load_path is None) or config.use_linear_schedule
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            tb_log_name="ppo_run",
            reset_num_timesteps=reset_timesteps,
            progress_bar=False,
        )
    finally:
        env.close()

    # --- 6. Final save --------------------------------------------------
    final_path = auto_naming(
        prefix=config.output_prefix,
        algo="ppo",
        level=config.level,
    )
    model.save(str(final_path))

    return model, final_path