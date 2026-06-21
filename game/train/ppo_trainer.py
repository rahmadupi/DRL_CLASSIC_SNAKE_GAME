"""
PPO Trainer
===========

Builds a :class:`stable_baselines3.PPO` agent supporting BOTH the
spatiotemporal (8×20×20) CNN+Transformer encoder AND the 12-bit
flat-vector baseline.

Hyperparameter defaults live in
``game/model/configs/ppo_config.json`` and are loaded by
:meth:`PPOTrainingConfig.from_json_dict`. The TUI launcher passes
per-run overrides (level, obs_type, total_timesteps, …).

Usage::

    from game.train.ppo_trainer import train_ppo, PPOTrainingConfig

    # Spatiotemporal run (default)
    config = PPOTrainingConfig.from_json_dict(
        level=1,
        obs_type="spatiotemporal",
    )

    # 12-bit MLP run
    config = PPOTrainingConfig.from_json_dict(
        level=2,
        obs_type="12bit",
        learning_rate=5e-4,
    )

    model, saved_path = train_ppo(config)

    # Continue learning (curriculum)
    config2 = PPOTrainingConfig.from_json_dict(
        level=3,
        obs_type="spatiotemporal",
        total_timesteps=500_000,
        load_path=saved_path,
    )
    model2, _ = train_ppo(config2)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv

from game.env.game_environment import game_environment  # noqa: F401  (re-exported)
from game.model.configs import load_config as load_ppo_config
from game.model.ppo_12bit import make_ppo_policy_kwargs as make_ppo_12bit_kwargs
from game.model.ppo_spatiotemporal import (
    SpatiotemporalExtractor,
    make_ppo_policy_kwargs as make_ppo_sptmp_kwargs,
)
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
class PPOTrainingConfig:
    """All knobs the PPO trainer cares about."""

    # Curriculum
    level: int = 1
    obs_type: str = "spatiotemporal"
    # Accepted values:
    #   "spatiotemporal"        — 8×20×20 tensor       (CNN+Attention)
    #   "spatiotemporal_legacy" — 4×20×20 tensor       (CNN+Attention)
    #   "12bit"                 — flat 12-dim vector    (MLP)

    # Parallelism
    # 4 envs is the sweet spot for Snake-on-PPO: enough decorrelation to
    # cut per-iter variance, small enough to keep iteration wall-time low.
    # TrainingRenderCallback auto-disables at n_envs > 1 (the demo window
    # only makes sense with a single in-process env), which also restores
    # fps after switching from n_envs=1.
    n_envs: int = 1

    # Schedule
    total_timesteps: int = 500_000
    # Optional user-facing episode target. When set, the progress bar
    # shows ``eps/total_episodes`` instead of just the running count.
    total_episodes: Optional[int] = None
    learning_rate: float = 1e-3       # conservative default; 1e-3 1e-4 7e-4
    # With n_envs=4, n_steps=2048 → 8192 transitions/rollout, split into
    # 32 minibatches of 256 → 160 gradient steps per iter. Same update
    # budget as the old (n_envs=1, n_steps=4096, batch_size=128), but
    # with 4× more decorrelated experience feeding GAE.
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 9
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.018
    vf_coef: float = 0.4
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
    lr_end_fraction: float = 0.1
    clip_end_fraction: float = 0.05

    # Architecture — 12-bit MLP path
    hidden_dim: int = 64

    # Architecture — spatiotemporal (CNN+attention) path
    # Defaults at (32, 64, 8) — head_dim = 64/8 = 8, healthy.
    # Total params: ~75k (well-matched to level-1 Snake's 20×20 grid).
    # The earlier (64, 128) was 4× larger; at iter 1–4 we saw
    # ``entropy_loss ≈ -ln(4)`` for four iters straight, suggesting the
    # policy was still uniform — extra capacity wasn't helping and just
    # slows each iteration. Bump back to (64, 128) only if level ≥ 3 or
    # if dynamic-food evasion starts to underfit.
    cnn_channels: int = 32
    d_model: int = 64
    n_heads: int = 8
    dropout: float = 0.1
    use_attention: bool = True         # toggle for the ablation experiment

    # Actor/critic post-extractor MLP widths. Defaults match the
    # spatiotemporal recipe (single 64-unit hidden layer before the
    # 4-logit / 1-scalar head). Only consulted when ``obs_type="12bit"``
    # AND the JSON provides them; spatiotemporal passes them through to
    # :func:`make_ppo_sptmp_kwargs` for the actor-critic policy.
    net_arch_pi: Optional[list] = field(default_factory=lambda: [64])
    net_arch_vf: Optional[list] = field(default_factory=lambda: [64])

    # Output
    output_prefix: str = "snake"
    checkpoint_freq: int = 50_000
    seed: Optional[int] = None
    device: str = field(default_factory=detect_device)

    # Resume from an existing checkpoint (for curriculum continuous-learning)
    load_path: Optional[str] = None

    # ------------------------------------------------------------------
    # JSON loader
    # ------------------------------------------------------------------
    @classmethod
    def from_json_dict(
        cls,
        json_or_algo: Any = None,
        **overrides: Any,
    ) -> "PPOTrainingConfig":
        """
        Build a config from the JSON file + per-run overrides.

        Call styles (mirrors :meth:`DQNTrainingConfig.from_json_dict`):

        1. ``PPOTrainingConfig.from_json_dict()``
        2. ``PPOTrainingConfig.from_json_dict("ppo", level=3, obs_type="12bit")``
        3. ``PPOTrainingConfig.from_json_dict({...})``
        """
        if isinstance(json_or_algo, dict):
            base: Dict[str, Any] = dict(json_or_algo)
        elif json_or_algo is None or json_or_algo == "ppo":
            base = load_ppo_config("ppo")
        elif isinstance(json_or_algo, (str, os.PathLike)):
            base = load_ppo_config("ppo", path=json_or_algo)
        else:
            raise TypeError(
                f"Unsupported first argument: {type(json_or_algo).__name__}. "
                f"Expected dict, 'ppo', or a path."
            )
        base.update(overrides)
        return cls(**base)


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
    # Pick the right policy-kwargs factory based on the chosen obs_type.
    # Spatiotemporal variants share the CNN+attention encoder
    # (4- and 8-channel layouts both work — the extractor reads the
    # channel count from the obs space shape). 12-bit uses an MLP.
    if config.obs_type == "12bit":
        policy_kwargs = make_ppo_12bit_kwargs(
            hidden_dim=config.hidden_dim,
            net_arch_pi=config.net_arch_pi,
            net_arch_vf=config.net_arch_vf,
        )
    else:
        # "spatiotemporal" or "spatiotemporal_legacy"
        policy_kwargs = make_ppo_sptmp_kwargs(
            cnn_channels=config.cnn_channels,
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_attention=config.use_attention,
            net_arch=dict(
                pi=list(config.net_arch_pi or [64]),
                vf=list(config.net_arch_vf or [64]),
            ),
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
    5. Save the final model to a uniquely-named ``.zip`` whose name
       embeds the obs_type token (``12bit``, ``sptmp``, ``sptmp_lgcy``).
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
    # The logger dir embeds the obs_type token too, so TensorBoard runs
    # for 12-bit and spatiotemporal variants never collide.
    tb_dir = resolve_logger_dir(
        prefix=config.output_prefix,
        algo="ppo",
        level=config.level,
        obs_type=config.obs_type,
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

    # Snake-specific rollout metrics — always on, no toggle. Cheap (a
    # handful of dict accesses and arithmetic per rollout), and the
    # ``snake_length`` curve is the single most useful diagnostic for
    # Snake training (much smoother than ``ep_rew_mean``, which is
    # dominated by ±10 eat/collision spikes). See
    # ``RolloutMetricsCallback`` for the full list of scalars logged.
    callbacks.append(RolloutMetricsCallback())

    if progress_bar:
        callbacks.append(
            RewardProgressBarCallback(
                total_timesteps=config.total_timesteps,
                desc=f"PPO {config.obs_type} level{config.level}",
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
                window_title=f"PPO {config.obs_type} level{config.level} — Training",
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
        obs_type=config.obs_type,
    )
    model.save(str(final_path))

    return model, final_path
