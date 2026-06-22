"""
DQN Spatiotemporal Feature Extractor
====================================

DQN counterpart of :mod:`game.model.ppo_spatiotemporal`. Same CNN
spatial encoder + Transformer temporal-attention stack, but the
features extractor feeds a single Q-value head (one Linear from
``features_dim`` → ``n_actions``) instead of PPO's split actor/critic
heads.

Reuses :class:`game.model.ppo_spatiotemporal.SpatiotemporalExtractor`
unchanged so the spatial features are directly comparable between the
two algorithms. The only thing this module adds is the Q-head wiring
— ``net_arch=[]`` tells SB3 to NOT insert extra hidden layers between
the features vector and the Q-head, matching the DQN paper baseline
("linear head on top of the feature extractor").

Pipeline
--------
    Input tensor (B, C, 20, 20)  — C is read from ``observation_space.shape[0]``
        │  C=4 (honest layout, current default) : Wall | DecayingBody
        │                                          | StaticFood | DynamicMomentum
        ▼
    +-----------------------------+
    |  SPATIAL EXTRACTOR (CNN)    |  2 × Conv2d(ReLU)
    +-----------------------------+
        │  (B, cnn_channels, 20, 20)
        ▼
    +-----------------------------+
    |  TEMPORAL ATTENTION         |  1 × TransformerEncoder
    |  (Multi-Head Attention)     |
    +-----------------------------+
        │  (B, 400, d_model) → CLS-pool → (B, d_model)
        ▼
    Context vector  (features_dim = d_model)
        │
        ▼
    +-----------------------------+
    |  Q-VALUE HEAD (SB3)         |
    |-----------------------------|
    | - Linear(features_dim → 4)  |
    +-----------------------------+
        │  one Q-value per discrete action
        ▼
    [argmax(Q)]  →  action ∈ {UP, RIGHT, DOWN, LEFT}

The previous v2 8-channel layout (which included head direction, food
direction, relative danger, and broadcast snake length as separate
channels) has been replaced by the 4-channel honest layout — see
:class:`game.model.ppo_spatiotemporal.SpatiotemporalExtractor` for
the full rationale.

Why share the SpatiotemporalExtractor with PPO?
----------------------------------------------
The whole point of this repo is PPO vs DQN on the same Snake task.
If the two architectures extract features in different ways the
comparison is contaminated — DQN's edge could come from a smarter
encoder, not from the algorithm. Reusing the exact same CNN+attention
stack means any measured performance difference is attributable to
PPO's on-policy trust region vs DQN's replay-buffer + target Q
learning.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

# Reuse the PPO SpatiotemporalExtractor unchanged — same spatial
# features, same attention block, same CLS pooling. SB3's DQNPolicy
# only requires a ``BaseFeaturesExtractor`` whose output is a flat
# (B, features_dim) tensor, which SpatiotemporalExtractor already
# produces.
from game.model.ppo_spatiotemporal import SpatiotemporalExtractor


# ----------------------------------------------------------------------
# Convenience factory for DQN.policy_kwargs
# ----------------------------------------------------------------------
def make_dqn_policy_kwargs(
    cnn_channels: int = 32,
    d_model: int = 64,
    n_heads: int = 8,
    dropout: float = 0.0,
    use_attention: bool = True,
    activation_fn: type = nn.ReLU,
) -> dict:
    """
    Build the ``policy_kwargs`` dict expected by ``DQN(policy_kwargs=...)``.

    The spatiotemporal encoder is the only "head" we use, so
    ``net_arch=[]`` tells SB3 to attach a single Linear
    (``features_dim=d_model`` → ``n_actions=4``) as the Q-value head
    — no extra hidden layers between features and Q-values, matching
    the DQN paper baseline architecture exactly.

    Args:
        cnn_channels:  Output channels of the two Conv2d layers.
        d_model:       Transformer embedding / hidden size (also
                       becomes ``features_dim``).
        n_heads:       Number of attention heads. SB3 requires
                       ``d_model % n_heads == 0``.
        dropout:       Dropout inside the transformer layer.
        use_attention: If False, the transformer is bypassed (returns
                       the flattened CNN output directly). Enables the
                       "architecture ablation" experiment (CNN-only vs
                       CNN+Attention) for DQN, mirroring the PPO one.
        activation_fn: Non-linearity for SB3's Q-head Linear (the
                       features extractor hard-codes ReLU).
    """
    return {
        "features_extractor_class": SpatiotemporalExtractor,
        "features_extractor_kwargs": {
            "cnn_channels": cnn_channels,
            "d_model": d_model,
            "n_heads": n_heads,
            "dropout": dropout,
            "use_attention": use_attention,
        },
        # Empty list = no extra hidden layers between features and Q-head.
        # SB3's DQNPolicy then adds a single Linear(features_dim → n_actions).
        "net_arch": [],
        "activation_fn": activation_fn,
    }
