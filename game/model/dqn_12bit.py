"""
DQN 12-bit Baseline Feature Extractor
=====================================

Baseline MLP architecture replicating the Deep Q-Snake paper (12-bit
feature vector). Built as a `BaseFeaturesExtractor` so it slots into
stable-baselines3 DQN via `policy_kwargs`.

Pipeline
--------
    Input vector (B, 12)
        │  Bits 0-3  : obstacle (wall/body) in [UP, RIGHT, DOWN, LEFT]
        │  Bits 4-7  : body proximity (1 step ahead)
        │  Bits 8-11 : relative food direction (signs)
        ▼
    +-----------------------------+
    |  DENSE NETWORK (MLP BASE)   |
    |-----------------------------|
    | - Linear(12 → 64) + ReLU    |
    | - Linear(64 → 64) + ReLU    |
    | - Linear(64 → 64) + ReLU    |
    +-----------------------------+
        │  (B, 64)  context vector
        ▼
    +-----------------------------+
    |        Q-VALUE HEAD         |
    |-----------------------------|
    | - Linear(64 → 4)            |
    +-----------------------------+
        │  one Q-value per discrete action
        ▼
    [argmax(Q)]  →  action ∈ {UP, RIGHT, DOWN, LEFT}

Why pure MLP?
-------------
The 12-bit vector is the *compressed* representation used in the
literature baseline — it deliberately discards geometric detail to
test whether richer state (4×20×20 tensor) buys better performance.
This module intentionally contains no convolution or attention.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DQN12BitExtractor(BaseFeaturesExtractor):
    """
    3-layer MLP matching the Deep Q-Snake 12-bit baseline.

    Args:
        observation_space: SB3 observation space (Box of shape (12,)).
        hidden_dim:        Width of each of the 3 hidden Dense layers.
        features_dim:      Width of the final context vector fed to
                           the Q-value head. Defaults to ``hidden_dim``.
    """

    def __init__(
        self,
        observation_space,
        hidden_dim: int = 64,
        features_dim: Optional[int] = None,
    ):
        if features_dim is None:
            features_dim = hidden_dim
        super().__init__(observation_space, features_dim=features_dim)

        in_dim = int(observation_space.shape[0])

        # 3 Dense (Linear + ReLU) layers per the paper baseline.
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: Tensor of shape (B, 12), float32.
        Returns:
            Context vector of shape (B, features_dim).
        """
        return self.mlp(observations)


# ----------------------------------------------------------------------
# Convenience factory for DQN.policy_kwargs
# ----------------------------------------------------------------------
def make_dqn_policy_kwargs(
    hidden_dim: int = 64,
    features_dim: Optional[int] = None,
    activation_fn: type = nn.ReLU,
) -> dict:
    """
    Build the ``policy_kwargs`` dict expected by ``DQN(policy_kwargs=...)``.

    The 12-bit MLP lives entirely in the features extractor (3 layers),
    so we set ``net_arch=[]`` so SB3 only attaches a single Linear
    (features → n_actions) as the Q-value head. That matches the
    paper baseline architecture exactly.
    """
    if features_dim is None:
        features_dim = hidden_dim
    return {
        "features_extractor_class": DQN12BitExtractor,
        "features_extractor_kwargs": {
            "hidden_dim": hidden_dim,
            "features_dim": features_dim,
        },
        # Empty list = no extra hidden layers between features and Q-head.
        "net_arch": [],
        "activation_fn": activation_fn,
    }