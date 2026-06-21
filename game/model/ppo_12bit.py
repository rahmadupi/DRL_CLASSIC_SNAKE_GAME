"""
PPO 12-bit Feature Extractor
============================

PPO variant of the Deep Q-Snake 12-bit baseline. The 12-dim feature
vector is small enough that a plain MLP captures every bit; convolutions
and attention would just add parameters without buying representational
power on a 12-dim input.

The feature-extractor class itself is the same as
:class:`game.model.dqn_12bit.DQN12BitExtractor` — a 3-layer MLP with
matching dimensions — so the two algorithms have a directly-comparable
"compressed" representation. Only the policy kwargs differ: PPO uses
``ActorCriticPolicy`` which splits the network into separate actor and
critic heads, so ``net_arch`` becomes a dict with ``pi`` / ``vf`` keys
instead of an empty list.

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
    |  ACTOR-CRITIC HEADS (SB3)   |
    |-----------------------------|
    | - mlp_extractor.policy_net  (Linear 64 → 64 + ReLU)
    | - mlp_extractor.value_net   (Linear 64 → 64 + ReLU)
    | - action_net                (Linear 64 → 4 logits)   ← ACTOR
    | - value_net                 (Linear 64 → 1 scalar)   ← CRITIC
    +-----------------------------+
        │
        ├──► Actor  → 4 action logits  (UP, RIGHT, DOWN, LEFT)
        └──► Critic → 1 state value estimate
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Reuse the DQN baseline's MLP — same architecture, same input shape,
# same hidden size. Keeps the PPO↔DQN comparison apples-to-apples on
# the "compressed 12-bit" representation.
from game.model.dqn_12bit import DQN12BitExtractor


# ----------------------------------------------------------------------
# Convenience factory for PPO.policy_kwargs
# ----------------------------------------------------------------------
def make_ppo_policy_kwargs(
    hidden_dim: int = 64,
    features_dim: Optional[int] = None,
    net_arch_pi: Optional[list] = None,
    net_arch_vf: Optional[list] = None,
    activation_fn: type = nn.ReLU,
) -> dict:
    """
    Build the ``policy_kwargs`` dict expected by ``PPO(policy_kwargs=...)``.

    The 12-bit MLP lives entirely in the features extractor (3 layers),
    so we put a single 64-unit hidden layer BEFORE the actor/critic
    heads. PPO's ``ActorCriticPolicy`` expects ``net_arch`` as a dict
    with ``pi`` (policy) and ``vf`` (value) keys; the defaults here
    mirror what the SpatiotemporalExtractor uses on the 8-channel path.

    Args:
        hidden_dim:   Width of the 3 Dense layers inside the features
                      extractor and the post-extractor actor/critic
                      MLP blocks.
        features_dim: Width of the final context vector fed to the
                      actor/critic MLPs. Defaults to ``hidden_dim``.
        net_arch_pi:  List of hidden-layer widths for the actor MLP
                      (after features, before the 4-logit head).
                      ``None`` → ``[hidden_dim]``.
        net_arch_vf:  Same, for the critic MLP. ``None`` → ``[hidden_dim]``.
        activation_fn:Non-linearity used by SB3's actor/critic MLPs.
                      The features extractor hard-codes ``nn.ReLU``
                      (matches the Deep Q-Snake baseline); this only
                      affects SB3's appended layers.
    """
    if features_dim is None:
        features_dim = hidden_dim
    if net_arch_pi is None:
        net_arch_pi = [hidden_dim]
    if net_arch_vf is None:
        net_arch_vf = [hidden_dim]

    return {
        "features_extractor_class": DQN12BitExtractor,
        "features_extractor_kwargs": {
            "hidden_dim": hidden_dim,
            "features_dim": features_dim,
        },
        # SB3 ≥ 1.8 prefers a flat dict (no list wrapper) so the actor
        # and critic heads get their own hidden stack instead of
        # sharing one. Defaults match the SpatiotemporalExtractor
        # recipe: features_dim → 64 → 4 logits (actor) and features_dim
        # → 64 → 1 scalar (critic).
        "net_arch": dict(pi=list(net_arch_pi), vf=list(net_arch_vf)),
        "activation_fn": activation_fn,
    }
