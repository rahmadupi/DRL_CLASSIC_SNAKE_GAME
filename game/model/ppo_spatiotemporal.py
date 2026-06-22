"""
Spatiotemporal PPO Feature Extractor
====================================

Proposed PPO architecture for the Snake environment. Built as a
`BaseFeaturesExtractor` so it slots directly into stable-baselines3
PPO via the `policy_kwargs={"features_extractor_class": ...}` hook.

Pipeline
--------
    Input tensor (B, C, 20, 20)  — C is read from ``observation_space.shape[0]``
        │  C=4 (honest layout, current default) : Wall | DecayingBody
        │                                          | StaticFood | DynamicMomentum
        ▼
    +-----------------------------+
    |  SPATIAL EXTRACTOR (CNN)    |  2 × Conv2d(ReLU) + Flatten
    +-----------------------------+
        │  (B, cnn_channels, 20, 20) flattened → (B, cnn_channels, 400)
        ▼
    +-----------------------------+
    |  TEMPORAL ATTENTION         |  1 × TransformerEncoder
    |  (Multi-Head Attention)     |
    +-----------------------------+
        │  (B, 400, d_model) → CLS-pool → (B, d_model)
        ▼
    Context vector  (features_dim = d_model)
        │
        │  ╔════════════════════════════════════════════════╗
        │  ║  Appended by stable-baselines3 (ActorCriticPolicy):
        │  ║  • mlp_extractor.policy_net  (Linear 64→64 + ReLU)
        │  ║  • mlp_extractor.value_net   (Linear 64→64 + ReLU)
        │  ║  • action_net                (Linear 64 → 4 logits)  ← ACTOR HEAD
        │  ║  • value_net                 (Linear 64 → 1 scalar)  ← CRITIC HEAD
        │  ╚════════════════════════════════════════════════╝
        │
        ├──► Actor head  → 4 action logits  (UP, RIGHT, DOWN, LEFT)
        └──► Critic head → 1 state value estimate

What was removed from the v2 layout (and why)?
----------------------------------------------
The previous v2 layout exposed 8 channels: the four honest channels
above plus head direction (Ch4), food direction (Ch5), relative danger
(Ch6), and a broadcast snake length (Ch7). Those were heuristic
crutches — they let the network read the answers instead of learning
the underlying geometry from the raw channels:

* Head direction    — deducible from Ch1's decay gradient.
* Food direction    — deducible from Ch3's momentum + Ch2's static cells.
* Relative danger   — deducible from Ch0 (walls) + Ch1 (body).
* Snake length      — deducible from Ch1's body extent (or implicitly
                       via the rollout signal).

Removing them is what makes the PPO↔DQN and 12-bit↔spatiotemporal
comparisons measure the algorithms, not the cheat-sheet.

Why a Transformer over a flat MLP?
----------------------------------
* The 4-channel grid encodes *where* obstacles and food are *right now*.
  The transformer allows the actor to weigh *which spatial locations*
  are most informative given the rest of the grid (e.g. food cells
  adjacent to the head, body segments forming a choke-point).
* A learnable [CLS]-style token is prepended so pooling has a
  dedicated attention sink rather than averaging noisy spatial tokens.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpatiotemporalExtractor(BaseFeaturesExtractor):
    """
    CNN spatial encoder + Transformer temporal attention.

    Args:
        observation_space: SB3 observation space (Box of shape (C, 20, 20)).
                           The number of input channels ``C`` is read from
                           ``observation_space.shape[0]`` and is forwarded
                           to the first Conv2d — currently always 4 for
                           the honest layout.
        cnn_channels:      Output channels of the two Conv2d layers.
        d_model:           Transformer embedding / hidden size.
        n_heads:           Number of attention heads.
        dropout:           Dropout inside the transformer layer.
        use_attention:     If False, the transformer is bypassed (returns
                           the flattened CNN output directly). Enables the
                           "architecture ablation" experiment described
                           in the PRD (CNN-only vs. CNN+Attention).
    """

    def __init__(
        self,
        observation_space,
        cnn_channels: int = 64,
        d_model: int = 128,
        n_heads: int = 8,
        dropout: float = 0.0,
        use_attention: bool = True,
    ):
        # The grid is 20×20, C input channels (currently always 4) →
        # after two stride-1 convs the spatial size is preserved, so the
        # flattened length is cnn_channels*20*20 regardless of C.
        super().__init__(observation_space, features_dim=d_model)

        self.use_attention = use_attention
        self.grid_h: int = observation_space.shape[1]
        self.grid_w: int = observation_space.shape[2]
        # Read the input channel count from the obs space so the same
        # extractor works for any C-channel honest layout (currently
        # always 4). SB3 only constructs the extractor once per env,
        # so the mismatch failure mode for a wrong-shape model is
        # caught at load time, not silently mid-training.
        self.in_channels: int = int(observation_space.shape[0])
        self.cnn_channels: int = cnn_channels
        self.d_model: int = d_model

        # ----------------------------------------------------------------
        # 1. Spatial Extractor — two Conv2d + ReLU, no pooling (preserve grid)
        # ----------------------------------------------------------------
        self.spatial = nn.Sequential(
            nn.Conv2d(self.in_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ----------------------------------------------------------------
        # 2. Temporal Attention — 1 Transformer encoder block
        # ----------------------------------------------------------------
        if use_attention:
            # Project each spatial cell into d_model dims
            self.cell_embed = nn.Linear(cnn_channels, d_model)

            # Learnable [CLS]-style sink token prepended to the sequence
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="relu",
                norm_first=True,  # Pre-LN — more stable for small models
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=1,
                # norm_first=True is incompatible with the nested-tensor
                # fast path (PyTorch ≥ 2.0). Disabling it silences the
                # UserWarning emitted by TransformerEncoder.__init__.
                enable_nested_tensor=False,
            )

            # Final projection keeps features_dim == d_model
            self.out_proj = nn.Linear(d_model, d_model)
        else:
            # Ablation path — flatten CNN output straight into a d_model vec
            self.cell_embed = None
            self.cls_token = None
            self.transformer = None
            self.out_proj = nn.Sequential(
                nn.Linear(cnn_channels * self.grid_h * self.grid_w, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model),
            )

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: Tensor of shape (B, C, 20, 20), float32, where
                          ``C`` matches the env's obs space (currently
                          always 4 — Wall, Decaying body, Static food,
                          Dynamic food momentum).
        Returns:
            Context vector of shape (B, features_dim).
        """
        # 1. Spatial encoding
        x = self.spatial(observations)              # (B, cnn_channels, H, W)
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, H*W, C)

        if not self.use_attention:
            return self.out_proj(tokens.reshape(b, -1))

        # 2. Embed to d_model and prepend CLS token
        tokens = self.cell_embed(tokens)             # (B, H*W, d_model)
        cls = self.cls_token.expand(b, -1, -1)       # (B, 1, d_model)
        seq = torch.cat([cls, tokens], dim=1)        # (B, 1 + H*W, d_model)

        # 3. Self-attention over the spatial+CLS sequence
        attended = self.transformer(seq)             # (B, 1 + H*W, d_model)

        # 4. Pool — use the CLS token as the context vector
        context = attended[:, 0, :]                  # (B, d_model)
        return self.out_proj(context)


# ----------------------------------------------------------------------
# Convenience factory for PPO.policy_kwargs
# ----------------------------------------------------------------------
def make_ppo_policy_kwargs(
    cnn_channels: int = 32,
    d_model: int = 64,
    n_heads: int = 4,
    dropout: float = 0.0,
    use_attention: bool = True,
    net_arch: Optional[list] = None,
    activation_fn: type = nn.ReLU,
) -> dict:
    """
    Build the `policy_kwargs` dict expected by `PPO(policy_kwargs=...)`.

    Default net_arch feeds the context vector through a single 64-unit
    hidden layer before the actor/critic heads — light enough for CPU
    training of a 20×20 grid.
    """
    if net_arch is None:
        # SB3 ≥ 1.8 prefers a flat dict (no list wrapper). The hidden
        # layers here are the LAST stage before the actor/critic heads:
        #   features_dim (64) → 64 → 4 logits   (actor)
        #   features_dim (64) → 64 → 1 scalar   (critic)
        net_arch = dict(pi=[64], vf=[64])
    return {
        "features_extractor_class": SpatiotemporalExtractor,
        "features_extractor_kwargs": {
            "cnn_channels": cnn_channels,
            "d_model": d_model,
            "n_heads": n_heads,
            "dropout": dropout,
            "use_attention": use_attention,
        },
        "net_arch": net_arch,
        "activation_fn": activation_fn,
    }