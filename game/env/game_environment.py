"""
Gymnasium-compatible Snake environment.

Features:
    - 20x20 grid, dual-outlet state space (4-channel tensor & 12-bit vector)
    - 5 curriculum levels (static & dynamic food targets)
    - Distance-based reward shaping + food/collision rewards
    - Speed asymmetry: snake moves 1.5x faster than dynamic food
"""

import math
import random
from collections import deque
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game.env.config import config as _cfg


# -----------------------------
# Constants (sourced from config.json)
# -----------------------------
GRID_SIZE = _cfg.GRID_SIZE
INITIAL_SNAKE_LENGTH = _cfg.INITIAL_SNAKE_LENGTH
MAX_GRID_AREA = GRID_SIZE * GRID_SIZE

# Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
ACTION_UP = _cfg.ACTION_UP
ACTION_RIGHT = _cfg.ACTION_RIGHT
ACTION_DOWN = _cfg.ACTION_DOWN
ACTION_LEFT = _cfg.ACTION_LEFT

DIRECTIONS = {
    ACTION_UP:    tuple(_cfg.DIRECTIONS["UP"]),
    ACTION_RIGHT: tuple(_cfg.DIRECTIONS["RIGHT"]),
    ACTION_DOWN:  tuple(_cfg.DIRECTIONS["DOWN"]),
    ACTION_LEFT:  tuple(_cfg.DIRECTIONS["LEFT"]),
}

# ---------------------------------------------------------------------------
# Spatial observation channel layouts
# ---------------------------------------------------------------------------
# v1 (legacy): 4 channels — wall, decaying body, static food, dynamic food.
# v2 (current): 8 channels — v1's 4 + head direction, food direction
#                (1 ch, 4 cells), relative danger (1 ch, 3 cells),
#                broadcast snake length. The dynamic-food per-cell map
#                (Ch3, current=1.0, previous=0.5) is critical on levels
#                3-4 where dynamic food is the ONLY target — without it
#                the agent only has the directional signal in Ch5 and no
#                position/long-range info to plan a path.
SPATIAL_OBS_CHANNELS_V1 = 4
SPATIAL_OBS_CHANNELS_V2 = 8

# Channel indices for v2 (8-channel) obs. Centralised so the network and
# any visualisations can refer to channels by name instead of magic ints.
CH_DYNAMIC     = 3  # 1.0 at dynamic food's current cell, 0.5 at previous
CH_HEAD_DIR    = 4  # 1.0 at the cell 1 step in current direction
CH_FOOD_DIR    = 5  # 1.0 at the 4 cells around head in directions where
                    #     ANY food exists; 0 otherwise (mirrors 12-bit Bits 8-11)
CH_DANGER_REL  = 6  # 1.0 at the 3 cells STRAIGHT/LEFT/RIGHT of head
                    #     (relative to current heading) if wall or body
CH_LENGTH      = 7  # broadcast `len(snake) / 400` everywhere

# ---------------------------------------------------------------------------
# Per-channel enable / danger-source flags (all driven from config.json)
# ---------------------------------------------------------------------------
# ``OBS_ENABLE_*`` gates whether a channel is populated at all. When a flag
# is False the corresponding channel is left at zero in the v2 obs, so the
# effective channel count shrinks — the trainer/launcher must rebuild the
# env so the observation_space (and the saved model's first conv) matches.
#
# ``DANGER_FROM_*`` splits the relative-danger channel by collision source.
# The cell is marked dangerous if ANY enabled source considers it deadly.
# Disable ``DANGER_FROM_BODY`` to make the snake ignore its own body in
# this channel (the body is still there in Ch1, so it isn't truly invisible).
#
# ``DANGER_TAIL_IS_SAFE`` = whether the deque tail is treated as a non-body
# for danger marking. The tail vacates on the next step (snake never grows
# AND moves in the same step), so it is technically safe — toggle to False
# to make the agent treat the tail as a body cell.
OBS_ENABLE_HEAD_DIRECTION = _cfg.OBS_ENABLE_HEAD_DIRECTION
OBS_ENABLE_FOOD_DIRECTION = _cfg.OBS_ENABLE_FOOD_DIRECTION
OBS_ENABLE_RELATIVE_DANGER = _cfg.OBS_ENABLE_RELATIVE_DANGER
OBS_ENABLE_SNAKE_LENGTH = _cfg.OBS_ENABLE_SNAKE_LENGTH

DANGER_FROM_WALL = _cfg.DANGER_FROM_WALL
DANGER_FROM_BODY = _cfg.DANGER_FROM_BODY
DANGER_TAIL_IS_SAFE = _cfg.DANGER_TAIL_IS_SAFE

# Distance-shaping source / target-tracking flags (see game/env/config.json).
#
# DISTANCE_INCLUDE_DYNAMIC = whether dynamic food contributes to the
# approach/away distance calc. Default False because the dynamic food
# moves 2/3 ticks, so changes in the nearest-food distance can be
# confounded with the food's own motion (snake gets approach credit
# when the food moved toward it, not when the snake did). Static food
# is stationary, so its distance is a clean shaping signal.
#
# TARGET_FOOD_ENABLED = whether the env commits to a single food cell
# as the approach target. When True, the env picks the nearest static
# food on the first step after reset (or after the current target is
# eaten) and rewards approach toward THAT food only, until it is
# consumed. This removes the "nearest-target-swap" confound where a
# different food becoming marginally closer mis-attributes approach
# credit to the snake. Set to False to fall back to the simpler
# "distance to the nearest food right now" behaviour.
DISTANCE_INCLUDE_DYNAMIC = _cfg.DISTANCE_INCLUDE_DYNAMIC
TARGET_FOOD_ENABLED = _cfg.TARGET_FOOD_ENABLED
TARGET_SWITCH_AFTER_AWAY_STEPS = _cfg.TARGET_SWITCH_AFTER_AWAY_STEPS

# Reward constants
DYNAMIC_PENALTY = _cfg.DYNAMIC_PENALTY
REWARD_APPROACH = _cfg.REWARD_APPROACH
REWARD_AWAY = _cfg.REWARD_AWAY
REWARD_EAT_STATIC = _cfg.REWARD_EAT_STATIC
REWARD_EAT_DYNAMIC = _cfg.REWARD_EAT_DYNAMIC
REWARD_COLLISION = _cfg.REWARD_COLLISION
REWARD_TIME = _cfg.REWARD_TIME

# COLLISION, EAT STATIC, EAT DYNAMIC
REWARD_MOD = list(_cfg.REWARD_MOD)
REWARD_MOD_CAP = list(_cfg.REWARD_MOD_CAP)

# Milestone bonus — modest linear growth applied ONLY when the snake
# crosses a length multiple of REWARD_MILESTONE_INTERVAL. Per-milestone
# bonus = REWARD_MILESTONE_MOD * (1 + REWARD_MILESTONE_GROWTH * index),
# so with default MOD=20 and GROWTH=0.1:
#   milestone 1 (length  5) → +22
#   milestone 2 (length 10) → +24
#   milestone 5 (length 25) → +30
#   milestone 10 (length 50) → +40
# Pushes the agent past the orbit/satisfice attractor with a tame ramp.
# All values driven from config.json.
REWARD_MILESTONE_ENABLED = _cfg.REWARD_MILESTONE_ENABLED
REWARD_MILESTONE_INTERVAL = _cfg.REWARD_MILESTONE_INTERVAL
REWARD_MILESTONE_MOD = _cfg.REWARD_MILESTONE_MOD
REWARD_MILESTONE_GROWTH = _cfg.REWARD_MILESTONE_GROWTH

# Stagnation penalty — charges the agent for steps that don't reduce
# its distance to the nearest food. Targets the orbit attractor
# (circling keeps distance ~constant) without changing terminal rewards.
STAGNATION_ENABLED = _cfg.STAGNATION_ENABLED
STAGNATION_THRESHOLD = _cfg.STAGNATION_THRESHOLD
STAGNATION_PENALTY = _cfg.STAGNATION_PENALTY

# Level configurations (JSON keys are strings → cast back to int)
LEVEL_CONFIG = {int(k): v for k, v in _cfg.LEVEL_CONFIG.items()}


class dynamic_food:
    """Dynamic food entity with stochastic momentum and evasion behavior."""

    EVASION_PROB = _cfg.EVASION_PROB
    MOMENTUM_MIN = _cfg.MOMENTUM_MIN
    MOMENTUM_MAX = _cfg.MOMENTUM_MAX

    def __init__(self, grid_size: int = GRID_SIZE, position: Tuple[int, int] = (0, 0)):
        self.grid_size = grid_size
        self.position = position
        self.prev_position = position
        # State machine: 1 = Momentum (moving straight), 2 = Rotation/Evasion
        self.state = 1
        self.remaining_momentum = random.randint(self.MOMENTUM_MIN, self.MOMENTUM_MAX)
        self.direction = random.choice(list(DIRECTIONS.values()))

    def move(self, snake_head: Tuple[int, int]) -> None:
        """Update dynamic food position one step."""
        self.prev_position = self.position

        # Proximity Collision Override: check if next step is 1 block from wall
        next_pos = (
            self.position[0] + self.direction[0],
            self.position[1] + self.direction[1],
        )
        if self._near_collision(next_pos):
            self.state = 2
            self.remaining_momentum = 0

        if self.state == 1:
            # State 1: straight momentum
            if self._is_valid(next_pos):
                self.position = next_pos
                self.remaining_momentum -= 1
                if self.remaining_momentum <= 0:
                    self.state = 2
            else:
                self.state = 2
        else:
            # State 2: rotation or evasion
            self._choose_new_direction(snake_head)
            next_pos = (
                self.position[0] + self.direction[0],
                self.position[1] + self.direction[1],
            )
            if self._is_valid(next_pos):
                self.position = next_pos
            # Return to momentum state with new random distance
            self.state = 1
            self.remaining_momentum = random.randint(self.MOMENTUM_MIN, self.MOMENTUM_MAX)

    def _near_collision(self, pos: Tuple[int, int]) -> bool:
        """Check if pos is 1 block from wall (proximity override)."""
        r, c = pos
        return (
            r <= 0 or r >= self.grid_size - 1
            or c <= 0 or c >= self.grid_size - 1
        )

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        r, c = pos
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _choose_new_direction(self, snake_head: Tuple[int, int]) -> None:
        """State 2: random rotation OR greedy evasion (Euclidean farthest)."""
        if random.random() < self.EVASION_PROB:
            # Active evasion: pick direction with max Euclidean distance from snake
            best_dir = None
            best_dist = -1.0
            for d in DIRECTIONS.values():
                candidate = (self.position[0] + d[0], self.position[1] + d[1])
                if self._is_valid(candidate):
                    dist = math.hypot(
                        candidate[0] - snake_head[0],
                        candidate[1] - snake_head[1],
                    )
                    if dist > best_dist:
                        best_dist = dist
                        best_dir = d
            if best_dir is not None:
                self.direction = best_dir
        else:
            # Standard rotation: random direction
            self.direction = random.choice(list(DIRECTIONS.values()))


class game_environment(gym.Env):
    """
    Snake environment with dual-outlet observations.

    Observation types:
        - "spatiotemporal"        : 8×20×20 tensor (v2 — current default)
            Ch0 Wall             — 1.0 on the four border rows/cols
            Ch1 Decaying body    — head=1.0, decays linearly to tail
            Ch2 Static food      — 1.0 at each static food cell
            Ch3 Dynamic food     — 1.0 at current cell, 0.5 at previous cell
                                   (per-cell map; critical on levels 3-4)
            Ch4 Head direction   — 1.0 at the cell 1 step in `direction_idx`
            Ch5 Food direction   — 1.0 at the 4 cells around the head in any
                                   direction where SOME food exists; mirrors
                                   the 12-bit obs's Bits 8-11 (dx, dy signs)
            Ch6 Relative danger  — 1.0 at the 3 cells STRAIGHT/LEFT/RIGHT
                                   of head (relative to current heading) if
                                   wall or body; "behind" is always safe
                                   (no 180° reversal), so omitted. Sources
                                   are split by config: DANGER_FROM_WALL,
                                   DANGER_FROM_BODY, DANGER_TAIL_IS_SAFE
                                   (see game/env/config.json).
            Ch7 Snake length     — broadcast `len(snake) / 400` on every cell
        - "spatiotemporal_legacy" : 4×20×20 tensor (v1 — backward-compat for
            any saved model trained on the 4-channel layout with dynamic
            food map)
        - "12bit"                 : 1D 12-bit vector (obstacles + food dir)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        level: int = 1,
        obs_type: str = "spatiotemporal",
        max_steps: int = _cfg.MAX_GAME_STEPS, # default 500,
    ):
        super().__init__()

        assert level in LEVEL_CONFIG, f"Level must be 1-5, got {level}"
        assert obs_type in (
            "spatiotemporal", "spatiotemporal_legacy", "12bit",
        ), f"Invalid obs_type: {obs_type!r}"
        if REWARD_MILESTONE_ENABLED:
            assert REWARD_MILESTONE_INTERVAL > 0, (
                f"REWARD_MILESTONE_INTERVAL must be > 0, got {REWARD_MILESTONE_INTERVAL}"
            )

        self.level = level
        self.obs_type = obs_type
        self.max_steps = max_steps
        self.grid_size = GRID_SIZE

        # Action space: 4 discrete directions
        self.action_space = spaces.Discrete(4)

        # Observation space
        if obs_type == "spatiotemporal":
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(SPATIAL_OBS_CHANNELS_V2, GRID_SIZE, GRID_SIZE),
                dtype=np.float32,
            )
        elif obs_type == "spatiotemporal_legacy":
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(SPATIAL_OBS_CHANNELS_V1, GRID_SIZE, GRID_SIZE),
                dtype=np.float32,
            )
        else:  # 12bit
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(12,), dtype=np.float32,
            )

        # State (initialized on reset)
        self.snake: deque = deque()
        self.direction_idx: int = ACTION_RIGHT
        self.static_food: List[Tuple[int, int]] = []
        self.dynamic_food: List[dynamic_food] = []
        self.steps_taken: int = 0
        self.step_counter_for_dynamic: int = 0  # 2 out of 3 ticks for dynamic movement

        # Stagnation tracking — number of consecutive steps without
        # reducing the head's distance to the nearest food. Reset to 0
        # in reset() and on any approach step (see step()).
        self._no_progress_steps: int = 0
        self._stagnation_threshold: int = STAGNATION_THRESHOLD
        self._stagnation_penalty: float = STAGNATION_PENALTY

        # Milestone tracking — last length multiple of MILESTONE_INTERVAL
        # credited in this episode. Reset to initial value in reset().
        self._last_milestone: int = 0

        # Target-food tracking — the food cell the env has committed to
        # for approach reward (see TARGET_FOOD_ENABLED). None means "no
        # commitment yet"; the next step() will pick the nearest static
        # food. Reset by reset() and by _consume_food() when the target
        # is eaten.
        self._target_food: Optional[Tuple[int, int]] = None

    # ---------- Public API ----------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._init_snake()
        self._init_food()
        self.steps_taken = 0
        self.step_counter_for_dynamic = 0
        self._no_progress_steps = 0
        self._last_milestone = INITIAL_SNAKE_LENGTH // REWARD_MILESTONE_INTERVAL
        # Drop any commitment to a target food from the previous episode.
        # The new episode starts uncommitted; the first step() will pick
        # the nearest static food.
        self._target_food = None
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Update penalty value
        global REWARD_COLLISION, REWARD_EAT_STATIC, REWARD_EAT_DYNAMIC
        if DYNAMIC_PENALTY:
            REWARD_COLLISION = _cfg.REWARD_COLLISION * (1 + min(REWARD_MOD[0] * len(self.snake), REWARD_MOD_CAP[0])) # Increase penalty as snake grows, capped at -25
            REWARD_EAT_STATIC = _cfg.REWARD_EAT_STATIC * (1 + min(REWARD_MOD[1] * len(self.snake), REWARD_MOD_CAP[1])) #
            REWARD_EAT_DYNAMIC = _cfg.REWARD_EAT_DYNAMIC * (1 + min(REWARD_MOD[2] * len(self.snake), REWARD_MOD_CAP[2])) #
            
        self.steps_taken += 1
        self.step_counter_for_dynamic += 1

        old_head = self.snake[0]
        new_head = self._apply_action(action)

        # Detect collision
        collision = self._check_collision(new_head)
        if collision:
            # Snake length at the moment of death — the head has NOT
            # been appended yet, so ``self.snake`` still holds the
            # pre-move length. This is what we want to log.
            return self._get_obs(), REWARD_COLLISION, True, False, {
                "reason": "collision",
                "snake_length": len(self.snake),
            }

        # Move snake — append the new head, then decide whether to pop
        # the tail. Doing the append FIRST lets ``_ensure_target_food``
        # commit based on the post-move head position (which is what
        # will be used for the approach/away distance check on the
        # next iteration). If we ate food, the snake grows by 1 and
        # the tail is kept; otherwise we pop to keep the length.
        self.snake.appendleft(new_head)
        ate_food = self._check_food_at(new_head)
        # Commit to a target food BEFORE consuming, so that eating the
        # target cleanly releases the commitment (``_consume_food``
        # resets ``_target_food`` to None when the consumed cell was
        # the committed target). The order matters: if we committed
        # after consumption, the agent would chase the respawned
        # food's direction even on the eat step, which is wrong.
        self._ensure_target_food()
        food_type = self._consume_food(new_head) if ate_food else None
        if not ate_food:
            self.snake.pop()

        # Move dynamic food: 2 out of every 3 ticks (1.5x slower than snake)
        if self.step_counter_for_dynamic % 3 != 0:
            for dfood in self.dynamic_food:
                dfood.move(self.snake[0])

        # Check win condition
        if len(self.snake) >= MAX_GRID_AREA:
            return self._get_obs(), REWARD_EAT_STATIC, True, False, {
                "reason": "win",
                "snake_length": len(self.snake),
            }

        # Check max steps
        truncated = self.steps_taken >= self.max_steps
        if truncated:
            return self._get_obs(), REWARD_TIME, False, True, {
                "reason": "truncated",
                "snake_length": len(self.snake),
            }

        # Stagnation tracking — use the same distance metric as the
        # shaping reward so the two signals agree on what "progress"
        # means. Reset the counter on any approach step; otherwise
        # increment. Counter is consumed by _compute_reward below and
        # also drives the commitment-timeout (see below).
        # When target tracking is enabled this naturally tracks progress
        # toward the committed target (not whichever food happens to
        # be closest this step).
        old_dist_stag = self._food_distance(old_head)
        new_dist_stag = self._food_distance(new_head)
        if new_dist_stag < old_dist_stag:
            self._no_progress_steps = 0
        else:
            self._no_progress_steps += 1

        # Commitment timeout — if the snake has been making no progress
        # toward its target for TARGET_SWITCH_AFTER_AWAY_STEPS
        # consecutive steps, the body has likely encircled the target
        # and the commitment is a trap. Drop it and let the next step
        # commit to a new food (the nearest static food in the open).
        if (
            TARGET_FOOD_ENABLED
            and self._target_food is not None
            and self._no_progress_steps >= TARGET_SWITCH_AFTER_AWAY_STEPS
        ):
            self._target_food = None
            # Reset the counter so the new commitment gets a fresh
            # grace period before the next timeout check fires.
            self._no_progress_steps = 0

        # Compute reward (uses _food_distance, which is target-aware
        # — _ensure_target_food was already called above so the target
        # is set or freshly released by the time we get here).
        reward = self._compute_reward(old_head, new_head, food_type)
        return self._get_obs(), reward, False, False, {}

    # ---------- Initialization ----------

    def _init_snake(self) -> None:
        """Initialize snake at center of grid with length 3.

        The deque stores [head, body, tail] in that order — index 0
        is the head because every other method (step, _check_collision,
        draw_snake) reads snake[0] as the head and snake[-1] as the tail.
        """
        center = GRID_SIZE // 2
        self.snake = deque([
            (center, center + 1),  # head (index 0)
            (center, center),      # body
            (center, center - 1),  # tail (index -1)
        ])
        self.direction_idx = ACTION_RIGHT

    def _init_food(self) -> None:
        """Spawn food based on current level configuration."""
        config = LEVEL_CONFIG[self.level]
        self.static_food = []
        self.dynamic_food = []

        for _ in range(config["static"]):
            pos = self._random_empty_cell()
            self.static_food.append(pos)

        for _ in range(config["dynamic"]):
            pos = self._random_empty_cell()
            self.dynamic_food.append(dynamic_food(self.grid_size, pos))

    def _random_empty_cell(self) -> Tuple[int, int]:
        """Return a random cell not occupied by snake or other food."""
        occupied = set(self.snake) | set(self.static_food)
        for dfood in self.dynamic_food:
            occupied.add(dfood.position)
        empty_cells = [
            (r, c) for r in range(self.grid_size) for c in range(self.grid_size)
            if (r, c) not in occupied
        ]
        if not empty_cells:
            return (0, 0)  # fallback
        return random.choice(empty_cells)

    # ---------- Game logic ----------

    def _apply_action(self, action: int) -> Tuple[int, int]:
        """Compute new head position based on action. Prevents 180° reversal."""
        # Prevent reversing into body (unless snake length > 2)
        opposite = (self.direction_idx + 2) % 4
        if action == opposite and len(self.snake) > 2:
            action = self.direction_idx
        self.direction_idx = action

        dr, dc = DIRECTIONS[action]
        old_head = self.snake[0]
        return (old_head[0] + dr, old_head[1] + dc)

    def _check_collision(self, pos: Tuple[int, int]) -> bool:
        """Wall or self-body collision."""
        r, c = pos
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return True
        if pos in list(self.snake)[:-1]:
            return True
        return False

    def _check_food_at(self, pos: Tuple[int, int]) -> bool:
        """Check if any food is at this position."""
        if pos in self.static_food:
            return True
        for dfood in self.dynamic_food:
            if dfood.position == pos:
                return True
        return False

    def _consume_food(self, pos: Tuple[int, int]) -> str:
        """Remove food at pos and respawn. Returns 'static' or 'dynamic'."""
        if pos in self.static_food:
            self.static_food.remove(pos)
            # Respawn static food
            new_pos = self._random_empty_cell()
            self.static_food.append(new_pos)
            # If we just ate the target, drop the commitment so the
            # next step() can pick the new nearest food.
            if self._target_food == pos:
                self._target_food = None
            return "static"

        for i, dfood in enumerate(self.dynamic_food):
            if dfood.position == pos:
                self.dynamic_food.pop(i)
                new_pos = self._random_empty_cell()
                self.dynamic_food.append(dynamic_food(self.grid_size, new_pos))
                if self._target_food == pos:
                    self._target_food = None
                return "dynamic"
        return None

    # ----------------------------------------------------------------
    # Food-distance helpers (fix A = target tracking, fix D = static-only)
    # ----------------------------------------------------------------
    def _nearest_food_distance(self, head: Tuple[int, int]) -> float:
        """Euclidean distance from ``head`` to the nearest food cell.

        Candidate pool is controlled by ``DISTANCE_INCLUDE_DYNAMIC``:
            * True  — all static + dynamic food (original behaviour)
            * False — static food only when static is available;
                      falls back to dynamic food when no static food
                      exists (e.g. level 3/4 with 0 static). Without
                      this fallback the snake would get no
                      approach/away shaping at all on those levels.
        Returns 0.0 when no food exists, so callers can compare against
        a sentinel without a special case.
        """
        candidates = list(self.static_food)
        if not candidates or DISTANCE_INCLUDE_DYNAMIC:
            # Either explicitly include dynamic, or static is empty so
            # dynamic is the only option. Pool the dynamic positions
            # in either case.
            candidates = candidates + [d.position for d in self.dynamic_food]
        if not candidates:
            return 0.0
        return min(math.hypot(head[0] - r, head[1] - c) for r, c in candidates)

    def _food_distance(self, head: Tuple[int, int]) -> float:
        """Distance from ``head`` to the *committed* target food, if any.

        When target tracking is on AND a target is set, this is the
        distance to that specific cell. Otherwise it falls back to
        :func:`_nearest_food_distance` so callers (reward shaping,
        stagnation penalty) get a consistent "progress metric".
        """
        if TARGET_FOOD_ENABLED and self._target_food is not None:
            return math.hypot(
                head[0] - self._target_food[0],
                head[1] - self._target_food[1],
            )
        return self._nearest_food_distance(head)

    def _ensure_target_food(self) -> None:
        """Commit to a static food cell if no target is set yet.

        Called once per step (after food consumption, before reward
        computation) so the very first reward the agent sees already
        uses a real target. The first call after ``reset()`` picks the
        nearest static food to the snake's CURRENT head. Subsequent
        calls are no-ops until the current target is eaten and reset
        to ``None`` by :func:`_consume_food`.
        """
        if not TARGET_FOOD_ENABLED:
            return
        if self._target_food is not None:
            return
        if not self.static_food:
            return  # no food to commit to (e.g. on a food-less level)
        head_r, head_c = self.snake[0]
        self._target_food = min(
            self.static_food,
            key=lambda p: math.hypot(p[0] - head_r, p[1] - head_c),
        )

    # ---------- Reward ----------

    def _compute_reward(
        self,
        old_head: Tuple[int, int],
        new_head: Tuple[int, int],
        food_type: Optional[str],
    ) -> float:
        """Distance-based shaping + food/collision/time rewards."""
        reward = 0.0

        # 1. Distance-based shaping — tracks the COMMITTED target food
        # when target tracking is enabled, otherwise the nearest food
        # right now. Both branches share _food_distance so the shaping
        # signal here matches the stagnation counter above.
        old_dist = self._food_distance(old_head)
        new_dist = self._food_distance(new_head)
        if new_dist < old_dist:
            reward += REWARD_APPROACH
        else:
            reward += REWARD_AWAY

        # 2. Food eaten
        if food_type == "static":
            reward += REWARD_EAT_STATIC
        elif food_type == "dynamic":
            reward += REWARD_EAT_DYNAMIC

        # 3. Milestone bonus — ONLY for crossing length thresholds.
        # Per-milestone bonus = REWARD_MILESTONE_MOD *
        #   (1 + REWARD_MILESTONE_GROWTH * milestone_index)
        # The bonus applies solely when the snake crosses a length
        # multiple of REWARD_MILESTONE_INTERVAL; no other reward term
        # is scaled by it. Multi-step safe via arithmetic-series sum:
        # sum(i for i in prev+1..current).
        if REWARD_MILESTONE_ENABLED:
            current_milestone = len(self.snake) // REWARD_MILESTONE_INTERVAL
            if current_milestone > self._last_milestone:
                prev = self._last_milestone
                steps = current_milestone - prev
                index_sum = (
                    current_milestone * (current_milestone + 1) // 2
                    - prev * (prev + 1) // 2
                )
                reward += REWARD_MILESTONE_MOD * (
                    steps + REWARD_MILESTONE_GROWTH * index_sum
                )
                self._last_milestone = current_milestone

        # 4. Time penalty
        reward += REWARD_TIME

        # 5. Stagnation penalty — per-step cost after the head has gone
        # ``STAGNATION_THRESHOLD`` consecutive steps without reducing
        # its distance to the nearest food. Per-step (not one-shot)
        # so the policy gradient is smooth across the threshold. Gated
        # by STAGNATION_ENABLED so the env still runs cleanly when
        # the feature is disabled via config.
        if (
            STAGNATION_ENABLED
            and self._no_progress_steps >= self._stagnation_threshold
        ):
            reward += self._stagnation_penalty

        return reward

    # ---------- Observations ----------

    def _get_obs(self) -> np.ndarray:
        """Dispatch to appropriate observation builder."""
        if self.obs_type == "spatiotemporal":
            return self._spatiotemporal_obs()
        if self.obs_type == "spatiotemporal_legacy":
            return self._spatiotemporal_legacy_obs()
        return self._twelve_bit_obs()

    # ----------------------------------------------------------------
    # Spatial obs builders
    # ----------------------------------------------------------------
    def _spatiotemporal_legacy_obs(self) -> np.ndarray:
        """4×20×20 v1 tensor (kept for backward compat with old models).

            Ch0 Wall
            Ch1 Decaying body (head=1.0, tail→0.0)
            Ch2 Static food
            Ch3 Dynamic momentum (t=1.0, t-1=0.5)
        """
        return self._build_spatial_obs_v1()

    def _spatiotemporal_obs(self) -> np.ndarray:
        """8×20×20 v2 tensor — the default spatial obs.

            Ch0 Wall               (binary)
            Ch1 Decaying body      (head=1.0, tail→0.0)
            Ch2 Static food        (1.0 at each static food cell)
            Ch3 Dynamic food       (1.0 at current cell, 0.5 at previous cell)
            Ch4 Head direction     (1.0 at the cell 1 step in `direction_idx`)
            Ch5 Food direction     (1.0 at the 4 cells around the head in
                                    any direction where food exists; mirrors
                                    12-bit obs Bits 8-11)
            Ch6 Relative danger    (1.0 at the 3 cells STRAIGHT/LEFT/RIGHT
                                    of head, if wall or body)
            Ch7 Snake length       (broadcast `len(snake) / 400` everywhere)
        """
        return self._build_spatial_obs_v2()

    def _build_spatial_obs_v1(self) -> np.ndarray:
        """Legacy 4-channel obs kept for backward compat with old models.

        v1 layout: Ch0 Wall, Ch1 Decaying body, Ch2 Static food, Ch3
        Dynamic food (current=1.0, previous=0.5). New models should
        use the v2 7-channel obs via ``obs_type="spatiotemporal"`` —
        v1 exists only so saved models with the 4-channel shape still
        load via ``obs_type="spatiotemporal_legacy"``.
        """
        obs = np.zeros((4, GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Channel 0: Walls
        obs[0, 0, :] = 1.0
        obs[0, -1, :] = 1.0
        obs[0, :, 0] = 1.0
        obs[0, :, -1] = 1.0

        # Channel 1: Decaying body (head=1.0, linearly degraded to 0)
        snake_list = list(self.snake)
        n = len(snake_list)
        for i, (r, c) in enumerate(snake_list):
            if i == 0:
                obs[1, r, c] = 1.0
            else:
                obs[1, r, c] = max(0.0, 1.0 - (i / max(n, 1)))

        # Channel 2: Static food
        for r, c in self.static_food:
            obs[2, r, c] = 1.0

        # Channel 3: Dynamic food momentum
        for dfood in self.dynamic_food:
            r, c = dfood.position
            obs[3, r, c] = 1.0
            pr, pc = dfood.prev_position
            if 0 <= pr < GRID_SIZE and 0 <= pc < GRID_SIZE:
                obs[3, pr, pc] = max(obs[3, pr, pc], 0.5)

        return obs

    def _build_spatial_obs_v2(self) -> np.ndarray:
        """Full 8-channel v2 obs (current default).

        Layout: Ch0 Wall, Ch1 Decaying body, Ch2 Static food,
                Ch3 Dynamic food (current=1.0, previous=0.5),
                Ch4 Head direction, Ch5 Food direction (4 cells),
                Ch6 Relative danger (3 cells STRAIGHT/LEFT/RIGHT of
                head), Ch7 Snake length.

        Ch0-Ch3 are reused from the v1 obs (which has the same first
        four channels) so the per-cell dynamic-food map is built only
        once. Ch4-Ch7 add the directional / danger / length signals
        on top.
        """
        obs = np.zeros(
            (SPATIAL_OBS_CHANNELS_V2, GRID_SIZE, GRID_SIZE), dtype=np.float32,
        )
        # Ch0-Ch3: Wall, body, static food, dynamic food (reused from v1).
        # The v1 obs already writes these four channels in the right
        # layout — re-running it here is cheaper than duplicating the
        # body / food loops and keeps the dynamic-food momentum marker
        # (1.0 current, 0.5 previous) consistent across both obs types.
        obs[:4] = self._build_spatial_obs_v1()

        head_r, head_c = self.snake[0]
        dr, dc = DIRECTIONS[self.direction_idx]

        # ----------------------------------------------------------------
        # Ch4: Head direction — mark the cell the head will move into.
        # ----------------------------------------------------------------
        next_r, next_c = head_r + dr, head_c + dc
        if 0 <= next_r < GRID_SIZE and 0 <= next_c < GRID_SIZE:
            obs[CH_HEAD_DIR, next_r, next_c] = 1.0

        # ----------------------------------------------------------------
        # Ch5: Food in each direction (1 channel, up to 4 cells).
        # 1.0 at the cell adjacent to the head in any of the 4 absolute
        # directions (UP/RIGHT/DOWN/LEFT) where SOME food exists
        # anywhere in that half-plane relative to the head. This mirrors
        # the 12-bit obs's Bits 8-11 (sign of dy, sign of dx), so a
        # PPO→DQN ablation can read the same scalar signal from either
        # obs type.
        # ----------------------------------------------------------------
        all_food = list(self.static_food) + [d.position for d in self.dynamic_food]
        for action, (fdr, fdc) in DIRECTIONS.items():
            # Same sign test as 12-bit obs's Bits 8-11: a direction is
            # "set" if ANY food cell has the matching sign of
            # (food - head) along that axis. Strict inequality (>, <)
            # matches the 12-bit obs exactly — food directly on the same
            # row/col as the head is still "in that direction" for the
            # 12-bit version, so we mirror that semantics.
            if fdr != 0:  # vertical: fdr ∈ {-1, +1}
                sign = 1 if fdr > 0 else -1
                if any(((fr - head_r) * sign) > 0 for fr, _ in all_food):
                    tr, tc = head_r + fdr, head_c
                    if 0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE:
                        obs[CH_FOOD_DIR, tr, tc] = 1.0
            else:  # horizontal: fdc ∈ {-1, +1}
                sign = 1 if fdc > 0 else -1
                if any(((fc - head_c) * sign) > 0 for _, fc in all_food):
                    tr, tc = head_r, head_c + fdc
                    if 0 <= tr < GRID_SIZE and 0 <= tc < GRID_SIZE:
                        obs[CH_FOOD_DIR, tr, tc] = 1.0

        # ----------------------------------------------------------------
        # Ch6: Relative danger (1 channel, up to 3 cells). Mark the 3
        # cells STRAIGHT/LEFT/RIGHT of the head — relative to the head's
        # CURRENT HEADING. The "behind" cell is omitted because
        # `_apply_action` enforces no 180° reversal, so the snake can
        # never move into that cell on the next step. A cell is 1.0
        # if any of the configured ``DANGER_FROM_*`` sources consider
        # it deadly; otherwise 0. Sources are split by config flag so
        # the user can train with body-only or wall-only danger without
        # touching code (see game/env/config.json).
        # ----------------------------------------------------------------
        # Rotate (dr, dc) by 90° CCW → (-dc, dr) → snake's LEFT side.
        # Rotate (dr, dc) by 90° CW  → (dc, -dr) → snake's RIGHT side.
        straight = (head_r + dr,     head_c + dc)
        left     = (head_r - dc,     head_c + dr)
        right    = (head_r + dc,     head_c - dr)
        snake_set = set(self.snake)
        tail = self.snake[-1]
        for nr, nc in (straight, left, right):
            in_grid = (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE)
            if not in_grid:
                # Out of grid (e.g., head at row 1 facing UP → STRAIGHT
                # cell is row 0 which is the wall). Mark the head cell
                # itself as a sentinel — there's no in-grid cell at
                # the relative position. The network learns "Ch6 at my
                # own position = wall in some relative direction".
                # Only emit this sentinel when the wall source is enabled,
                # so a wall-ignorant agent doesn't see a phantom signal.
                if DANGER_FROM_WALL:
                    obs[CH_DANGER_REL, head_r, head_c] = 1.0
                continue
            # In-grid. The cell is dangerous per source:
            #   * DANGER_FROM_WALL : row 0/19 or col 0/19 (the border)
            #   * DANGER_FROM_BODY : a body cell, optionally treating
            #                         the tail as safe (it vacates next
            #                         step, so is not a real obstacle)
            # DANGER_TAIL_IS_SAFE toggles the tail exception; the body
            # source is considered off entirely when DANGER_FROM_BODY=False.
            is_wall = DANGER_FROM_WALL and (
                nr == 0 or nr == GRID_SIZE - 1
                or nc == 0 or nc == GRID_SIZE - 1
            )
            is_tail_cell = DANGER_TAIL_IS_SAFE and (nr, nc) == tail
            is_body = DANGER_FROM_BODY and (
                (nr, nc) in snake_set and not is_tail_cell
            )
            if is_wall or is_body:
                obs[CH_DANGER_REL, nr, nc] = 1.0

        # ----------------------------------------------------------------
        # Ch7: Snake length broadcast — single global scalar the CNN
        # can read off any cell. Normalised to MAX_GRID_AREA so it
        # stays in [0, 1] and matches the Box space high bound.
        # ----------------------------------------------------------------
        obs[CH_LENGTH, :, :] = min(1.0, len(self.snake) / float(MAX_GRID_AREA))

        return obs

    def _twelve_bit_obs(self) -> np.ndarray:
        """12-bit vector: 8 directions for obstacles + 4 for food direction.

        Bits 0-3: danger in directions [UP, RIGHT, DOWN, LEFT]
        Bits 4-7: body presence in those directions
        Bits 8-11: relative food direction (signs: dx>0, dx<0, dy>0, dy<0)
        """
        obs = np.zeros(12, dtype=np.float32)
        head_r, head_c = self.snake[0]

        # Bits 0-3: immediate obstacles
        for i, (dr, dc) in DIRECTIONS.items():
            nr, nc = head_r + dr, head_c + dc
            if (
                nr < 0 or nr >= GRID_SIZE
                or nc < 0 or nc >= GRID_SIZE
                or (nr, nc) in self.snake
            ):
                obs[i] = 1.0

        # Bits 4-7: body proximity (1 step ahead only)
        snake_set = set(self.snake)
        for i, (dr, dc) in DIRECTIONS.items():
            nr, nc = head_r + dr, head_c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr, nc) in snake_set:
                obs[4 + i] = 1.0

        # Bits 8-11: food direction (sign)
        all_food = list(self.static_food) + [d.position for d in self.dynamic_food]
        if all_food:
            # Pick nearest food
            nearest = min(
                all_food,
                key=lambda p: math.hypot(p[0] - head_r, p[1] - head_c),
            )
            dr = nearest[0] - head_r
            dc = nearest[1] - head_c
            if dr > 0:
                obs[8] = 1.0
            if dr < 0:
                obs[9] = 1.0
            if dc > 0:
                obs[10] = 1.0
            if dc < 0:
                obs[11] = 1.0

        return obs

    # ---------- Rendering ----------

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Simple ASCII render for debugging."""
        grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        # Walls
        for c in range(GRID_SIZE):
            grid[0][c] = "#"
            grid[-1][c] = "#"
        for r in range(GRID_SIZE):
            grid[r][0] = "#"
            grid[r][-1] = "#"
        # Snake
        for i, (r, c) in enumerate(self.snake):
            grid[r][c] = "H" if i == 0 else "o"
        # Static food
        for r, c in self.static_food:
            grid[r][c] = "F"
        # Dynamic food
        for dfood in self.dynamic_food:
            r, c = dfood.position
            grid[r][c] = "D"

        line = "\n".join("".join(row) for row in grid)
        print(line)
        return None

    def close(self) -> None:
        pass