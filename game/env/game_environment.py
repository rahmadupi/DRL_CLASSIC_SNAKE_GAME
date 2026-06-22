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
# Honest obs (current default) — a single 4-channel 20×20 tensor:
#   Ch0 Wall
#   Ch1 Decaying body    — head=1.0, decays linearly to tail
#   Ch2 Static food
#   Ch3 Dynamic food     — 1.0 at current cell, 0.5 at previous cell
#
# The previous v2 layout also exposed head direction, food direction,
# relative danger, and snake length as separate channels on the grid.
# Those were "heuristic crutches" — the network could simply read
# them instead of learning the underlying geometry from Ch1's decay
# gradient or the global dynamic-food motion in Ch3. Removing them is
# what makes the comparison between PPO and DQN apples-to-apples on
# the spatial representation. Snake length is no longer broadcast at
# all — the agent must learn to deduce it from Ch1's body extent and
# decay gradient (or learn it implicitly via the rollout signal).
#
# ``CH_DYNAMIC`` is the only channel index that survives — it
# identifies the dynamic-food momentum layer (1.0 current, 0.5
# previous) inside the 4-channel grid, which is critical on levels 3-4
# where dynamic food is the ONLY target. Without the
# ``prev_position=0.5`` marker the agent only knows the food's current
# cell but not where it is moving.
SPATIAL_OBS_CHANNELS = 4
CH_DYNAMIC = 3  # 1.0 at dynamic food's current cell, 0.5 at previous

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

# Note: the previous MILESTONE-bonus and STAGNATION-penalty features
# were removed. Milestone bonuses caused instability in the late-game
# reward signal (sawtooth spikes on every length threshold). Stagnation
# was redundant with the constant REWARD_TIME pressure and added noise
# to the approach/away distance metric. The reward surface is now:
#   * distance shaping (approach / away)
#   * food (static / dynamic)
#   * collision
#   * constant time penalty
#   * encircle penalty (see below — kept; it catches a failure mode
#     that the distance shaping alone does NOT catch)
#
# Encircle penalty — per-step cost while the snake's head is inside a
# closed loop formed by its own body (i.e. unreachable from any
# non-snake boundary cell without crossing the body). Targets the
# "approach food but trap self" failure mode: the snake may still be
# reducing distance to food while its body is curving into an
# inescapable ring, so plain distance shaping does not catch it.
# Detection is BFS flood-fill from the first unoccupied corner;
# O(grid_size²) per step.
ENCIRCLE_PENALTY_ENABLED = _cfg.ENCIRCLE_PENALTY_ENABLED
ENCIRCLE_PENALTY = _cfg.ENCIRCLE_PENALTY
ENCIRCLE_DETECTION_MODE = _cfg.ENCIRCLE_DETECTION_MODE

# Spawn-proximity curriculum — during the first SPAWN_PROXIMITY_STEPS
# env steps, food is placed inside the Manhattan ball of radius
# SPAWN_PROXIMITY_RADIUS cells around the snake's head (only on
# initial episode placement, not on respawn after eating). This gives
# the agent a "warm-up" phase where it can reach food in a handful of
# steps and learn the food-is-good association before it has to
# navigate a 20×20 grid blind.
#
# Counter is per-env (one class-level int shared across instances in
# the same Python process). Inside SubprocVecEnv each worker has its
# own process and its own counter, so the warm-up applies per-env —
# with n_envs=4 and steps=50000 the rollout gets 4×50000 = 200k
# warm-up transitions in total. Set SPAWN_PROXIMITY_ENABLED=false to
# disable (default — preserves the original random spawn behaviour).
SPAWN_PROXIMITY_ENABLED = _cfg.SPAWN_PROXIMITY_ENABLED
SPAWN_PROXIMITY_RADIUS = _cfg.SPAWN_PROXIMITY_RADIUS
SPAWN_PROXIMITY_STEPS = _cfg.SPAWN_PROXIMITY_STEPS

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
    Snake environment with multi-outlet observations.

    Observation types:
        - "spatiotemporal"        : 4×20×20 tensor — HONEST layout (default)
            Ch0 Wall             — 1.0 on the four border rows/cols
            Ch1 Decaying body    — head=1.0, decays linearly to tail
                                   (also implicitly encodes length via
                                   the extent of the decay)
            Ch2 Static food      — 1.0 at each static food cell
            Ch3 Dynamic food     — 1.0 at current cell, 0.5 at previous
                                   (per-cell map; critical on levels 3-4)
            The previous v2 layout also exposed head direction, food
            direction, relative danger, and snake length as separate
            channels on the grid. Those were heuristic crutches — they
            let the network read the answers instead of learning the
            geometry from Ch1 (decay gradient → heading, extent →
            length) and Ch3 (momentum → food trajectory). Removed so
            PPO↔DQN and 12-bit↔spatiotemporal comparisons measure the
            algorithms, not the cheat-sheet.
        - "spatiotemporal_legacy" : 4×20×20 tensor — identical layout to
            ``spatiotemporal``. Kept purely so old saved-model filenames
            still resolve correctly via the obs_type-detection regex.
        - "12bit"                 : 1D 12-bit vector (obstacles + food dir)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # Class-level counter for the spawn-proximity curriculum. Reset to
    # 0 once at class-definition time; persists across resets within a
    # single Python process and is incremented once per ``step()`` call.
    #
    # In ``SubprocVecEnv`` each worker runs in its own process and
    # therefore has its own copy of this int — the warm-up window
    # applies per-env, not globally across the vec-env. With n_envs=4
    # and ``SPAWN_PROXIMITY_STEPS=50000`` the rollout receives 200k
    # warm-up transitions in absolute timesteps.
    _proximity_steps_taken: int = 0

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

        self.level = level
        self.obs_type = obs_type
        self.max_steps = max_steps
        self.grid_size = GRID_SIZE

        # Action space: 4 discrete directions
        self.action_space = spaces.Discrete(4)

        # Observation space — honest 4-channel spatial grid (no scalar
        # side-channel, no heuristic crutches). See the class docstring
        # for why Ch4-Ch7 (head dir, food dir, danger, length) were
        # removed from the spatial tensor.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(SPATIAL_OBS_CHANNELS, GRID_SIZE, GRID_SIZE),
            dtype=np.float32,
        )

        # State (initialized on reset)
        self.snake: deque = deque()
        self.direction_idx: int = ACTION_RIGHT
        self.static_food: List[Tuple[int, int]] = []
        self.dynamic_food: List[dynamic_food] = []
        self.steps_taken: int = 0
        self.step_counter_for_dynamic: int = 0  # 2 out of 3 ticks for dynamic movement

        # Target-food tracking — the food cell the env has committed to
        # for approach reward (see TARGET_FOOD_ENABLED). None means "no
        # commitment yet"; the next step() will pick the nearest static
        # food. Reset by reset() and by _consume_food() when the target
        # is eaten.
        self._target_food: Optional[Tuple[int, int]] = None

        # Approach/away counters — used to compute the
        # ``approach_ratio`` rollout metric (fraction of steps where
        # the snake's head moved CLOSER to its target food vs farther).
        # Reset to zero at every reset() and incremented inside
        # _compute_reward based on the distance comparison. Both
        # counters and the derived ratio are exposed via the terminal
        # info dict so ``ConfigurableMonitor`` can surface them as
        # ``info["episode"]["approach_ratio"]`` for the TensorBoard
        # callback to read.
        self._approach_count: int = 0
        self._away_count: int = 0

    @property
    def _approach_ratio(self) -> float:
        """Fraction of completed steps that moved the head closer to
        its current target food. Returns 0.0 when no reward-shaping
        steps have been recorded yet (e.g. immediate collision on
        step 1).
        """
        total = self._approach_count + self._away_count
        return self._approach_count / total if total > 0 else 0.0

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
        # Drop any commitment to a target food from the previous episode.
        # The new episode starts uncommitted; the first step() will pick
        # the nearest static food.
        self._target_food = None
        # Reset the approach/away counters so the per-episode ratio is
        # computed only from this episode's shaping steps.
        self._approach_count = 0
        self._away_count = 0
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
        # Spawn-proximity curriculum gate — increment the per-process
        # counter once per env step (including terminal/truncating
        # steps). The next ``reset()`` will consult this counter when
        # deciding where to place the new episode's initial food. See
        # ``SPAWN_PROXIMITY_*`` in config.json for the budget window.
        game_environment._proximity_steps_taken += 1

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
                "approach_ratio": self._approach_ratio,
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
                "approach_ratio": self._approach_ratio,
            }

        # Check max steps
        truncated = self.steps_taken >= self.max_steps
        if truncated:
            return self._get_obs(), REWARD_TIME, False, True, {
                "reason": "truncated",
                "snake_length": len(self.snake),
                "approach_ratio": self._approach_ratio,
            }

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
        """Spawn food based on current level configuration.

        When the spawn-proximity curriculum is active (see
        ``SPAWN_PROXIMITY_*`` in config.json), initial food cells are
        drawn from the Manhattan ball of radius ``SPAWN_PROXIMITY_RADIUS``
        around the head instead of from the whole grid. Respawns after
        eating always use the global random sampler so the long-tail
        behaviour is unaffected once the warm-up ends.
        """
        config = LEVEL_CONFIG[self.level]
        self.static_food = []
        self.dynamic_food = []

        proximity_on = self._proximity_active()

        for _ in range(config["static"]):
            if proximity_on:
                pos = self._random_empty_cell_in_radius(SPAWN_PROXIMITY_RADIUS)
                if pos is None:
                    # Snake has grown long enough that the radius is
                    # saturated; fall back to the global sampler so the
                    # episode still starts cleanly.
                    pos = self._random_empty_cell()
            else:
                pos = self._random_empty_cell()
            self.static_food.append(pos)

        for _ in range(config["dynamic"]):
            if proximity_on:
                pos = self._random_empty_cell_in_radius(SPAWN_PROXIMITY_RADIUS)
                if pos is None:
                    pos = self._random_empty_cell()
            else:
                pos = self._random_empty_cell()
            self.dynamic_food.append(dynamic_food(self.grid_size, pos))

    def _proximity_active(self) -> bool:
        """Whether the spawn-proximity curriculum is currently in effect.

        True iff the feature is enabled in config AND the class-level
        step counter is still below the configured window. The counter
        is shared across instances in the same process (one per worker
        in ``SubprocVecEnv``), so the window is measured in env steps,
        not in wall-clock seconds.
        """
        return (
            SPAWN_PROXIMITY_ENABLED
            and game_environment._proximity_steps_taken < SPAWN_PROXIMITY_STEPS
        )

    def _random_empty_cell_in_radius(
        self, radius: int
    ) -> Optional[Tuple[int, int]]:
        """Pick a uniformly-random empty cell within Manhattan radius
        ``radius`` of the snake's head, or ``None`` if no such cell
        exists.

        "Empty" means not occupied by the snake or any food already
        placed earlier in this ``_init_food`` call (static and dynamic
        food positions are added to the occupied set incrementally).
        Cells outside the grid are filtered out before the candidate
        list is built.
        """
        if radius < 0:
            return None
        head_r, head_c = self.snake[0]
        occupied = set(self.snake) | set(self.static_food)
        for dfood in self.dynamic_food:
            occupied.add(dfood.position)
        candidates = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue  # skip the head cell itself
                if abs(dr) + abs(dc) > radius:
                    continue  # outside the Manhattan ball
                r, c = head_r + dr, head_c + dc
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    if (r, c) not in occupied:
                        candidates.append((r, c))
        if not candidates:
            return None
        return random.choice(candidates)

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
        :func:`_nearest_food_distance` so callers (approach/away
        reward shaping) get a consistent "progress metric".
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

    def _is_enclosed(self, head: Tuple[int, int]) -> bool:
        """Whether ``head`` is trapped inside a closed loop of the snake.

        Algorithm: BFS from the first unoccupied corner cell over the
        grid, treating every snake cell **except the head** as a wall.
        If the head is not reached by the BFS, no path exists from
        outside the snake to the head without crossing another body
        cell — the head is enclosed and a future collision is
        mathematically certain (the snake has filled its own reachable
        region).

        Edge cases:
            * All four corners occupied by snake — return False
              (can't safely seed the BFS; this only happens near
              game-end when the snake has effectively won anyway).
            * Snake length = 1 — head is the only snake cell, so the
              BFS trivially reaches it; returns False.
            * The head's own cell is treated as walkable so the BFS
              can confirm "yes, the head is reachable from outside".
              Treating the head as a wall would make EVERY snake
              report as enclosed (false positive).

        Cost: O(grid_size²) per call. With grid_size=20 that's 400
        cell visits per step — sub-millisecond. Caching is not worth
        it because the body's position changes every step (head moves,
        tail vacates), so enclosure topology can flip at any time.
        """
        # Pick the first unoccupied corner as BFS source. The four
        # corners are tried in a fixed order; on the initial reset the
        # snake is at the center so (0,0) is always free.
        source: Optional[Tuple[int, int]] = None
        for corner in (
            (0, 0),
            (0, self.grid_size - 1),
            (self.grid_size - 1, 0),
            (self.grid_size - 1, self.grid_size - 1),
        ):
            if corner not in self.snake:
                source = corner
                break
        if source is None:
            # No safe corner to seed from — defer to "not enclosed"
            # (snake has filled the entire frame, so this is a near-
            # win state where the penalty would be moot anyway).
            return False

        # Wall set = body cells excluding the head. Removing the head
        # lets the BFS reach it if a path exists from outside — the
        # enclosure test is "head ∉ visited" after BFS completes.
        walls = set(self.snake)
        walls.discard(head)
        visited = {source}
        queue = deque([source])
        while queue:
            r, c = queue.popleft()
            for dr, dc in DIRECTIONS.values():
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.grid_size
                    and 0 <= nc < self.grid_size
                    and (nr, nc) not in visited
                    and (nr, nc) not in walls
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return head not in visited

    # ---------- Reward ----------

    def _compute_reward(
        self,
        old_head: Tuple[int, int],
        new_head: Tuple[int, int],
        food_type: Optional[str],
    ) -> float:
        """Distance-based shaping + food/collision/time rewards.

        Reward surface (in evaluation order):
            1. Distance shaping (approach / away)
            2. Food eaten (static / dynamic)
            3. Constant time penalty
            4. Encircle penalty (BFS-detected self-trap)

        The previous milestone-bonus and stagnation-penalty terms
        were removed — see the comment block above the
        ``REWARD_MOD`` constants for the rationale.
        """
        reward = 0.0

        # 1. Distance-based shaping — tracks the COMMITTED target food
        # when target tracking is enabled, otherwise the nearest food
        # right now. Both branches share _food_distance so the
        # approach/away signal is consistent.
        old_dist = self._food_distance(old_head)
        new_dist = self._food_distance(new_head)
        if new_dist < old_dist:
            reward += REWARD_APPROACH
            # Tally for the ``approach_ratio`` rollout metric. Counts
            # EVERY step where distance strictly decreased — including
            # the eat step itself (distance becomes 0).
            self._approach_count += 1
        else:
            reward += REWARD_AWAY
            self._away_count += 1

        # 2. Food eaten
        if food_type == "static":
            reward += REWARD_EAT_STATIC
        elif food_type == "dynamic":
            reward += REWARD_EAT_DYNAMIC

        # 3. Time penalty
        reward += REWARD_TIME

        # 4. Encircle penalty — per-step cost while the snake's head is
        # inside a closed loop of its own body (BFS from outside cannot
        # reach the head without crossing the snake). Targets the
        # "approach food but trap self" failure mode: the snake may
        # still be reducing distance to food while its body is curving
        # into an inescapable ring, so plain distance shaping does
        # not catch it. Per-step so the agent learns "don't close the
        # loop" well before the inevitable collision. Gated by
        # ENCIRCLE_PENALTY_ENABLED and the detection mode (currently
        # only "flood_fill" — other modes can be added later as
        # cheaper approximations).
        if (
            ENCIRCLE_PENALTY_ENABLED
            and ENCIRCLE_DETECTION_MODE == "flood_fill"
            and self._is_enclosed(new_head)
        ):
            reward += ENCIRCLE_PENALTY

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
    def _spatiotemporal_obs(self) -> np.ndarray:
        """4×20×20 tensor — honest layout (current default).

            Ch0 Wall
            Ch1 Decaying body (head=1.0, tail→0.0)
            Ch2 Static food
            Ch3 Dynamic food (current=1.0, previous=0.5)
        """
        return self._build_spatial_obs()

    def _spatiotemporal_legacy_obs(self) -> np.ndarray:
        """Alias for :meth:`_spatiotemporal_obs`.

        The legacy obs_type exists purely for backward-compat with
        saved-model filenames — both ``spatiotemporal`` and
        ``spatiotemporal_legacy`` now produce the exact same 4-channel
        tensor since Ch4-Ch7 were removed (see class docstring).
        """
        return self._build_spatial_obs()

    def _build_spatial_obs(self) -> np.ndarray:
        """Build the 4-channel spatial grid.

        Layout: Ch0 Wall, Ch1 Decaying body, Ch2 Static food, Ch3
        Dynamic food (current=1.0, previous=0.5). This is the only
        spatial obs produced by the env — both ``spatiotemporal`` and
        ``spatiotemporal_legacy`` ``obs_type`` values return this
        tensor.

        Notes on what is NOT here (removed from the previous v2 layout):
        * Head direction     — deducible from Ch1's decay gradient.
        * Food direction     — deducible from Ch3's momentum + Ch2.
        * Relative danger    — deducible from Ch0 (walls) + Ch1 (body).
        * Snake length       — no longer broadcast; the agent must
                                 learn it from Ch1's body extent and
                                 decay gradient (or implicitly via the
                                 rollout signal).
        """
        obs = np.zeros((SPATIAL_OBS_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

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