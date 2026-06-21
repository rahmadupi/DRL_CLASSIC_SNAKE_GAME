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
        - "spatiotemporal": 4x20x20 tensor (Wall, Body, Static, Dynamic)
        - "12bit": 1D 12-bit vector (obstacles + food direction)
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
        assert obs_type in ("spatiotemporal", "12bit"), f"Invalid obs_type: {obs_type}"

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
                shape=(4, GRID_SIZE, GRID_SIZE),
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

        # Move snake
        ate_food = self._check_food_at(new_head)
        food_type = self._consume_food(new_head) if ate_food else None
        self.snake.appendleft(new_head)
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
        # increment. Counter is consumed by _compute_reward below.
        old_dist_stag = self._nearest_food_distance(old_head)
        new_dist_stag = self._nearest_food_distance(new_head)
        if new_dist_stag < old_dist_stag:
            self._no_progress_steps = 0
        else:
            self._no_progress_steps += 1

        # Compute reward
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
            return "static"

        for i, dfood in enumerate(self.dynamic_food):
            if dfood.position == pos:
                self.dynamic_food.pop(i)
                new_pos = self._random_empty_cell()
                self.dynamic_food.append(dynamic_food(self.grid_size, new_pos))
                return "dynamic"
        return None

    def _nearest_food_distance(self, head: Tuple[int, int]) -> float:
        """Euclidean distance from head to nearest food (static or dynamic)."""
        candidates = list(self.static_food) + [d.position for d in self.dynamic_food]
        if not candidates:
            return 0.0
        return min(math.hypot(head[0] - r, head[1] - c) for r, c in candidates)

    # ---------- Reward ----------

    def _compute_reward(
        self,
        old_head: Tuple[int, int],
        new_head: Tuple[int, int],
        food_type: Optional[str],
    ) -> float:
        """Distance-based shaping + food/collision/time rewards."""
        reward = 0.0

        # 1. Distance-based shaping
        old_dist = self._nearest_food_distance(old_head)
        new_dist = self._nearest_food_distance(new_head)
        if new_dist < old_dist:
            reward += REWARD_APPROACH
        else:
            reward += REWARD_AWAY

        # 2. Food eaten
        if food_type == "static":
            reward += REWARD_EAT_STATIC
        elif food_type == "dynamic":
            reward += REWARD_EAT_DYNAMIC

        # 3. Time penalty
        reward += REWARD_TIME

        # 4. Stagnation penalty — per-step cost after the head has gone
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
        return self._twelve_bit_obs()

    def _spatiotemporal_obs(self) -> np.ndarray:
        """4x20x20 tensor:
            Ch1 Wall (binary)
            Ch2 Decaying body (head=1.0, tail→0.0)
            Ch3 Static food (1.0)
            Ch4 Dynamic momentum (t=1.0, t-1=0.5)
        """
        obs = np.zeros((4, GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Channel 1: Walls
        obs[0, 0, :] = 1.0
        obs[0, -1, :] = 1.0
        obs[0, :, 0] = 1.0
        obs[0, :, -1] = 1.0

        # Channel 2: Decaying body (head=1.0, linearly degraded to 0)
        snake_list = list(self.snake)
        n = len(snake_list)
        for i, (r, c) in enumerate(snake_list):
            if i == 0:
                obs[1, r, c] = 1.0
            else:
                obs[1, r, c] = max(0.0, 1.0 - (i / max(n, 1)))

        # Channel 3: Static food
        for r, c in self.static_food:
            obs[2, r, c] = 1.0

        # Channel 4: Dynamic food momentum
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