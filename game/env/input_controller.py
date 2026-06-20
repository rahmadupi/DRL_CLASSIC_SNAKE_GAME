import os
import random
from typing import Tuple, Optional
import gymnasium as gym

import pygame

from game.env.config import config as _cfg
# ============================================================================
# Input Controller (HUMAN + AI PLACEHOLDER)
# ============================================================================

class input_controller:
    """
    Handles all input logic for the game:
        - Human input via arrow keys (with 180° anti-reverse buffering)
        - AI input placeholder (will load trained model later)

    State (for human mode):
        - current_action: last applied direction (None = waiting for first input)
        - queued_action: next direction to apply (buffered)
        - game_started: False until first key pressed
    """

    # Action constants (sourced from config.json — single source of truth)
    ACTION_UP = _cfg.ACTION_UP
    ACTION_RIGHT = _cfg.ACTION_RIGHT
    ACTION_DOWN = _cfg.ACTION_DOWN
    ACTION_LEFT = _cfg.ACTION_LEFT

    # Map pygame keys to action indices
    KEY_TO_ACTION = {
        pygame.K_UP: ACTION_UP,
        pygame.K_DOWN: ACTION_DOWN,
        pygame.K_LEFT: ACTION_LEFT,
        pygame.K_RIGHT: ACTION_RIGHT,
    }

    def __init__(self):
        # Human input state
        self.current_action: Optional[int] = None
        self.queued_action: Optional[int] = None
        self.game_started: bool = False

        # AI state (placeholder)
        self.ai_model = None  # TODO: Load trained model here

    # ----------------------------------------------------------------
    # Human Input
    # ----------------------------------------------------------------
    def process_human_input(self) -> Tuple[Optional[int], bool]:
        """
        Read pygame events and update input state.

        Returns:
            action: Action to step the env with (None = no action / reset / quit)
            keep_running: False if user wants to quit, True otherwise
        """
        keep_running = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keep_running = False
                action = None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    keep_running = False
                    action = None
                elif event.key == pygame.K_r:
                    # Reset request — return None to signal reset
                    self.reset_state()
                    action = None
                elif event.key in self.KEY_TO_ACTION:
                    action = self.KEY_TO_ACTION[event.key]
                    # First key starts the game
                    if not self.game_started:
                        self.current_action = action
                        self.game_started = True
                    # Buffer subsequent input
                    self.queued_action = action
                else:
                    action = None
            else:
                action = None

        # Also poll for held keys (more responsive for continuous press)
        keys = pygame.key.get_pressed()
        for key, action in self.KEY_TO_ACTION.items():
            if keys[key]:
                if not self.game_started:
                    self.current_action = action
                    self.game_started = True
                self.queued_action = action
                break  # Only one direction at a time

        # Apply queued action (with anti-reverse protection)
        if self.game_started and self.queued_action is not None:
            # Check 180° reverse: prevent immediate reversal
            # This requires the env's current direction_idx — handled by env.step()

            # If you want pre-check (optional, since env also checks):
            action = self.queued_action
            self.queued_action = None
            self.current_action = action   # Track the last applied direction
            return action, keep_running

        # No new input this frame
        if self.game_started and self.current_action is not None:
            return self.current_action, keep_running

        return None, keep_running

    # ----------------------------------------------------------------
    # AI Input (Placeholder)
    # ----------------------------------------------------------------
    def process_ai_input(self, obs=None) -> int:
        """
        Get the next action for the AI agent.

        If a model has been loaded via `load_model(...)` and `obs` is
        provided, the model's prediction is used. Otherwise (or if the
        model raises during prediction) this falls back to a random
        action that respects the 180° anti-reverse rule, so the game
        never enters an impossible state.

        Args:
            obs: Current observation from the env. Shape/type must
                 match the obs space the loaded model was trained on
                 (e.g., 4×20×20 for "spatiotemporal", 12-dim for
                 "12bit"). If None, falls back to a random action.

        Returns:
            action: Integer 0-3 (UP, RIGHT, DOWN, LEFT)
        """
        if not self.game_started or self.current_action is None:
            action = self._random_ai_action()
        # --- 1. Try the trained model ---------------------------------
        if self.ai_model is not None and obs is not None:
            try:
                action, _ = self.ai_model.predict(obs, deterministic=True)
                action = int(action)
            except Exception as exc:
                # Wrong obs shape, mismatched model, etc.
                print(
                    f"[input_controller] Model predict() failed: {exc!r}. "
                    f"Falling back to random action."
                )
                action = self._random_ai_action()
        else:
            # --- 2. Fallback: random anti-reverse policy --------------
            action = self._random_ai_action()

        # Track the last chosen action (mirrors the human-input path)
        self.current_action = action
        self.game_started = True
        return action

    def _random_ai_action(self) -> int:
        """
        Random action with 180° anti-reverse protection.

        90% of the time the snake will not pick the immediate reverse
        direction; 10% it does, so the snake still has a chance of
        recovering from a dead-end during early training / fallback.
        """
        if not self.game_started or self.current_action is None:
            # First move: any direction is valid
            return 0

        if random.random() < 0.9:
            opposite = (self.current_action + 2) % 4
            safe = [a for a in range(4) if a != opposite and a != self.current_action]
            return random.choice(safe)
        return random.randint(0, 3)

    def load_model(self, model_path: str) -> None:
        """
        Load a trained PPO model from disk.

        The env's `obs_type` must match the obs space the model was
        trained on, otherwise the model's predictions will be garbage.

        Usage:
            controller.load_model("saved_models/ppo_level5.zip")
        """
        # Imported lazily so non-AI users don't need stable_baselines3
        from stable_baselines3 import PPO

        self.ai_model = PPO.load(model_path)
        print(f"[input_controller] Loaded AI model: {model_path}")

    def reset_state(self):
        """Reset input state (call when env resets)."""
        self.current_action = None
        self.queued_action = None
        self.game_started = False
