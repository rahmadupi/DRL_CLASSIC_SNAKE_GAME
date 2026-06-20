"""
Snake Game Manual Tester
========================

A pygame GUI for testing the Snake environment with human controls.

Controls:
    Arrow Keys  - Move snake
    R           - Reset
    ESC         - Quit

Run:
    python test/test_pygame.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.env.game_environment import game_environment
from game.env.game_renderer import game_renderer
from game.env.input_controller import input_controller


def main():
    # 1. Create env (game logic)
    env = game_environment(level=2, obs_type="spatiotemporal")
    env.reset(seed=42)

    # 2. Create renderer + window (display)
    renderer = game_renderer(cell_size=50, grid_size=20, wall_thickness=8)
    screen = renderer.create_window(title="Snake Env Test (Manual)")

    # 3. Create input controller (handles keyboard)
    controller = input_controller()

    # Game loop — UI handles everything, just step the env
    running = True
    while running:
        # 4. Get action from input controller (handles UI state)
        action, running = controller.process_human_input()

        # 5. Handle quit
        if action is None and not running:
            break

        # 6. Handle "no input" (R-key reset) vs "waiting for first key"
        if action is None:
            if controller.game_started:
                # R was pressed during play → reset
                env.reset(seed=None)
                controller.reset_state()
            # else: still waiting for first arrow key — fall through to draw
        else:
            # 7. Step environment (controller flips game_started on first key)
            if controller.game_started:
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    env.reset(seed=None)
                    controller.reset_state()

        # 8. Always draw a frame (so the start prompt and reset are visible)
        renderer.draw_frame(screen, env, game_started=controller.game_started)
        pygame.display.flip()
        renderer.tick()

    pygame.quit()


if __name__ == "__main__":
    import pygame
    main()
