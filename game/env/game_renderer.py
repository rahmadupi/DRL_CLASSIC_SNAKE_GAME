"""
Game Renderer Module
====================

Shared rendering + input control for the Snake game environment.

Handles:
    - Text rendering (PIL → pygame.Surface, avoids pygame 2.6.1 font.blit bug)
    - Background rendering (wall, checker pattern)
    - Entity rendering (snake, food)
    - Score / status overlays
    - Human input processing (arrow keys with 180° anti-reverse buffering)
    - AI input placeholder (returns random action — will load model later)

Usage:
    from game.env.game_renderer import game_renderer

    renderer = game_renderer(cell_size=50, grid_size=20, wall_thickness=8)
    screen = renderer.create_window("Snake")

    while True:
        action = renderer.process_human_input(env)
        if action is None:
            break  # User quit
        env.step(action)
        renderer.draw_frame(screen, env)
        pygame.display.flip()
"""

import os
import random
from typing import Tuple, Optional

import pygame
from PIL import Image, ImageDraw, ImageFont

from game.env.config import config as _cfg


# ============================================================================
# Color Constants (sourced from config.json → COLORS section)
# ============================================================================
class Colors:
    """Central color palette for the game renderer.

    Values are loaded once from `config.json` so they can be tweaked
    without touching code. Each color is a 3-tuple of (R, G, B).
    """

    # Background
    WALL       = tuple(_cfg.COLORS["WALL"])            # Black wall border
    BG_BASE    = tuple(_cfg.COLORS["BG_BASE"])         # Cream beige background
    BG_CHECKER = tuple(_cfg.COLORS["BG_CHECKER"])      # Subtle beige checker

    # Snake
    SNAKE_HEAD = tuple(_cfg.COLORS["SNAKE_HEAD"])      # Bright blue head
    SNAKE_BODY = tuple(_cfg.COLORS["SNAKE_BODY"])      # Darker blue body

    # Food
    STATIC_FOOD  = tuple(_cfg.COLORS["STATIC_FOOD"])   # Green
    DYNAMIC_FOOD = tuple(_cfg.COLORS["DYNAMIC_FOOD"])  # Red

    # Wall highlight
    WALL_HIGHLIGHT = tuple(_cfg.COLORS["WALL_HIGHLIGHT"])  # Grey inner border

    # Score / text
    SCORE = tuple(_cfg.COLORS["SCORE"])                # Dark blue


# ============================================================================
# Text Rendering (Pillow-based, avoids pygame.font.blit segfault)
# ============================================================================
def render_text_pil(
    text: str,
    font_size: int = 36,
    color: Tuple[int, int, int] = Colors.SCORE,
) -> pygame.Surface:
    """
    Render text using Pillow (PIL) → convert to pygame Surface.
    Avoids pygame.font.blit segfault bug in pygame 2.6.1.

    Args:
        text: The string to render
        font_size: Pixel size of the text
        color: RGB tuple for text color

    Returns:
        pygame.Surface with the rendered text (transparent background)
    """
    # Try to load a system font (fallback chain)
    module_dir = os.path.dirname(os.path.abspath(__file__))
    font_paths = [
        os.path.join(module_dir, "..", "assets", "Minecraft.ttf"),
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/Arial.ttf",
        "/usr/share/fonts/TTF/LiberationSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    pil_font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                pil_font = ImageFont.truetype(path, font_size)
                break
            except Exception:
                continue
    if pil_font is None:
        pil_font = ImageFont.load_default()

    # Measure text bounding box
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=pil_font)
    text_w = bbox[2] - bbox[0] + 8
    text_h = bbox[3] - bbox[1] + 8

    # Render to RGBA image (transparent background)
    pil_img = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
    pil_draw = ImageDraw.Draw(pil_img)
    pil_draw.text((4, 4 - bbox[1]), text, font=pil_font, fill=color)

    # Convert PIL Image → pygame Surface (uses image.fromstring, NOT font.blit)
    raw_bytes = pil_img.tobytes()
    return pygame.image.fromstring(raw_bytes, pil_img.size, pil_img.mode)


# ============================================================================
# Game Renderer Class
# ============================================================================
class game_renderer:
    """
    Renders the Snake game environment to a pygame Surface.

    Handles:
        - Wall outline (drawn outside the play area)
        - Cream + subtle checkerboard background
        - Snake body/head with distinct colors
        - Static and dynamic food entities
        - Score display (top-left corner via PIL)
        - "Press arrow to start" prompt (when waiting)
    """

    def __init__(
        self,
        cell_size: int = _cfg.DEFAULT_CELL_SIZE,
        grid_size: int = _cfg.GRID_SIZE,
        wall_thickness: int = _cfg.DEFAULT_WALL_THICKNESS,
        fps: int = _cfg.DEFAULT_FPS,
    ):
        """
        Args:
            cell_size: Pixel size of each grid cell
            grid_size: Number of cells per side (default 20 = 20x20 grid)
            wall_thickness: Pixel width of the wall border
            fps: Target frames per second for game loop
        """
        self.cell_size = cell_size
        self.grid_size = grid_size
        self.wall_thickness = wall_thickness
        self.fps = fps

        # Derived dimensions
        self.play_size = grid_size * cell_size              # Play area (e.g., 1000x1000)
        self.window_w = self.play_size + 2 * wall_thickness  # Total window width
        self.window_h = self.window_w                         # Square window
        self.offset = wall_thickness                         # Grid offset from window origin

    def create_window(self, title: str = "Snake Game") -> pygame.Surface:
        """
        Create the pygame display window.
        Returns the screen Surface.
        """
        pygame.init()
        screen = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption(title)
        return screen

    def tick(self, fps: Optional[int] = None) -> None:
        """Wait to maintain target FPS. Call once per frame."""
        if not hasattr(self, '_clock'):
            self._clock = pygame.time.Clock()
        self._clock.tick(fps if fps is not None else self.fps)

    # ----------------------------------------------------------------
    # Background
    # ----------------------------------------------------------------
    def draw_background(self, screen: pygame.Surface) -> None:
        """Draw wall outline + cream base + subtle checker pattern."""
        # 1. Black background (= wall area)
        screen.fill(Colors.WALL)

        # 2. Cream base inside the wall
        pygame.draw.rect(
            screen, Colors.BG_BASE,
            (self.wall_thickness, self.wall_thickness, self.play_size, self.play_size)
        )

        # 3. Subtle checkerboard overlay
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r + c) % 2 == 0:
                    pygame.draw.rect(
                        screen, Colors.BG_CHECKER,
                        (self.offset + c * self.cell_size,
                         self.offset + r * self.cell_size,
                         self.cell_size, self.cell_size)
                    )

    # ----------------------------------------------------------------
    # Entities (snake, food)
    # ----------------------------------------------------------------
    def draw_snake(self, screen: pygame.Surface, snake: list) -> None:
        """
        Draw the snake. Head is brighter than body.
        Args:
            snake: deque of (row, col) tuples, head at index 0
        """
        for i, (r, c) in enumerate(snake):
            color = Colors.SNAKE_HEAD if i == 0 else Colors.SNAKE_BODY
            pygame.draw.rect(
                screen, color,
                (self.offset + c * self.cell_size,
                 self.offset + r * self.cell_size,
                 self.cell_size, self.cell_size)
            )

    def draw_static_food(self, screen: pygame.Surface, food_list: list) -> None:
        """Draw static food items (green)."""
        for r, c in food_list:
            pygame.draw.rect(
                screen, Colors.STATIC_FOOD,
                (self.offset + c * self.cell_size,
                 self.offset + r * self.cell_size,
                 self.cell_size, self.cell_size)
            )

    def draw_dynamic_food(self, screen: pygame.Surface, dynamic_food_list: list) -> None:
        """Draw dynamic food items (red)."""
        for dfood in dynamic_food_list:
            r, c = dfood.position
            pygame.draw.rect(
                screen, Colors.DYNAMIC_FOOD,
                (self.offset + c * self.cell_size,
                 self.offset + r * self.cell_size,
                 self.cell_size, self.cell_size)
            )

    # ----------------------------------------------------------------
    # Overlays (score, prompt)
    # ----------------------------------------------------------------
    def draw_score(self, screen: pygame.Surface, score: int) -> None:
        """Draw the score in the top-left corner using PIL text rendering."""
        try:
            score_text = render_text_pil(f"Score: {score}", font_size=36, color=Colors.SCORE)
            screen.blit(score_text, (self.offset + 15, self.offset + 10))
        except Exception:
            pass  # Silently skip on error

    def draw_start_prompt(self, screen: pygame.Surface) -> None:
        """Draw blinking arrow indicators when waiting for first key press."""
        pulse = (pygame.time.get_ticks() // 500) % 2
        if not pulse:
            return

        cx, cy = self.window_w // 2, self.window_h // 2
        arrow_size = 30
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            pygame.draw.polygon(
                screen, (100, 200, 100),
                [
                    (cx + dx * arrow_size * 2, cy + dy * arrow_size * 2),
                    (cx + dx * arrow_size * 2 + dx * arrow_size // 2,
                     cy + dy * arrow_size * 2 + dy * arrow_size // 2),
                    (cx + dx * arrow_size * 2 - dy * arrow_size // 2,
                     cy + dy * arrow_size * 2 + dx * arrow_size // 2),
                ]
            )

    # ----------------------------------------------------------------
    # Master frame draw (call this each frame)
    # ----------------------------------------------------------------
    def draw_frame(
        self,
        screen: pygame.Surface,
        env,
        game_started: bool = True,
    ) -> None:
        """
        Render a complete frame.

        Args:
            screen: pygame Surface to draw on
            env: game_environment instance with snake, static_food, dynamic_food
            game_started: whether the game has started (affects start prompt)
        """
        # Background (wall + cream + checker)
        self.draw_background(screen)

        # Entities
        self.draw_static_food(screen, env.static_food)
        self.draw_dynamic_food(screen, env.dynamic_food)
        self.draw_snake(screen, env.snake)

        # Overlays
        score = len(env.snake) - 3  # Initial snake length is 3
        self.draw_score(screen, score)

        if not game_started:
            self.draw_start_prompt(screen)


