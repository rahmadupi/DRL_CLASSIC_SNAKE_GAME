"""
Snake Game Launcher
===================

Neubrutal-style main menu for the Snake game.

Flow:
    MENU  →  (Play / Watch AI)  →  LEVELS  →  FPS_MODAL  →  PLAYING
                                          ┌──────────────┴──────────────┐
                                       (death)                        (win: 20×20 filled)
                                          ▼                                ▼
                                    DEATH_MODAL                       WIN_MODAL
                                          │                                │
                                 ┌────────┴────────┐              ┌────────┴────────┐
                              Retry           Main Menu       Play Again       Main Menu

Run:
    python game_launcher.py
"""

from __future__ import annotations

import glob
import os
import sys
from enum import Enum
from typing import List, Optional, Tuple

# Ensure the project root is on sys.path regardless of CWD
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pygame

from game.env.config import config as _cfg
from game.env.game_environment import (
    game_environment,
    LEVEL_CONFIG,
)
from game.env.game_renderer import game_renderer, render_text_pil
from game.env.input_controller import input_controller
from game.env.ui_components import (
    Panel,
    Button,
    ListBox,
    TextInput,
    Modal,
    BG_CREAM,
    BLACK,
    WHITE,
    PANEL_FILL,
    BORDER_WIDTH,
    SHADOW_OFFSET,
    BUTTON_MIN_WIDTH,
    TITLE_FONT_SIZE,
    BODY_FONT_SIZE,
)


# ============================================================================
# State machine
# ============================================================================
class State(Enum):
    MENU = "menu"
    MODELS = "models"
    LEVELS = "levels"
    FPS_MODAL = "fps_modal"
    PLAYING = "playing"
    DEATH_MODAL = "death_modal"
    WIN_MODAL = "win_modal"
    RETURN_MODAL = "return_modal"
    QUIT = "quit"


# ============================================================================
# Helpers
# ============================================================================
def _level_description(level: int) -> str:
    cfg = LEVEL_CONFIG.get(level, {"static": 0, "dynamic": 0})
    static = cfg.get("static", 0)
    dynamic = cfg.get("dynamic", 0)
    parts = []
    if static:
        parts.append(f"{static} static")
    if dynamic:
        parts.append(f"{dynamic} dynamic")
    if not parts:
        return f"Level {level} - empty"
    return f"Level {level} - " + " + ".join(parts)


def _discover_models() -> List[str]:
    if not os.path.isdir("saved_models"):
        return []
    return sorted(glob.glob(os.path.join("saved_models", "*.zip")))


def _filename(path: str) -> str:
    return os.path.basename(path)


# ============================================================================
# Launcher
# ============================================================================
class Launcher:
    WINDOW_W = int(_cfg.LAUNCHER_WIDTH)
    WINDOW_H = int(_cfg.LAUNCHER_HEIGHT)

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H))
        pygame.display.set_caption("SNAKE GAME")
        self.clock = pygame.time.Clock()

        # State
        self.state: State = State.MENU
        self.mode: str = "human"  # "human" or "ai"
        self.model_path: Optional[str] = None
        self.level: int = 1
        self.fps: int = int(_cfg.FPS_DEFAULT)

        # Per-screen data
        self.menu_index: int = 0
        self.menu_items: List[Tuple[str, str]] = [
            ("human", "Play as Human"),
            ("ai",    "Watch AI Play"),
        ]
        self.menu_button_rects: List[pygame.Rect] = []

        self.model_paths: List[str] = _discover_models()
        self.model_index: int = 0
        self.model_listbox: Optional[ListBox] = None

        self.level_index: int = 0
        self.level_button_rects: List[pygame.Rect] = []
        self.level_items: List[Tuple[int, str]] = [
            (lvl, _level_description(lvl)) for lvl in sorted(LEVEL_CONFIG.keys())
        ]

        self.fps_input: Optional[TextInput] = None
        self.fps_modal: Optional[Modal] = None

        # Death modal
        self.death_modal: Optional[Modal] = None
        self.death_score: int = 0
        # "collision" (real death) or "truncated" (clock ran out).
        # Set in _run_playing right before transitioning to DEATH_MODAL.
        self.death_reason: str = "collision"
        self.retry_button: Optional[Button] = None
        self.menu_button: Optional[Button] = None

        # Win modal — shown when the snake fills the entire grid
        # (``reason="win"`` from the env, only possible at level 1 with
        # 1 static food after the agent has lengthened to 400 cells).
        # Two buttons: "Play Again" resets the env on the SAME level;
        # "Main Menu" tears down and returns to the level picker.
        self.win_modal: Optional[Modal] = None
        self.win_score: int = 0
        self.win_retry_button: Optional[Button] = None
        self.win_menu_button: Optional[Button] = None

        # Pause / return-to-menu modal (triggered by R during PLAYING)
        self.return_modal: Optional[Modal] = None
        self.continue_button: Optional[Button] = None
        self.return_menu_button: Optional[Button] = None

        # Game runtime handles
        self.env: Optional[game_environment] = None
        self.renderer: Optional[game_renderer] = None
        self.controller: Optional[input_controller] = None
        self.renderer_screen: Optional[pygame.Surface] = None
        self._last_obs = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        while self.state != State.QUIT:
            if self.state == State.MENU:
                self._run_menu()
            elif self.state == State.MODELS:
                self._run_models()
            elif self.state == State.LEVELS:
                self._run_levels()
            elif self.state == State.FPS_MODAL:
                self._run_fps_modal()
            elif self.state == State.PLAYING:
                self._run_playing()
            elif self.state == State.DEATH_MODAL:
                self._run_death_modal()
            elif self.state == State.WIN_MODAL:
                self._run_win_modal()
            elif self.state == State.RETURN_MODAL:
                self._run_return_modal()
        pygame.quit()

    # ------------------------------------------------------------------
    # MENU
    # ------------------------------------------------------------------
    def _run_menu(self) -> None:
        # Window is created once in __init__ and never resized — no flicker.
        self._layout_menu()
        self.menu_index = 0

        running = True
        while running and self.state == State.MENU:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state = State.QUIT
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.menu_index = max(0, self.menu_index - 1)
                    elif event.key == pygame.K_DOWN:
                        self.menu_index = min(len(self.menu_items) - 1, self.menu_index + 1)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self._activate_menu(self.menu_index)
                        running = False
                else:
                    if event.type == pygame.MOUSEMOTION:
                        for i, rect in enumerate(self.menu_button_rects):
                            if rect.collidepoint(event.pos):
                                self.menu_index = i
                                break
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        for i, rect in enumerate(self.menu_button_rects):
                            if rect.collidepoint(event.pos):
                                self._activate_menu(i)
                                running = False
                                break

            self._draw_menu()
            self.clock.tick(30)

    def _layout_menu(self) -> None:
        cx = self.WINDOW_W // 2
        button_w = 270
        button_h = 40
        gap = 0  # no vertical gap — buttons stack flush
        total_h = button_h * len(self.menu_items) + gap * (len(self.menu_items) - 1)
        start_y = (self.WINDOW_H - total_h) // 2 + 30
        self.menu_button_rects = []
        for i in range(len(self.menu_items)):
            r = pygame.Rect(0, 0, button_w, button_h)
            r.center = (cx, start_y + i * (button_h + gap))
            self.menu_button_rects.append(r)

    def _activate_menu(self, index: int) -> None:
        _, label = self.menu_items[index]
        if "Quit" in label:
            self.state = State.QUIT
        elif "Play as Human" in label:
            self.mode = "human"
            self.state = State.LEVELS
        elif "Watch AI" in label:
            self.mode = "ai"
            self.model_paths = _discover_models()
            self.model_index = 0
            self.state = State.MODELS

    def _draw_menu(self) -> None:
        # Defensive: ensure button rects are laid out even if _draw_menu
        # is invoked outside the normal _run_menu() flow (e.g. tests).
        if not self.menu_button_rects:
            self._layout_menu()

        self.screen.fill(BG_CREAM)

        # try:
        #     title = render_text_pil("SNAKE GAME", font_size=TITLE_FONT_SIZE, color=BLACK)
        #     title_rect = title.get_rect(center=(self.WINDOW_W // 2, 110))
        #     self.screen.blit(title, title_rect)
        #     sub = render_text_pil(
        #         "DRL · CLASSIC SNAKE", font_size=BODY_FONT_SIZE, color=BLACK
        #     )
        #     sub_rect = sub.get_rect(center=(self.WINDOW_W // 2, 160))
        #     self.screen.blit(sub, sub_rect)
        # except Exception:
        #     pass

        for i, (_, label) in enumerate(self.menu_items):
            rect = self.menu_button_rects[i]
            # No fill override → inherits BG_CREAM (blends into the menu bg).
            # Inversion happens automatically when selected/hovered.
            btn = Button(rect, label)
            btn.selected = (i == self.menu_index)
            btn.draw(self.screen)

        try:
            hint = render_text_pil(
                "UP/DOWN navigate   ·   ENTER select", font_size=16, color=BLACK
            )
            self.screen.blit(hint, (20, self.WINDOW_H - 30))
        except Exception:
            pass

        pygame.display.flip()

    # ------------------------------------------------------------------
    # MODELS
    # ------------------------------------------------------------------
    def _run_models(self) -> None:
        if not self.model_paths:
            self._draw_models_empty()
            waiting = True
            while waiting and self.state == State.MODELS:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.state = State.QUIT
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
                            self.state = State.MENU
                            waiting = False
                self.clock.tick(30)
            return

        list_rect = pygame.Rect(40, 200, self.WINDOW_W - 80, self.WINDOW_H - 280)
        items = [_filename(p) for p in self.model_paths]
        self.model_listbox = ListBox(list_rect, items)
        self.model_listbox.selected_index = min(self.model_index, len(items) - 1)

        running = True
        while running and self.state == State.MODELS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state = State.QUIT
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.state = State.MENU
                        running = False
                        continue
                    self.model_listbox.handle_event(event)
                    self.model_index = self.model_listbox.selected_index
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self._activate_model(self.model_index)
                        running = False

            self._draw_models()
            self.clock.tick(30)

    def _activate_model(self, index: int) -> None:
        if index >= len(self.model_paths):
            self.state = State.MENU
            return
        self.model_path = self.model_paths[index]
        self.state = State.LEVELS

    def _draw_models(self) -> None:
        self.screen.fill(BG_CREAM)
        self._draw_title_bar("Select Model", subtitle="")
        if self.model_listbox is not None:
            self.model_listbox.draw(self.screen)
        self._draw_footer_hint("UP/DOWN navigate   ·   ENTER select   ·   ESC back")
        pygame.display.flip()

    def _draw_models_empty(self) -> None:
        self.screen.fill(BG_CREAM)
        self._draw_title_bar("Select Model", subtitle="")
        msg_rect = pygame.Rect(40, 220, self.WINDOW_W - 80, 200)
        Panel(msg_rect, fill=WHITE).draw(self.screen)
        try:
            line1 = render_text_pil("No models found.", font_size=28, color=BLACK)
            line2 = render_text_pil(
                "Train a model first — it should appear in",
                font_size=18, color=BLACK,
            )
            line3 = render_text_pil(
                "the saved_models/ folder as a .zip file.",
                font_size=18, color=BLACK,
            )
            self.screen.blit(line1, line1.get_rect(center=(msg_rect.centerx, msg_rect.y + 60)))
            self.screen.blit(line2, line2.get_rect(center=(msg_rect.centerx, msg_rect.y + 110)))
            self.screen.blit(line3, line3.get_rect(center=(msg_rect.centerx, msg_rect.y + 140)))
        except Exception:
            pass
        self._draw_footer_hint("ENTER / ESC = back to menu")
        pygame.display.flip()

    # ------------------------------------------------------------------
    # LEVELS
    # ------------------------------------------------------------------
    def _run_levels(self) -> None:
        self._layout_levels()
        self.level_index = 0

        running = True
        while running and self.state == State.LEVELS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state = State.QUIT
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # Back goes to MODELS for AI, MENU for human
                        if self.mode == "ai":
                            self.state = State.MODELS
                        else:
                            self.state = State.MENU
                        running = False
                        continue
                    if event.key == pygame.K_UP:
                        self.level_index = max(0, self.level_index - 1)
                    elif event.key == pygame.K_DOWN:
                        self.level_index = min(len(self.level_items) - 1, self.level_index + 1)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        self._activate_level(self.level_index)
                        running = False
                else:
                    if event.type == pygame.MOUSEMOTION:
                        for i, rect in enumerate(self.level_button_rects):
                            if rect.collidepoint(event.pos):
                                self.level_index = i
                                break
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        for i, rect in enumerate(self.level_button_rects):
                            if rect.collidepoint(event.pos):
                                self.level_index = i
                                self._activate_level(self.level_index)
                                running = False
                                break

            self._draw_levels()
            self.clock.tick(30)

    def _layout_levels(self) -> None:
        """Same flush-stack pattern as the main menu (gap=0, no border)."""
        cx = self.WINDOW_W // 2
        button_w = 270
        button_h = 33
        gap = 0
        n = len(self.level_items)
        total_h = button_h * n + gap * (n - 1)
        start_y = (self.WINDOW_H - total_h) // 2 + 40
        self.level_button_rects = []
        for i in range(n):
            r = pygame.Rect(0, 0, button_w, button_h)
            r.center = (cx, start_y + i * (button_h + gap))
            self.level_button_rects.append(r)

    def _activate_level(self, index: int) -> None:
        self.level = self.level_items[index][0]
        self.state = State.FPS_MODAL

    def _draw_levels(self) -> None:
        # Defensive: ensure button rects exist even if _draw_levels is
        # called from a path that didn't run _run_levels (e.g. tests,
        # or _run_fps_modal drawing the levels screen underneath).
        if not self.level_button_rects:
            self._layout_levels()

        self.screen.fill(BG_CREAM)
        self._draw_title_bar(
            "SELECT LEVEL",
            subtitle=f"Model: {_filename(self.model_path) if self.model_path else 'N/A'}",
        )

        for i, (_, label) in enumerate(self.level_items):
            rect = self.level_button_rects[i]
            # Inherits BG_CREAM fill; inverts to BLACK + BG_CREAM text on select.
            btn = Button(rect, label)
            btn.selected = (i == self.level_index)
            btn.draw(self.screen)

        self._draw_footer_hint("UP/DOWN navigate   ·   ENTER select   ·   ESC back")
        pygame.display.flip()

    # ------------------------------------------------------------------
    # FPS_MODAL
    # ------------------------------------------------------------------
    def _run_fps_modal(self) -> None:
        self.fps_input = TextInput(
            pygame.Rect(0, 0, 120, 50),
            initial_text=str(self.fps),
            max_length=3,
        )
        ok_button = Button(
            pygame.Rect(0, 0, BUTTON_MIN_WIDTH, 30),
            "Start",
            on_click=lambda: self._commit_fps(),
        )
        cancel_button = Button(
            pygame.Rect(0, 0, BUTTON_MIN_WIDTH, 30),
            "Back",
            on_click=lambda: self._cancel_fps(),
        )
        modal_buttons = [ok_button, cancel_button]
        selected_btn = 0            # 0 = Start, 1 = Back
        focus = "input"             # "input" or "buttons"

        modal = Modal(
            (self.WINDOW_W, self.WINDOW_H),
            title="SET FPS",
            width_ratio=float(_cfg.UI["MODAL_WIDTH_RATIO"]),
            height_ratio=float(_cfg.UI["MODAL_HEIGHT_RATIO"]),
        )

        # Layout — relative to modal center so it works for any size
        input_rect = pygame.Rect(0, 0, 100, 50)
        input_rect.center = (modal.rect.centerx, modal.rect.centery - 10)
        self.fps_input.rect = input_rect
        self.fps_input.focused = True

        ok_rect = pygame.Rect(0, 0, BUTTON_MIN_WIDTH, 56)
        cancel_rect = pygame.Rect(0, 0, BUTTON_MIN_WIDTH, 56)
        button_y = modal.rect.centery + 50
        ok_rect.midright = (modal.rect.centerx - 10, button_y)
        cancel_rect.midleft = (modal.rect.centerx + 10, button_y)
        ok_button.rect = ok_rect
        cancel_button.rect = cancel_rect

        modal.add(self.fps_input)
        modal.add(ok_button)
        modal.add(cancel_button)
        self.fps_modal = modal

        running = True
        while running and self.state == State.FPS_MODAL:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state = State.QUIT
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.state = State.LEVELS
                        running = False
                        continue
                    if event.key == pygame.K_TAB:
                        # Move focus between the text input and the button row
                        focus = "buttons" if focus == "input" else "input"
                        self.fps_input.focused = (focus == "input")
                        continue
                    if focus == "buttons":
                        if event.key in (pygame.K_LEFT,):
                            selected_btn = (selected_btn - 1) % len(modal_buttons)
                        elif event.key in (pygame.K_RIGHT,):
                            selected_btn = (selected_btn + 1) % len(modal_buttons)
                        elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                            modal_buttons[selected_btn].on_click()
                            running = False
                            continue
                    else:
                        # Focus is on the text input — let it handle the key
                        if event.key == pygame.K_RETURN:
                            self._commit_fps()
                            running = False
                            continue
                        self.fps_input.handle_event(event)
                else:
                    # Mouse / other events
                    self.fps_modal.handle_event(event)
                    for btn in modal_buttons:
                        if btn.clicked_this_frame():
                            running = False

            # Refresh "selected" flag each frame for the current button
            for i, btn in enumerate(modal_buttons):
                btn.selected = (focus == "buttons" and i == selected_btn)

            # Draw underlying LEVELS screen, then the modal on top
            self._draw_levels()
            self.fps_modal.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(30)

    def _commit_fps(self) -> None:
        raw = self.fps_input.text.strip() if self.fps_input else ""
        try:
            value = int(raw) if raw else int(_cfg.FPS_DEFAULT)
        except ValueError:
            value = int(_cfg.FPS_DEFAULT)
        value = max(int(_cfg.FPS_MIN), min(int(_cfg.FPS_MAX), value))
        self.fps = value
        self.state = State.PLAYING

    def _cancel_fps(self) -> None:
        self.state = State.LEVELS

    # ------------------------------------------------------------------
    # PLAYING
    # ------------------------------------------------------------------
    def _run_playing(self) -> None:
        # Detect a "resume from RETURN_MODAL" call. After the pause
        # modal's Continue button is pressed, ``run()`` re-dispatches
        # us here with state==PLAYING, but the env, controller, and
        # renderer are still alive (frozen by the modal). Skipping the
        # full init preserves the snake's position, score, AI belief
        # state, and controller input buffer — so Continue actually
        # continues, instead of wiping the game via env.reset().
        resuming = (
            self.env is not None
            and self.controller is not None
            and self.renderer is not None
            and self.renderer_screen is not None
        )

        if not resuming:
            # Fresh-start path: build controller, load model (AI mode),
            # then build env + renderer.
            self.controller = input_controller()
            if self.mode == "ai":
                self.controller.game_started = True

            obs_type = "spatiotemporal"  # safe default for human mode
            if self.mode == "ai" and self.model_path:
                try:
                    model_info = self.controller.load_model(self.model_path)
                    obs_type = model_info["obs_type"]
                    print(
                        f"[launcher] Model requires obs_type={obs_type!r} "
                        f"(algo={model_info['algo']}, shape={model_info['obs_shape']})"
                    )
                except Exception as exc:
                    print(
                        f"[launcher] Could not load model: {exc!r}. "
                        f"Falling back to random AI."
                    )
                    self.controller.ai_model = None

            self._init_game(obs_type=obs_type)
            assert (
                self.env is not None
                and self.renderer is not None
                and self.controller is not None
            )
        else:
            # Resume path: env/controller/renderer already exist and
            # are frozen. The modal's event loop consumed any input
            # events that arrived while it was up, so the controller's
            # current/queued action is still the last one the player
            # applied before pressing R.
            assert (
                self.env is not None
                and self.renderer is not None
                and self.controller is not None
            )

        running = True
        while running and self.state == State.PLAYING:
            # 0. System-level keys (work in both human + AI mode).
            # ``R`` pops the pause/return modal. Drained separately
            # so it doesn't leak into the modal's event loop later.
            for event in pygame.event.get(pygame.KEYDOWN):
                if event.key == pygame.K_r:
                    self.state = State.RETURN_MODAL
                    running = False
                    break
            if not running or self.state != State.PLAYING:
                continue

            # 1. Input
            if self.mode == "human":
                action, keep_running = self.controller.process_human_input()
                if not keep_running:
                    self.state = State.MENU
                    break
                # Controller's ``return_requested`` flag is the human-
                # mode hook for the same pause key the AI branch catches
                # above (the controller owns its own event loop).
                if self.controller.return_requested:
                    self.controller.return_requested = False
                    self.state = State.RETURN_MODAL
                    running = False
                    continue
                if action is None:
                    if self.controller.game_started:
                        obs, _ = self.env.reset(seed=None)
                        self.controller.reset_state()
                else:
                    if self.controller.game_started:
                        obs, reward, term, trunc, info = self.env.step(action)
                        if term or trunc:
                            reason = info.get("reason", "collision")
                            if reason == "win":
                                self.win_score = len(self.env.snake) - 3
                                self.state = State.WIN_MODAL
                            else:
                                self.death_score = len(self.env.snake) - 3
                                self.death_reason = reason
                                self.state = State.DEATH_MODAL
                            running = False
                            continue
            else:  # AI mode
                action = self.controller.process_ai_input(self._last_obs)
                obs, reward, term, trunc, info = self.env.step(action)
                self._last_obs = obs
                if term or trunc:
                    reason = info.get("reason", "collision")
                    if reason == "win":
                        self.win_score = len(self.env.snake) - 3
                        self.state = State.WIN_MODAL
                    else:
                        self.death_score = len(self.env.snake) - 3
                        self.death_reason = reason
                        self.state = State.DEATH_MODAL
                    running = False
                    continue
            # 2. Draw — game renders to its own surface, then we blit
            # it centered into the launcher's single window
            self.renderer.draw_frame(
                self.renderer_screen,
                self.env,
                game_started=self.controller.game_started,
            )
            self.screen.fill(BG_CREAM)
            self.screen.blit(
                self.renderer_screen,
                (self.game_offset_x, self.game_offset_y),
            )
            pygame.display.flip()
            self.renderer.tick(self.fps)

        if self.state == State.MENU:
            self._teardown_game()

    def _init_game(self, obs_type: str = "spatiotemporal") -> None:
        # obs_type is decided by the caller:
        #   - human mode → "spatiotemporal" (default; doesn't matter for rendering)
        #   - AI mode    → from controller.load_model(...) return value,
        #                  so the env's observation_space exactly matches
        #                  what the loaded policy was trained on.
        assert obs_type in ("spatiotemporal", "spatiotemporal_legacy", "12bit"), (
            f"Unsupported obs_type: {obs_type!r}"
        )
        # Both gameplay modes (human AND AI) get an effectively-uncapped
        # step budget. MAX_GAME_STEPS=500 is a TRAINING-time device only:
        # the trainer calls game_environment() without overriding
        # max_steps, so the env default of _cfg.MAX_GAME_STEPS applies.
        # For gameplay the only "exit" conditions should be collision or
        # the user pressing R — not the clock running out.
        max_steps = 10_000
        self.env = game_environment(
            level=self.level, obs_type=obs_type, max_steps=max_steps,
        )
        obs, _ = self.env.reset(seed=None)
        self._last_obs = obs

        # Use DEFAULT_CELL_SIZE directly so the cell size is controlled
        # by config. The game area is `cell_size * grid + 2 * wall`.
        grid = int(_cfg.GRID_SIZE)
        wall = int(_cfg.DEFAULT_WALL_THICKNESS)
        cell_size = int(_cfg.DEFAULT_CELL_SIZE)

        self.renderer = game_renderer(
            cell_size=cell_size,
            grid_size=grid,
            wall_thickness=wall,
            fps=self.fps,
        )

        # Resize the window to fit the game with 20px cream padding
        # around it, so the entire grid is visible.
        game_w = self.renderer.window_w
        game_h = self.renderer.window_h
        padding = 0
        new_w = game_w + 2 * padding
        new_h = game_h + 2 * padding
        self.screen = pygame.display.set_mode((new_w, new_h))
        self.game_padding = padding
        self.game_offset_x = padding
        self.game_offset_y = padding

        # Game surface (separate, game-sized) — the renderer draws to
        # this, then we blit it centered into the resized launcher screen.
        self.renderer_screen = pygame.Surface((game_w, game_h))
        pygame.display.set_caption(
            f"Level {self.level} — {'AI' if self.mode == 'ai' else 'Human'}"
        )

        # ``self.controller`` is created in ``_run_playing`` (before
        # the model load) so it already exists at this point — don't
        # overwrite it here.

    def _teardown_game(self) -> None:
        self.env = None
        self.renderer = None
        self.controller = None
        self._last_obs = None
        self.renderer_screen = None
        # Resize the window back to launcher dimensions.
        self.screen = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H))
        pygame.display.set_caption("SNAKE GAME")
        self.screen.fill(BG_CREAM)
        pygame.display.flip()

    # ------------------------------------------------------------------
    # DEATH_MODAL
    # ------------------------------------------------------------------
    def _run_death_modal(self) -> None:
        assert self.renderer_screen is not None

        # Use the ACTUAL current screen size (not the launcher size) —
        # the game has resized the window in _init_game, so this picks
        # up the new dimensions and centers the modal correctly.
        screen_w, screen_h = self.screen.get_size()

        # Auto-size the modal to fit its content (title, score, 2 buttons).
        # Title reflects the actual cause: collision death vs clock timeout.
        is_truncated = getattr(self, "death_reason", "collision") == "truncated"
        title_str = "TIME UP" if is_truncated else "YOU DIED"
        try:
            title_surf = render_text_pil(title_str, font_size=44, color=BLACK)
            title_w, title_h = title_surf.get_size()
        except Exception:
            title_w, title_h = 200, 50
        try:
            score_text = f"Final score: {self.death_score}"
            score_surf = render_text_pil(score_text, font_size=24, color=BLACK)
            score_w, score_h = score_surf.get_size()
        except Exception:
            score_w, score_h = 220, 30

        button_w = BUTTON_MIN_WIDTH + 20
        button_h = 56
        pad = 80
        gap = 10
        content_w = max(title_w, score_w, 2 * button_w + gap) + 2 * pad
        content_h = title_h + score_h + button_h + 2 * gap + 1 * pad
        # Cap to 60% of the screen so it never feels overwhelming
        max_w = int(screen_w * 0.6)
        max_h = int(screen_h * 0.4)
        modal_w = max(380, min(content_w, max_w))
        modal_h = max(220, min(content_h, max_h))

        modal = Modal(
            (screen_w, screen_h),
            title=title_str,
            fixed_size=(modal_w, modal_h),
        )

        # Buttons
        retry_btn = Button(
            pygame.Rect(0, 0, button_w, button_h),
            "Retry",
            on_click=lambda: self._death_choice("retry"),
        )
        menu_btn = Button(
            pygame.Rect(0, 0, button_w, button_h),
            "Main Menu",
            on_click=lambda: self._death_choice("menu"),
        )
        modal_buttons = [retry_btn, menu_btn]
        selected_btn = 0  # 0 = Retry, 1 = Main Menu

        # Even vertical spread: 3 rows centered at 1/4, 1/2, 3/4 of the
        # modal height so the gaps between rows are equal regardless of
        # the modal's size. Title is drawn at 1/4 in the draw method.
        button_y = modal.rect.top + 3 * modal.rect.height // 4
        retry_btn.rect.midright = (modal.rect.centerx - 12, button_y)
        menu_btn.rect.midleft = (modal.rect.centerx + 12, button_y)

        self.death_modal = modal
        self.retry_button = retry_btn
        self.menu_button = menu_btn

        running = True
        while running and self.state == State.DEATH_MODAL:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state = State.QUIT
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._death_choice("menu")
                        running = False
                        continue
                    if event.key in (pygame.K_LEFT,):
                        selected_btn = (selected_btn - 1) % len(modal_buttons)
                    elif event.key in (pygame.K_RIGHT,):
                        selected_btn = (selected_btn + 1) % len(modal_buttons)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        modal_buttons[selected_btn].on_click()
                        running = False
                        continue
                # Forward mouse events to buttons
                if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                                  pygame.MOUSEMOTION):
                    for child in modal_buttons:
                        child.handle_event(event)
                for btn in modal_buttons:
                    if btn.clicked_this_frame():
                        running = False

            # Refresh "selected" flag each frame for the active button
            for i, btn in enumerate(modal_buttons):
                btn.selected = (i == selected_btn)

            # Re-draw the last game frame, blit it centered into the
            # launcher window, then overlay the modal on top.
            if self.env is not None and self.renderer is not None:
                self.renderer.draw_frame(
                    self.renderer_screen,
                    self.env,
                    game_started=True,
                )
            self.screen.fill(BG_CREAM)
            self.screen.blit(
                self.renderer_screen,
                (self.game_offset_x, self.game_offset_y),
            )
            self._draw_death_modal_on_screen()
            self.clock.tick(30)

        if self.state == State.MENU:
            self._teardown_game()
        elif self.state == State.PLAYING:
            self._reset_game()

    def _draw_death_modal_on_screen(self) -> None:
        # Draw on the LAUNCHER window (not the game surface) so the
        # dim + modal cover the whole screen, not just the game area.
        screen = self.screen
        modal = self.death_modal
        # Dim
        dim = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 160))
        screen.blit(dim, (0, 0))
        # Panel
        Panel(modal.rect, fill=PANEL_FILL).draw(screen)
        # Title — center of top quarter (matches the 1/4, 1/2, 3/4
        # row layout used for the buttons). "TIME UP" for truncation,
        # "YOU DIED" for collision — matches _run_death_modal's sizing.
        is_truncated = getattr(self, "death_reason", "collision") == "truncated"
        title_str = "TIME UP" if is_truncated else "YOU DIED"
        try:
            t = render_text_pil(title_str, font_size=44, color=BLACK)
            title_y = modal.rect.top + modal.rect.height // 4
            screen.blit(t, t.get_rect(center=(modal.rect.centerx, title_y)))
        except Exception:
            pass
        # Score — dead center of the modal
        try:
            s = render_text_pil(
                f"Final score: {self.death_score}",
                font_size=24, color=BLACK,
            )
            screen.blit(
                s, s.get_rect(center=(modal.rect.centerx, modal.rect.centery))
            )
        except Exception:
            pass
        # Buttons
        self.retry_button.draw(screen)
        self.menu_button.draw(screen)
        pygame.display.flip()

    def _death_choice(self, choice: str) -> None:
        if choice == "retry":
            self.state = State.PLAYING
        else:
            self.state = State.MENU

    # ------------------------------------------------------------------
    # WIN_MODAL  (snake filled the entire grid — terminal success)
    # ------------------------------------------------------------------
    def _run_win_modal(self) -> None:
        """Modal shown when the env returns ``reason="win"``.

        This is only reachable at level 1 where a single static food
        lets the snake grow until it occupies every cell. The same
        layout / input handling as the death modal, but with "YOU WIN!"
        title and a celebratory green accent. Two buttons: "Play Again"
        resets the env on the current level; "Main Menu" tears down
        and returns to the level picker.
        """
        assert self.renderer_screen is not None

        screen_w, screen_h = self.screen.get_size()

        try:
            title_surf = render_text_pil("YOU WIN!", font_size=44, color=BLACK)
            title_w, title_h = title_surf.get_size()
        except Exception:
            title_w, title_h = 200, 50
        try:
            score_text = f"Final score: {self.win_score}"
            score_surf = render_text_pil(score_text, font_size=24, color=BLACK)
            score_w, score_h = score_surf.get_size()
        except Exception:
            score_w, score_h = 220, 30

        button_w = BUTTON_MIN_WIDTH + 20
        button_h = 56
        pad = 80
        gap = 10
        content_w = max(title_w, score_w, 2 * button_w + gap) + 2 * pad
        content_h = title_h + score_h + button_h + 2 * gap + 1 * pad
        max_w = int(screen_w * 0.6)
        max_h = int(screen_h * 0.4)
        modal_w = max(380, min(content_w, max_w))
        modal_h = max(220, min(content_h, max_h))

        modal = Modal(
            (screen_w, screen_h),
            title="YOU WIN!",
            fixed_size=(modal_w, modal_h),
        )

        retry_btn = Button(
            pygame.Rect(0, 0, button_w, button_h),
            "Play Again",
            on_click=lambda: self._win_choice("retry"),
        )
        menu_btn = Button(
            pygame.Rect(0, 0, button_w, button_h),
            "Main Menu",
            on_click=lambda: self._win_choice("menu"),
        )
        modal_buttons = [retry_btn, menu_btn]
        selected_btn = 0  # 0 = Play Again, 1 = Main Menu

        button_y = modal.rect.top + 3 * modal.rect.height // 4
        retry_btn.rect.midright = (modal.rect.centerx - 12, button_y)
        menu_btn.rect.midleft = (modal.rect.centerx + 12, button_y)

        self.win_modal = modal
        self.win_retry_button = retry_btn
        self.win_menu_button = menu_btn

        running = True
        while running and self.state == State.WIN_MODAL:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state = State.QUIT
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._win_choice("menu")
                        running = False
                        continue
                    if event.key in (pygame.K_LEFT,):
                        selected_btn = (selected_btn - 1) % len(modal_buttons)
                    elif event.key in (pygame.K_RIGHT,):
                        selected_btn = (selected_btn + 1) % len(modal_buttons)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        modal_buttons[selected_btn].on_click()
                        running = False
                        continue
                if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                                  pygame.MOUSEMOTION):
                    for child in modal_buttons:
                        child.handle_event(event)
                for btn in modal_buttons:
                    if btn.clicked_this_frame():
                        running = False

            for i, btn in enumerate(modal_buttons):
                btn.selected = (i == selected_btn)

            # Re-draw the winning frame behind the modal.
            if self.env is not None and self.renderer is not None:
                self.renderer.draw_frame(
                    self.renderer_screen,
                    self.env,
                    game_started=True,
                )
            self.screen.fill(BG_CREAM)
            self.screen.blit(
                self.renderer_screen,
                (self.game_offset_x, self.game_offset_y),
            )
            self._draw_win_modal_on_screen()
            self.clock.tick(30)

        if self.state == State.MENU:
            self._teardown_game()
        elif self.state == State.PLAYING:
            self._reset_game()

    def _draw_win_modal_on_screen(self) -> None:
        """Render the win modal overlay on the LAUNCHER window."""
        screen = self.screen
        modal = self.win_modal
        # Dim the whole screen
        dim = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 160))
        screen.blit(dim, (0, 0))
        # Panel
        Panel(modal.rect, fill=PANEL_FILL).draw(screen)
        # Title at 1/4 — celebratory green-tinted black via the existing
        # BLACK palette (no new colour added; future-proof: drop in a
        # LIME from the UI palette if a brighter title is wanted).
        try:
            t = render_text_pil("YOU WIN!", font_size=44, color=BLACK)
            title_y = modal.rect.top + modal.rect.height // 4
            screen.blit(t, t.get_rect(center=(modal.rect.centerx, title_y)))
        except Exception:
            pass
        # Score at vertical center
        try:
            s = render_text_pil(
                f"Final score: {self.win_score}",
                font_size=24, color=BLACK,
            )
            screen.blit(
                s, s.get_rect(center=(modal.rect.centerx, modal.rect.centery))
            )
        except Exception:
            pass
        # Buttons at 3/4
        self.win_retry_button.draw(screen)
        self.win_menu_button.draw(screen)
        pygame.display.flip()

    def _win_choice(self, choice: str) -> None:
        if choice == "retry":
            self.state = State.PLAYING
        else:
            self.state = State.MENU

    # ------------------------------------------------------------------
    # RETURN_MODAL  (paused — R pressed during PLAYING)
    # ------------------------------------------------------------------
    def _run_return_modal(self) -> None:
        """
        Pause modal: offers "Continue" or "Main Menu".

        Triggered by ``R`` during :data:`State.PLAYING` in either
        human or AI mode. The env is *not* stepped while this modal
        is up — only the last game frame is re-rendered behind it so
        the player sees a frozen snapshot of the game.
        """
        assert self.renderer_screen is not None

        screen_w, screen_h = self.screen.get_size()

        # Auto-size the modal to fit title + 2 buttons.
        try:
            title_surf = render_text_pil("PAUSED", font_size=44, color=BLACK)
            title_w, title_h = title_surf.get_size()
        except Exception:
            title_w, title_h = 200, 50

        button_w = BUTTON_MIN_WIDTH + 20
        button_h = 56
        pad = 80
        gap = 10
        content_w = max(title_w, 2 * button_w + gap) + 2 * pad
        content_h = title_h + button_h + 2 * gap + 1 * pad
        max_w = int(screen_w * 0.6)
        max_h = int(screen_h * 0.4)
        modal_w = max(380, min(content_w, max_w))
        modal_h = max(200, min(content_h, max_h))

        modal = Modal(
            (screen_w, screen_h),
            title="PAUSED",
            fixed_size=(modal_w, modal_h),
        )

        continue_btn = Button(
            pygame.Rect(0, 0, button_w, button_h),
            "Continue",
            on_click=lambda: self._return_choice("continue"),
        )
        menu_btn = Button(
            pygame.Rect(0, 0, button_w, button_h),
            "Main Menu",
            on_click=lambda: self._return_choice("menu"),
        )
        modal_buttons = [continue_btn, menu_btn]
        selected_btn = 0  # 0 = Continue, 1 = Main Menu

        # Same 1/4 + 3/4 row layout as the death modal — title at 1/4,
        # buttons centred at 3/4.
        button_y = modal.rect.top + 3 * modal.rect.height // 4
        continue_btn.rect.midright = (modal.rect.centerx - 12, button_y)
        menu_btn.rect.midleft = (modal.rect.centerx + 12, button_y)

        self.return_modal = modal
        self.continue_button = continue_btn
        self.return_menu_button = menu_btn

        running = True
        while running and self.state == State.RETURN_MODAL:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.state = State.QUIT
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # ESC = leave the game (same as "Main Menu").
                        self._return_choice("menu")
                        running = False
                        continue
                    if event.key == pygame.K_r:
                        # R toggles: press again → resume.
                        self._return_choice("continue")
                        running = False
                        continue
                    if event.key in (pygame.K_LEFT,):
                        selected_btn = (selected_btn - 1) % len(modal_buttons)
                    elif event.key in (pygame.K_RIGHT,):
                        selected_btn = (selected_btn + 1) % len(modal_buttons)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        modal_buttons[selected_btn].on_click()
                        running = False
                        continue
                # Forward mouse events to buttons
                if event.type in (
                    pygame.MOUSEBUTTONDOWN,
                    pygame.MOUSEBUTTONUP,
                    pygame.MOUSEMOTION,
                ):
                    for child in modal_buttons:
                        child.handle_event(event)
                for btn in modal_buttons:
                    if btn.clicked_this_frame():
                        running = False

            # Refresh "selected" flag each frame for the active button.
            for i, btn in enumerate(modal_buttons):
                btn.selected = (i == selected_btn)

            # Re-draw the last game frame, blit it centered into the
            # launcher window, then overlay the modal on top. The env
            # is NOT stepped — the frame is a frozen snapshot of the
            # snake's position when R was pressed.
            if self.env is not None and self.renderer is not None:
                self.renderer.draw_frame(
                    self.renderer_screen,
                    self.env,
                    game_started=self.controller.game_started,
                )
            self.screen.fill(BG_CREAM)
            self.screen.blit(
                self.renderer_screen,
                (self.game_offset_x, self.game_offset_y),
            )
            self._draw_return_modal_on_screen()
            self.clock.tick(30)

        if self.state == State.MENU:
            self._teardown_game()
        elif self.state == State.PLAYING:
            # Resume — env was frozen during the modal, no reset needed.
            # The controller's input state (current/queued action) is
            # already correct because the modal's event loop consumed
            # any arrow keys pressed while it was up.
            pass

    def _draw_return_modal_on_screen(self) -> None:
        """Draw the PAUSED modal overlay on the launcher window."""
        screen = self.screen
        modal = self.return_modal
        # Dim
        dim = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 160))
        screen.blit(dim, (0, 0))
        # Panel
        Panel(modal.rect, fill=PANEL_FILL).draw(screen)
        # Title — center of top quarter (same row layout as buttons).
        try:
            t = render_text_pil("PAUSED", font_size=44, color=BLACK)
            title_y = modal.rect.top + modal.rect.height // 4
            screen.blit(t, t.get_rect(center=(modal.rect.centerx, title_y)))
        except Exception:
            pass
        # Buttons
        self.continue_button.draw(screen)
        self.return_menu_button.draw(screen)
        pygame.display.flip()

    def _return_choice(self, choice: str) -> None:
        """
        Route the pause-modal choice.

        ``"continue"`` → resume the frozen game (no env reset).
        ``"menu"``     → tear down and return to the launcher menu.
        """
        if choice == "continue":
            self.state = State.PLAYING
        else:
            self.state = State.MENU

    def _reset_game(self) -> None:
        assert self.env is not None and self.controller is not None
        obs, _ = self.env.reset(seed=None)
        self._last_obs = obs
        self.controller.reset_state()
        if self.mode == "ai":
            self.controller.game_started = True

    # ------------------------------------------------------------------
    # Shared drawing helpers
    # ------------------------------------------------------------------
    def _draw_title_bar(self, title: str, subtitle: str = "") -> None:
        try:
            t = render_text_pil(title, font_size=44, color=BLACK)
            if subtitle:
                self.screen.blit(t, t.get_rect(center=(self.WINDOW_W // 2, 90)))
                s = render_text_pil(subtitle, font_size=18, color=BLACK)
                self.screen.blit(s, s.get_rect(center=(self.WINDOW_W // 2, 130)))
            else:
                self.screen.blit(t, t.get_rect(center=(self.WINDOW_W // 2, 110)))
        except Exception:
            pass

    def _draw_footer_hint(self, hint: str) -> None:
        try:
            surf = render_text_pil(hint, font_size=16, color=BLACK)
            self.screen.blit(surf, (20, self.WINDOW_H - 30))
        except Exception:
            pass


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    Launcher().run()
