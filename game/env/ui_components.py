"""
Neubrutal UI Components
=======================

Reusable widgets for the game launcher. Aesthetic = neubrutalism:
    - 4px solid black borders
    - 6px solid black drop shadow (no blur, no gradient)
    - Flat high-contrast colors
    - Square corners
    - "Pressed" feedback: element shifts 4px and shadow disappears

Components:
    Panel     - bordered rectangle with drop shadow
    Button    - clickable panel with text + pressed feedback
    ListBox   - vertical scrollable list with selection highlight
    TextInput - single-line numeric text input with blinking cursor
    Modal     - dimmed overlay containing other widgets

All coordinates use pygame.Rect. Colors come from config.UI.PALETTE
via the central `config` singleton in game.env.config.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import pygame

from game.env.config import config as _cfg
from game.env.game_renderer import render_text_pil


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
Palette = _cfg.UI["PALETTE"]


def _c(name: str) -> Tuple[int, int, int]:
    """Look up a color by name from UI.PALETTE."""
    return tuple(Palette[name])


BLACK = _c("BLACK")
WHITE = _c("WHITE")
BG_CREAM = _c("BG_CREAM")
YELLOW = _c("YELLOW")
PINK = _c("PINK")
LIME = _c("LIME")
CYAN = _c("CYAN")
CORAL = _c("CORAL")
PANEL_FILL = _c("PANEL_FILL")
DIM = _c("DIM")

# Layout constants
BORDER_WIDTH = int(_cfg.UI["BORDER_WIDTH"])
SHADOW_OFFSET = int(_cfg.UI["SHADOW_OFFSET"])
ITEM_HEIGHT = int(_cfg.UI["ITEM_HEIGHT"])
ITEM_GAP = int(_cfg.UI["ITEM_GAP"])
BUTTON_MIN_WIDTH = int(_cfg.UI["BUTTON_MIN_WIDTH"])
TITLE_FONT_SIZE = int(_cfg.UI["TITLE_FONT_SIZE"])
ITEM_FONT_SIZE = int(_cfg.UI["ITEM_FONT_SIZE"])
BUTTON_FONT_SIZE = int(_cfg.UI["BUTTON_FONT_SIZE"])
BODY_FONT_SIZE = int(_cfg.UI["BODY_FONT_SIZE"])


# ===========================================================================
# Panel — bordered rect with hard drop shadow
# ===========================================================================
class Panel:
    """Solid-fill rectangle with a thick black border and an offset shadow."""

    def __init__(
        self,
        rect: pygame.Rect,
        fill: Tuple[int, int, int] = PANEL_FILL,
        border: Tuple[int, int, int] = BLACK,
        border_width: int = BORDER_WIDTH,
        shadow_offset: int = SHADOW_OFFSET,
    ):
        self.rect = pygame.Rect(rect)
        self.fill = fill
        self.border = border
        self.border_width = border_width
        self.shadow_offset = shadow_offset
        self.pressed = False  # when True, shadow is hidden (pressed-in look)

    def draw(self, screen: pygame.Surface) -> None:
        # Shadow first (so it sits behind everything)
        if not self.pressed:
            shadow = self.rect.move(self.shadow_offset, self.shadow_offset)
            pygame.draw.rect(screen, BLACK, shadow)
        # Fill
        pygame.draw.rect(screen, self.fill, self.rect)
        # Border
        pygame.draw.rect(screen, self.border, self.rect, self.border_width)


# ===========================================================================
# Button — flat, tonally-restrained widget with hover/select inversion
# ===========================================================================
class Button:
    """
    A clickable button with text. Two visual states:

        Unselected (default):
            - Flat fill in the button's color (e.g. YELLOW)
            - Black text
            - No border, no drop shadow

        Selected (hover OR keyboard-focused):
            - Black fill (inverted)
            - Text rendered in the original color
            - 4px border in the original color (a thin "ring" of color
              around the black button — reads as a selection outline)

    The caller is responsible for setting `selected = True/False` per
    frame based on its focus-tracking logic. Mouse hover also flips
    `selected` for convenience, so the same look-and-feel works for
    both input methods.

    `clicked_this_frame()` returns True exactly once, on the frame the
    left mouse button is released over the button — used by the
    launcher's event loops to detect "Enter pressed" via mouse.
    """

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        fill: Tuple[int, int, int] = BG_CREAM,
        text_color: Tuple[int, int, int] = BLACK,
        font_size: int = BUTTON_FONT_SIZE,
        on_click: Optional[Callable[[], None]] = None,
    ):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.fill_color = fill        # the "base" color (also used as text/border when inverted)
        self.text_color = text_color  # text color in the unselected state (default BLACK)
        self.font_size = font_size
        self.on_click = on_click

        # Visual state
        self.selected: bool = False   # keyboard-focused / selected by the parent
        self.hovered: bool = False    # mouse is over the button
        self._pressed: bool = False
        self._was_pressed: bool = False

    # Convenience: "is the button currently visually active?"
    @property
    def is_active(self) -> bool:
        return self.selected or self.hovered

    def handle_event(self, event: pygame.event.Event) -> None:
        """Forward mouse events. Keyboard focus is set by the parent."""
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self._pressed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._pressed = False

    def clicked_this_frame(self) -> bool:
        """Return True exactly once, on the frame the button is released over."""
        was_over = self._was_pressed and self.rect.collidepoint(pygame.mouse.get_pos())
        self._was_pressed = self._pressed
        if was_over and not self._pressed:
            if self.on_click is not None:
                self.on_click()
            return True
        return False

    def draw(self, screen: pygame.Surface) -> None:
        if self.is_active:
            # Inverted look: black fill, cream text, no border
            fill = BLACK
            text_c = BG_CREAM
        else:
            # Flat look: color fill, black text, no border / no shadow
            fill = self.fill_color
            text_c = self.text_color

        # Fill (no border at all — even when selected)
        pygame.draw.rect(screen, fill, self.rect)

        # Text
        try:
            surf = render_text_pil(self.text, font_size=self.font_size, color=text_c)
            text_rect = surf.get_rect(center=self.rect.center)
            screen.blit(surf, text_rect)
        except Exception:
            pass


# ===========================================================================
# ListBox — vertical scrollable list with selection highlight
# ===========================================================================
class ListBox:
    """
    A vertical list of items. The selected item is highlighted with a flat
    yellow fill and a black border. Navigation is via keyboard
    (Up/Down/Home/End/PageUp/PageDown).
    """

    def __init__(
        self,
        rect: pygame.Rect,
        items: List[str],
        item_height: int = ITEM_HEIGHT,
        on_select: Optional[Callable[[int], None]] = None,
        on_activate: Optional[Callable[[int], None]] = None,
    ):
        self.rect = pygame.Rect(rect)
        self.items = items
        self.item_height = item_height
        self.selected_index = 0
        self.scroll_offset = 0
        self.on_select = on_select
        self.on_activate = on_activate
        self._triggered_select = False

    def set_items(self, items: List[str]) -> None:
        self.items = items
        self.selected_index = 0
        self.scroll_offset = 0

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if not self.items:
                return
            prev = self.selected_index
            if event.key == pygame.K_UP:
                self.selected_index = max(0, self.selected_index - 1)
            elif event.key == pygame.K_DOWN:
                self.selected_index = min(len(self.items) - 1, self.selected_index + 1)
            elif event.key == pygame.K_HOME:
                self.selected_index = 0
            elif event.key == pygame.K_END:
                self.selected_index = len(self.items) - 1
            elif event.key == pygame.K_PAGEUP:
                self.selected_index = max(0, self.selected_index - 5)
            elif event.key == pygame.K_PAGEDOWN:
                self.selected_index = min(len(self.items) - 1, self.selected_index + 5)
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                if self.on_activate is not None:
                    self.on_activate(self.selected_index)
            if self.selected_index != prev and self.on_select is not None:
                self.on_select(self.selected_index)
            self._ensure_visible()

    def _ensure_visible(self) -> None:
        """Scroll so the selected item is in view."""
        sel_top = self.selected_index * self.item_height
        sel_bottom = sel_top + self.item_height
        if sel_top < self.scroll_offset:
            self.scroll_offset = sel_top
        elif sel_bottom > self.scroll_offset + self.rect.height:
            self.scroll_offset = sel_bottom - self.rect.height

    def draw(self, screen: pygame.Surface) -> None:
        """
        Visual style:
            - 4px black border around the outer ListBox rect.
            - Unselected items: BG_CREAM fill (blends into the cream
              background → only the black text is visible).
            - Selected item: BLACK fill with BG_CREAM text (inverted),
              no border.
        """
        # Clip to the ListBox area so items don't bleed outside the rect
        screen.set_clip(self.rect)

        y = self.rect.top - self.scroll_offset
        for i, item in enumerate(self.items):
            item_rect = pygame.Rect(
                self.rect.left,
                y,
                self.rect.width,
                self.item_height,
            )
            if item_rect.bottom > self.rect.top and item_rect.top < self.rect.bottom:
                if i == self.selected_index:
                    # Inverted: black fill, cream text, no border
                    pygame.draw.rect(screen, BLACK, item_rect)
                    text_color = BG_CREAM
                else:
                    # Unselected: cream fill, no border (blends with bg)
                    pygame.draw.rect(screen, BG_CREAM, item_rect)
                    text_color = BLACK
                # Text
                try:
                    surf = render_text_pil(
                        item, font_size=ITEM_FONT_SIZE, color=text_color
                    )
                    text_rect = surf.get_rect(midleft=(item_rect.left + 16, item_rect.centery))
                    screen.blit(surf, text_rect)
                except Exception:
                    pass
            y += self.item_height

        screen.set_clip(None)

        # Outer border (drawn after items so it sits on top)
        pygame.draw.rect(screen, BLACK, self.rect, BORDER_WIDTH)

        screen.set_clip(None)

    @property
    def panel_border_inset(self) -> int:
        return BORDER_WIDTH


# ===========================================================================
# TextInput — single-line numeric input with blinking cursor
# ===========================================================================
class TextInput:
    """A single-line text input. Filters characters via an `allowed` predicate."""

    def __init__(
        self,
        rect: pygame.Rect,
        initial_text: str = "",
        max_length: int = 4,
        allowed: Callable[[str], bool] = lambda ch: ch.isdigit(),
        placeholder: str = "",
    ):
        self.rect = pygame.Rect(rect)
        self.text = initial_text
        self.max_length = max_length
        self.allowed = allowed
        self.placeholder = placeholder
        self.focused = False
        self._cursor_timer = 0
        self._cursor_visible = True

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.focused = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN and self.focused:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                # Caller checks .text / .value; Enter doesn't auto-commit
                pass
            elif event.unicode and self.allowed(event.unicode) and len(self.text) < self.max_length:
                self.text += event.unicode

    def update(self, dt_ms: int = 0) -> None:
        if dt_ms:
            self._cursor_timer += dt_ms
            if self._cursor_timer >= 500:
                self._cursor_timer = 0
                self._cursor_visible = not self._cursor_visible

    def draw(self, screen: pygame.Surface) -> None:
        panel = Panel(self.rect, fill=WHITE)
        panel.draw(screen)

        # Cursor (only when focused and blinking)
        cursor_x_offset = 0
        if self.focused:
            # self._cursor_visible = (pygame.time.get_ticks() // 500) % 2 == 0
            self._cursor_visible = True
            if self._cursor_visible and self.text:
                try:
                    pre = render_text_pil(self.text, font_size=BUTTON_FONT_SIZE, color=BLACK)
                    cursor_x_offset = pre.get_width()
                except Exception:
                    pass

        # Text
        try:
            display = self.text if self.text else self.placeholder
            color = BLACK if self.text else (120, 120, 120)
            surf = render_text_pil(display, font_size=BUTTON_FONT_SIZE, color=color)
            text_rect = surf.get_rect(midleft=(self.rect.left + 16, self.rect.centery))
            screen.blit(surf, text_rect)
            if self.focused and self._cursor_visible:
                cx = text_rect.left + cursor_x_offset + 2
                cy = self.rect.top + 12
                ch = self.rect.height - 24
                pygame.draw.rect(screen, BLACK, (cx, cy, 3, ch))
        except Exception:
            pass

    @property
    def value(self) -> str:
        return self.text


# ===========================================================================
# Modal — dimmed overlay containing other widgets
# ===========================================================================
class Modal:
    """A blocking overlay. Draws a dim layer, then a panel, then child widgets."""

    def __init__(
        self,
        screen_size: Tuple[int, int],
        title: str,
        width_ratio: float = 0.7,
        height_ratio: float = 0.55,
        fixed_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            screen_size: (width, height) of the window the modal is drawn on.
            title: Display title rendered at the top of the modal.
            width_ratio / height_ratio: Fraction of screen_size used to
                size the modal when no fixed_size is given.
            fixed_size: Explicit (width, height) override. When set, the
                modal is exactly this size and centered in screen_size.
                Used by the death modal to auto-size to its content.
        """
        self.screen_w, self.screen_h = screen_size
        if fixed_size is not None:
            w, h = fixed_size
        else:
            w = int(self.screen_w * width_ratio)
            h = int(self.screen_h * height_ratio)
        x = (self.screen_w - w) // 2
        y = (self.screen_h - h) // 2
        self.rect = pygame.Rect(x, y, w, h)
        self.title = title
        self.children: List[object] = []  # anything with handle_event() and draw()
        self._surface = pygame.Surface(screen_size, pygame.SRCALPHA)

    def add(self, widget) -> None:
        self.children.append(widget)

    def handle_event(self, event: pygame.event.Event) -> None:
        for child in self.children:
            if hasattr(child, "handle_event"):
                child.handle_event(event)

    def update(self, dt_ms: int = 0) -> None:
        for child in self.children:
            if hasattr(child, "update"):
                child.update(dt_ms)

    def draw(self, screen: pygame.Surface) -> None:
        # Dim background
        self._surface.fill((0, 0, 0, 220))
        screen.blit(self._surface, (0, 0))

        # Modal panel
        panel = Panel(self.rect, fill=PANEL_FILL)
        panel.draw(screen)

        # Title
        try:
            surf = render_text_pil(self.title, font_size=TITLE_FONT_SIZE // 1.5, color=BLACK)
            text_rect = surf.get_rect(midtop=(self.rect.centerx, self.rect.top + 24))
            screen.blit(surf, text_rect)
        except Exception:
            pass

        # Children
        for child in self.children:
            if hasattr(child, "draw"):
                child.draw(screen)
