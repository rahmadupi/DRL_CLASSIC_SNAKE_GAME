"""
Training TUI Launcher
=====================

Curses-based menu for configuring and launching PPO / DQN training runs.

6-row state machine (all keyboard-driven):

    [Algorithm]        PPO  <->  DQN
    [Existing Model]   <New Model>  /  saved_models/<file>.zip  ...
    [Output Prefix]    snake  →  preview: snake_<algo>_level<X>.zip
    [Level Select]     1  <->  5
    [Parallelization]  1  <->  CPU count   (forced to 1 for DQN)
    [Episode]          0   (target episode count; 0 = use timesteps budget)

    [Start Training]   (Enter on this row)

Navigation:
    UP/DOWN        move row
    LEFT/RIGHT        cycle option (Algorithm, Level, Parallelization)
    Enter      when on a list row - show modal picker
    Enter      on Output Prefix   - edit text inline
    Esc        back / cancel
    q          quit

Run:
    python train_launcher.py
"""

from __future__ import annotations

import curses
import glob
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure the project root is on sys.path regardless of CWD
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from game.train.utility import (
    auto_naming,
    detect_device,
    get_cpu_count,
    preview_name,
    SAVED_MODELS_DIR,
)
from game.model.configs import (
    obs_type_token,
    obs_type_from_token,
    OBS_TYPE_TO_TOKEN,
    TOKEN_TO_OBS_TYPE,
)


# ---------------------------------------------------------------------------
# Allowed obs types per algorithm (used by the modal picker + cycling)
# ---------------------------------------------------------------------------
# PPO and DQN share the same set of obs_type options (both algorithms
# support both 12-bit and spatiotemporal). "spatiotemporal_legacy" is
# offered in the modal picker only — it's a backward-compat hatch for
# loading old 4-channel models and shouldn't be the default for new
# runs.
_OBS_TYPE_CYCLE = {
    # Algorithm → ordered list of obs types used by ←/→ cycling.
    # The first entry is the algorithm's default obs_type.
    "ppo": ["spatiotemporal", "12bit"],
    "dqn": ["12bit", "spatiotemporal"],
}
_OBS_TYPE_PICKER_ITEMS = [
    "spatiotemporal (4x20x20, honest)",
    "12bit (12-dim vector)",
    "spatiotemporal_legacy (4x20x20, alias)",
]   # labels are kept short to fit the centered 50-char modal


# ---------------------------------------------------------------------------
# Visual constants — match the beige/cream launcher aesthetic of
# game_launcher.py (BG_CREAM in game/env/config.json).
# ---------------------------------------------------------------------------
BG_CREAM_RGB = (255, 247, 217)              # RGB for BG_CREAM
COLOR_CREAM_256 = 230                      # xterm-256color index nearest BG_CREAM
# Foreground: a deep warm brown instead of pure black.
# Pure black on cream is fine on dark terminals, but on a light
# Ghostty theme the cream area blends with the white terminal
# background and the whole launcher looks washed-out / hard to read.
# A slightly tinted, very dark color keeps high contrast against
# cream in BOTH dark and light terminal themes.
FG_DARK_RGB = (40, 30, 20)                 # RGB for FG_DARK — deep warm brown
COLOR_DARK_256 = 235                       # xterm-256color index nearest FG_DARK
PAIR_DEFAULT = 1                           # cream bg, dark fg — body text
PAIR_INVERSE = 2                           # dark bg, cream fg — inverse highlight
CONTAINER_WIDTH = 50                       # width (chars) of the centered options block


# ---------------------------------------------------------------------------
# Configuration state
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    algorithm: str = "PPO"                  # "PPO" | "DQN"
    obs_type: str = "spatiotemporal"        # "spatiotemporal" | "12bit"
                                            # | "spatiotemporal_legacy"
    existing_model: Optional[str] = None    # path to .zip or None (new)
    output_prefix: str = "snake"
    level: int = 1
    parallelization: int = 1
    episodes: int = 0                        # 0 → ignored, use total_timesteps

    def parallelization_capped(self, cpu_count: int) -> int:
        """
        Clamp the user's choice to safe bounds per algorithm.

        * DQN is off-policy (replay buffer) — too many parallel envs
          introduce off-policy bias. Capped at 4.
        * PPO is on-policy — every CPU core is fair game.
        """
        if self.algorithm == "DQN":
            return max(1, min(self.parallelization, 4))
        return max(1, min(self.parallelization, cpu_count))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _discover_models() -> List[str]:
    """Return sorted list of *.zip files in saved_models/."""
    if not SAVED_MODELS_DIR.exists():
        return []
    return sorted(str(p) for p in SAVED_MODELS_DIR.glob("*.zip"))


def _short_filename(path: Optional[str]) -> str:
    if path is None:
        return "<New Model>"
    return os.path.basename(path)


def _obs_type_display(obs_type: str) -> str:
    """Render ``obs_type`` for the TUI row.

    Maps the long obs_type names to compact labels that fit in the
    centered 50-char value column. The token shown in parentheses is
    the same one that gets embedded in the saved filename — making
    the mapping explicit saves the user from guessing what
    "spatiotemporal" turns into inside ``saved_models/``.
    """
    labels = {
        "spatiotemporal":        "spatiotemporal (sptmp)",
        "12bit":                 "12bit (12bit)",
        "spatiotemporal_legacy": "spatiotemporal_legacy (sptmp_lgcy)",
    }
    return labels.get(obs_type, obs_type)


# Matches the convention enforced by `game.train.utility.auto_naming`:
#     <prefix>_<algo>[_<obstype>]_level<L>[_<n>].zip
# e.g. snake_ppo_sptmp_level1.zip, myrun_dqn_12bit_level3_2.zip
# The obs_type token is OPTIONAL — pre-existing models saved before the
# naming change still match the legacy pattern (no obs token).
# Used to filter the model picker by algorithm and to detect when a
# user has selected a model whose algorithm no longer matches the
# currently chosen algorithm (a PPO zip selected under DQN, etc.).
_ALGO_FROM_NAME = re.compile(
    r"_(?P<algo>ppo|dqn)"
    r"(?:_(?P<obstype>12bit|sptmp_lgcy|sptmp))?"
    r"_level\d+",
    re.IGNORECASE,
)

# Extracts only the obs_type token (or None) from a model filename.
# Mirrors the optional group in ``_ALGO_FROM_NAME`` so the picker
# can filter on obs_type too (e.g. don't offer a sptmp PPO model when
# the user has selected the 12bit variant).
_OBS_FROM_NAME = re.compile(
    r"_(?:ppo|dqn)_(?P<obstype>12bit|sptmp_lgcy|sptmp)(?:_level\d+|$)",
    re.IGNORECASE,
)


def _algo_from_filename(path: str) -> Optional[str]:
    """
    Extract the algorithm token from a saved-model filename.

    Returns ``"PPO"`` / ``"DQN"`` (capitalised to match
    :attr:`TrainConfig.algorithm`) or ``None`` if the filename doesn't
    follow the project's naming convention. A ``None`` result is treated
    as "unknown / not safe to resume" and the file is excluded from the
    picker.
    """
    name = os.path.basename(path)
    m = _ALGO_FROM_NAME.search(name)
    if not m:
        return None
    return m.group("algo").upper()


def _obs_type_from_filename(path: str) -> Optional[str]:
    """
    Extract the obs_type token from a saved-model filename.

    Returns the long obs_type name (e.g. ``"spatiotemporal"``,
    ``"12bit"``, ``"spatiotemporal_legacy"``) or ``None`` if the
    filename doesn't embed one (legacy naming convention). ``None``
    is treated as "unknown / backward-compat" — the loader falls
    back to the SB3 ``observation_space.shape`` lookup in
    :mod:`game.env.input_controller` when this returns ``None``.
    """
    name = os.path.basename(path)
    m = _OBS_FROM_NAME.search(name)
    if not m:
        return None
    return obs_type_from_token(m.group("obstype"))


def _filter_models_for_algo(
    model_paths: List[str], algorithm: str
) -> List[str]:
    """
    Keep only models whose filename advertises ``algorithm``.

    Files that don't match the naming convention are dropped silently —
    they cannot be safely resumed into either algorithm because the
    features extractor / obs type can't be inferred.
    """
    target = algorithm.upper()
    return [p for p in model_paths if _algo_from_filename(p) == target]


def _filter_models_for_obs_type(
    model_paths: List[str], obs_type: str
) -> List[str]:
    """
    Keep only models whose filename advertises ``obs_type``.

    Files with no obs_type token in the filename (legacy naming) are
    silently dropped — the picker filters by EXACT obs type, so an
    ambiguous file shouldn't appear in the list and risk an incorrect
    resume. The launcher's SB3-metadata fallback in
    :meth:`game.env.input_controller.load_model` still works for
    legacy files when the user manually selects them through other
    means.
    """
    return [p for p in model_paths if _obs_type_from_filename(p) == obs_type]


def _drop_mismatched_model(cfg: TrainConfig) -> None:
    """
    Clear :attr:`TrainConfig.existing_model` if it doesn't match the
    currently selected algorithm OR obs_type. Call this after every
    algorithm / obs_type change so the displayed selection stays
    valid (avoids the silent state where a PPO sptmp zip is still
    "selected" while the user is configuring a DQN 12-bit run).
    """
    if cfg.existing_model is None:
        return
    if _algo_from_filename(cfg.existing_model) != cfg.algorithm:
        cfg.existing_model = None
        return
    # Only enforce obs_type match when the filename actually advertises
    # one — legacy files (no obs token) are passed through and the
    # loader's SB3-metadata fallback decides.
    file_obs = _obs_type_from_filename(cfg.existing_model)
    if file_obs is not None and file_obs != cfg.obs_type:
        cfg.existing_model = None


# ---------------------------------------------------------------------------
# Color & layout helpers
# ---------------------------------------------------------------------------
def _init_colors() -> None:
    """
    Initialize the curses color palette so the launcher matches the
    beige/cream aesthetic of game_launcher.py.
    Exact RGB is used when the terminal supports it (most Linux terminals);
    otherwise we fall back to the xterm-256color indices 230 (cream)
    and 235 (dark gray) which are visually close to BG_CREAM /
    FG_DARK. As a last resort we drop to the ANSI basic palette.
    """
    curses.start_color()
    try:
        if curses.can_change_color():
            # curses takes RGB components in the 0-1000 range, not 0-255.
            r, g, b = BG_CREAM_RGB
            # Color slot 80 — well outside the 0-7 reserved range.
            curses.init_color(
                80, r * 1000 // 255, g * 1000 // 255, b * 1000 // 255
            )
            cream_idx = 80

            fr, fg_, fb = FG_DARK_RGB
            # Color slot 81 — paired with slot 80 above.
            curses.init_color(
                81, fr * 1000 // 255, fg_ * 1000 // 255, fb * 1000 // 255
            )
            dark_idx = 81
        else:
            cream_idx = COLOR_CREAM_256
            dark_idx = COLOR_DARK_256
    except curses.error:
        cream_idx = curses.COLOR_WHITE
        dark_idx = curses.COLOR_BLACK

    curses.init_pair(PAIR_DEFAULT, dark_idx, cream_idx)
    curses.init_pair(PAIR_INVERSE, cream_idx, dark_idx)


def _container_left_x(w: int) -> int:
    """X coordinate of the left edge of the centered options container."""
    return max(0, (w - CONTAINER_WIDTH) // 2)


# ---------------------------------------------------------------------------
# Curses UI primitives
# ---------------------------------------------------------------------------
def _safe_addstr(win, y: int, x: int, text: str, attr: int = 0) -> None:
    """addstr() that swallows "would extend past screen" errors."""
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def _draw_centered_title(stdscr, title: str, subtitle: str = "") -> None:
    h, w = stdscr.getmaxyx()
    _safe_addstr(stdscr, 1, max(0, (w - len(title)) // 2), title, curses.A_BOLD)
    if subtitle:
        _safe_addstr(stdscr, 2, max(0, (w - len(subtitle)) // 2), subtitle, curses.A_DIM)


def _draw_footer(stdscr, hint: str) -> None:
    h, w = stdscr.getmaxyx()
    _safe_addstr(stdscr, h - 2, max(0, (w - len(hint)) // 2), hint, curses.A_DIM)


# ---------------------------------------------------------------------------
# Modal pickers
# ---------------------------------------------------------------------------
def _modal_pick(stdscr, title: str, items: List[str], initial: int = 0) -> Optional[int]:
    """
    Show a centered list of items; arrow keys move, Enter selects, Esc cancels.
    Returns the selected index, or None on cancel.
    """
    h, w = stdscr.getmaxyx()
    modal_w = min(w - 8, max(40, max(len(s) for s in items) + 6))
    modal_h = min(h - 6, len(items) + 4)
    modal_y = (h - modal_h) // 2
    modal_x = (w - modal_w) // 2

    win = curses.newwin(modal_h, modal_w, modal_y, modal_x)
    win.keypad(True)
    # Modal inherits the beige/cream background so the popup matches the
    # launcher aesthetic instead of standing out as a black box.
    win.bkgd(' ', curses.color_pair(PAIR_DEFAULT))
    win.attrset(curses.color_pair(PAIR_DEFAULT))

    idx = max(0, min(initial, len(items) - 1))

    while True:
        win.clear()
        win.border()
        _safe_addstr(win, 0, 2, f" {title} ", curses.A_BOLD)

        inner_w = modal_w - 4
        for i, item in enumerate(items):
            display = item if len(item) <= inner_w else item[: inner_w - 1] + "…"
            attr = curses.A_REVERSE if i == idx else curses.A_NORMAL
            _safe_addstr(win, 2 + i, 2, display.ljust(inner_w), attr)

        _safe_addstr(win, modal_h - 1, 2, " ↑/↓ move · Enter select · Esc cancel ", curses.A_DIM)
        win.refresh()

        ch = win.getch()
        if ch == curses.KEY_UP:
            idx = (idx - 1) % len(items)
        elif ch == curses.KEY_DOWN:
            idx = (idx + 1) % len(items)
        elif ch in (curses.KEY_ENTER, 10, 13, ord(" ")):
            return idx
        elif ch == 27:   # Esc
            return None
        elif ch in (ord("q"), ord("Q")):
            return None


def _modal_text_input(stdscr, title: str, initial: str = "") -> Optional[str]:
    """Inline text editor with backspace + Enter (Esc cancels)."""
    h, w = stdscr.getmaxyx()
    modal_w = min(w - 8, 50)
    modal_h = 5
    modal_y = (h - modal_h) // 2
    modal_x = (w - modal_w) // 2

    win = curses.newwin(modal_h, modal_w, modal_y, modal_x)
    win.keypad(True)
    # Modal inherits the beige/cream background so the popup matches the
    # launcher aesthetic instead of standing out as a black box.
    win.bkgd(' ', curses.color_pair(PAIR_DEFAULT))
    win.attrset(curses.color_pair(PAIR_DEFAULT))
    buf = list(initial)

    while True:
        win.clear()
        win.border()
        _safe_addstr(win, 0, 2, f" {title} ", curses.A_BOLD)
        text = "".join(buf)
        _safe_addstr(win, 2, 2, text.ljust(modal_w - 4))
        win.move(2, 2 + len(text))
        _safe_addstr(win, modal_h - 1, 2, " Enter=save · Esc=cancel · Backspace=delete ", curses.A_DIM)
        win.refresh()

        ch = win.getch()
        if ch == 27:    # Esc
            return None
        elif ch in (curses.KEY_ENTER, 10, 13):
            return "".join(buf)
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            if buf:
                buf.pop()
        elif 32 <= ch <= 126:   # printable ASCII
            if len(buf) < modal_w - 6:
                buf.append(chr(ch))


# ---------------------------------------------------------------------------
# Main TUI loop
# ---------------------------------------------------------------------------
# Row order MUST match the indices below.
ROWS: List[str] = [
    "algorithm",
    "obs_type",
    "existing_model",
    "output_prefix",
    "level",
    "parallelization",
    "episodes",
    "start",
]


def _render(stdscr, cfg: TrainConfig, focus: int) -> None:
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    _draw_centered_title(
        stdscr,
        "TRAIN SNAKE AGENT",
        f"device={detect_device()}   cpus={get_cpu_count()}",
    )

    # ---- Left column: option labels --------------------------------------
    labels = [
        ("Algorithm",       f"{cfg.algorithm}"),
        ("Obs Type",        _obs_type_display(cfg.obs_type)),
        ("Existing Model",  _short_filename(cfg.existing_model)),
        ("Output Prefix",   cfg.output_prefix or "<empty>"),
        ("Level Select",    f"{cfg.level}"),
        ("Parallelization", f"{cfg.parallelization} {'Demo mode' if cfg.parallelization == 1 else 'Headless mode'}"),
        ("Episode",         f"{cfg.episodes}"),
        ("Start Training",  ""),
    ]

    # Preview row (under the form, computed on-the-fly). Includes the
    # obs_type token so the user can see the final filename shape
    # (e.g. "snake_ppo_12bit_level1.zip").
    preview_stem, preview_path = preview_name(
        prefix=cfg.output_prefix or "model",
        algo=cfg.algorithm.lower(),
        level=cfg.level,
        obs_type=cfg.obs_type,
    )
    preview_line = f"Preview → {preview_stem}.zip"

    # Container is centered horizontally; rows are left-aligned inside it.
    cx = _container_left_x(w)

    # Tight layout: 1 line per row, matching game_launcher.py's flush
    # button stack (gap=0) so vertical spacing between options is small.
    line_h = 1
    row_top = 4
    for i, (label, value) in enumerate(labels):
        y = row_top + i * line_h
        if y >= h - 4:
            break  # don't overwrite the preview/footer rows
        is_focus = (i == focus)
        attr_l = curses.A_BOLD | (curses.A_REVERSE if is_focus else 0)
        attr_v = (curses.A_REVERSE if is_focus else 0)
        prefix = "▶ " if is_focus else "  "
        _safe_addstr(stdscr, y, cx, f"{prefix}[{label}]", attr_l)
        _safe_addstr(stdscr, y, cx + 20, value, attr_v)

    # Preview — left-aligned within the centered container.
    _safe_addstr(stdscr, h - 4, cx, preview_line, curses.A_DIM)
    _safe_addstr(stdscr, h - 3, cx + 5, preview_path, curses.A_DIM)

    _draw_footer(
        stdscr,
        "↑/↓ row · ←/→ change · Enter select/edit · q quit",
    )
    stdscr.refresh()


def _activate_row(
    stdscr, cfg: TrainConfig, focus: int, model_paths: List[str]
) -> bool:
    """
    Handle Enter on the focused row.

    Returns True if the user pressed Start (training should launch);
    False otherwise.
    """
    row = ROWS[focus]

    if row == "algorithm":
        cfg.algorithm = "DQN" if cfg.algorithm == "PPO" else "PPO"
        # Re-clamp to the new algorithm's allowed range
        cfg.parallelization = cfg.parallelization_capped(get_cpu_count())
        # A previously selected model may no longer be compatible with
        # the new algorithm; clear it so the form stays consistent.
        _drop_mismatched_model(cfg)
        return False

    if row == "obs_type":
        # Modal picker offers the full set of supported obs types;
        # the cycling handled by ←/→ covers the two "primary" options
        # (12bit / spatiotemporal) which is what the user changes most
        # often. ``spatiotemporal_legacy`` is a backward-compat hatch
        # for loading 4-channel models — best surfaced via the modal.
        options = ["spatiotemporal", "12bit", "spatiotemporal_legacy"]
        labels = _OBS_TYPE_PICKER_ITEMS
        # Pre-select the currently active option so the user can
        # confirm with Enter without re-navigating.
        try:
            initial = options.index(cfg.obs_type)
        except ValueError:
            initial = 0
        chosen = _modal_pick(stdscr, "Select Obs Type", labels, initial=initial)
        if chosen is None:
            return False
        cfg.obs_type = options[chosen]
        # An existing model selected earlier may no longer match the
        # new obs_type; clear it so the form stays consistent.
        _drop_mismatched_model(cfg)
        return False

    if row == "existing_model":
        # Only show models whose filename matches BOTH the active
        # algorithm AND the active obs_type. Resuming a PPO sptmp
        # zip under DQN 12-bit would crash inside SB3's loader with
        # an opaque features-extractor error, and resuming across
        # obs_types (same algo) would silently feed the wrong shape
        # into ``model.predict()``.
        compatible = _filter_models_for_obs_type(
            _filter_models_for_algo(model_paths, cfg.algorithm),
            cfg.obs_type,
        )
        items = ["<New Model>"] + [_short_filename(p) for p in compatible]
        title = f"Select {cfg.algorithm} {cfg.obs_type} Model"
        if len(compatible) == 0:
            items.append(f"<no saved {cfg.algorithm} {cfg.obs_type} models>")
        chosen = _modal_pick(stdscr, title, items, initial=0)
        if chosen is None:
            return False
        # The trailing "<no saved ...>" sentinel is a non-actionable
        # hint; treat it like a cancel.
        if chosen >= len(compatible) + 1:
            return False
        cfg.existing_model = None if chosen == 0 else compatible[chosen - 1]
        return False

    if row == "output_prefix":
        new_prefix = _modal_text_input(stdscr, "Output Prefix", cfg.output_prefix)
        if new_prefix is not None:
            cfg.output_prefix = new_prefix.strip() or "snake"
        return False

    if row == "level":
        items = [f"Level {i}" for i in range(1, 6)]
        chosen = _modal_pick(stdscr, "Select Level", items, initial=cfg.level - 1)
        if chosen is not None:
            cfg.level = chosen + 1
        return False

    if row == "parallelization":
        cpu = get_cpu_count()
        if cfg.algorithm == "DQN":
            # Cap DQN at 4 (off-policy replay buffer stays well-mixed)
            items = [str(i) for i in range(1, 5)]
            title = "Parallelization (DQN capped at 4)"
        else:
            items = [str(i) for i in range(1, cpu + 1)]
            title = "Select Parallelization"
        chosen = _modal_pick(stdscr, title, items, initial=cfg.parallelization - 1)
        if chosen is not None:
            cfg.parallelization = int(items[chosen].split()[0])
        return False

    if row == "episodes":
        new_value = _modal_text_input(stdscr, "Episodes (0 = ignore)", str(cfg.episodes))
        if new_value is not None:
            try:
                cfg.episodes = max(0, int(new_value.strip() or "0"))
            except ValueError:
                cfg.episodes = 0
        return False

    if row == "start":
        return True

    return False


def _adjust_value(cfg: TrainConfig, focus: int, delta: int) -> None:
    """←/→ handler — change the value of the focused row."""
    row = ROWS[focus]
    cpu = get_cpu_count()

    if row == "algorithm":
        cfg.algorithm = "DQN" if cfg.algorithm == "PPO" else "PPO"
        # Re-clamp parallelization to the new algorithm's allowed range
        # (DQN max=4; PPO max=cpu_count). Keeps the displayed value valid.
        cfg.parallelization = cfg.parallelization_capped(get_cpu_count())
        # Same compat-hygiene as the Enter handler: drop the selection
        # if its filename no longer matches the new algorithm.
        _drop_mismatched_model(cfg)
    elif row == "obs_type":
        # Cycle through the algorithm's primary obs_type options.
        # "spatiotemporal_legacy" is intentionally NOT in the cycle —
        # it's a backward-compat hatch best picked from the modal
        # picker (Enter on the obs_type row) so the user picks it
        # deliberately instead of bumping into it on every ←/→ press.
        cycle = _OBS_TYPE_CYCLE.get(cfg.algorithm, ["spatiotemporal", "12bit"])
        try:
            idx = cycle.index(cfg.obs_type)
        except ValueError:
            idx = 0
        cfg.obs_type = cycle[(idx + delta) % len(cycle)]
        _drop_mismatched_model(cfg)
    elif row == "level":
        cfg.level = max(1, min(5, cfg.level + delta))
    elif row == "parallelization":
        if cfg.algorithm == "DQN":
            cfg.parallelization = max(1, min(4, cfg.parallelization + delta))
        else:
            cfg.parallelization = max(1, min(cpu, cfg.parallelization + delta))
    elif row == "episodes":
        cfg.episodes = max(0, cfg.episodes + delta)


# ---------------------------------------------------------------------------
# Launch training (called when user presses Enter on the Start row)
# ---------------------------------------------------------------------------
def _launch_training(stdscr, cfg: TrainConfig) -> None:
    """
    Defer to the trainer modules. Show a status line at the bottom of the
    TUI while training runs, so the user has feedback.
    """
    curses.def_prog_mode()    # save the current (TUI) screen state
    curses.endwin()           # hand control back to the terminal

    try:
        # Algorithm-name validation — fallback to the main menu if the
        # selected model's filename doesn't contain the algorithm name
        # or doesn't match the currently selected algorithm.
        #
        # The picker already filters by algorithm (see
        # `_filter_models_for_algo`), but a model could still slip
        # through with a mislabeled filename (e.g. a PPO zip renamed
        # to `*_dqn_level*.zip`) or no algorithm token at all. In that
        # case SB3's `PPO.load` / `DQN.load` would fail deep inside
        # the zip unpickler with an opaque features-extractor error.
        # We catch it here, print a clear message, and let the
        # `finally` block below restore the TUI (i.e. "fallback to
        # the main menu UI").
        if cfg.existing_model is not None:
            file_algo = _algo_from_filename(cfg.existing_model)
            if file_algo != cfg.algorithm:
                detected = file_algo if file_algo else "<no algorithm token found>"
                print(
                    f"\n[ERROR] Selected model's filename does not match "
                    f"the '{cfg.algorithm}' algorithm.\n"
                    f"         File:     {cfg.existing_model}\n"
                    f"         Detected: {detected}\n"
                    f"         Falling back to the main menu."
                )
                return

            # Obs-type validation — the picker already filters by
            # obs_type, but a renamed legacy file (no obs token) could
            # still slip through. ``_drop_mismatched_model`` only
            # enforces the match when the filename DOES advertise one,
            # so this catches files that DO but mismatch.
            file_obs = _obs_type_from_filename(cfg.existing_model)
            if file_obs is not None and file_obs != cfg.obs_type:
                print(
                    f"\n[ERROR] Selected model's obs_type does not match "
                    f"the selected obs_type.\n"
                    f"         File:         {cfg.existing_model}\n"
                    f"         File obs:     {file_obs}\n"
                    f"         Selected obs: {cfg.obs_type}\n"
                    f"         Falling back to the main menu."
                )
                return

        # Concern #4 — stale-pick guard.
        # The picker snapshot is re-scanned every ~2 s, but a file can
        # still be deleted/moved between the last redraw and the moment
        # the user presses Start. Re-validate here so we fail with a
        # clear message instead of a raw `FileNotFoundError` deep
        # inside SB3's zip loader.
        if cfg.existing_model is not None and not Path(cfg.existing_model).is_file():
            print(
                f"\n[ERROR] Selected model file is missing:\n"
                f"         {cfg.existing_model}\n"
                f"         It may have been moved or deleted. "
                f"Re-open the launcher and pick another model "
                f"(or choose <New Model>)."
            )
            return

        # Translate "episodes" → "total_timesteps".
        #
        # Snake episodes are bounded by ``game_environment.max_steps=1000``
        # and typically last 100-500 steps depending on agent skill. The
        # previous 2000 multiplier was an over-generous upper-bound that
        # made the tqdm bar's "total" misleading (e.g. entering 10
        # episodes would show ``0/20000`` instead of something intuitive).
        #
        # 5000 steps/episode is a middle estimate for early-to-mid training:
        # * Random agent:  ~50-150 steps/episode
        # * Beginner agent: ~150-300 steps/episode
        # * Skilled agent:  ~300-1000 steps/episode
        #
        # The floor at ``MIN_TIMESTEPS`` guarantees enough steps for at
        # least one PPO rollout (``n_steps=2048``) regardless of the
        # user's episode count.
        STEPS_PER_EPISODE = 2048
        MIN_TIMESTEPS = 2048  # >= PPO n_steps; enough for one rollout

        if cfg.episodes > 0:
            total_timesteps = max(MIN_TIMESTEPS, cfg.episodes * STEPS_PER_EPISODE)
        else:
            total_timesteps = 200_000  # default when user leaves Episode=0

        # The user's episode input is the source of truth for the
        # progress bar's "eps/N" display — propagate it down.
        total_episodes = cfg.episodes if cfg.episodes > 0 else None

        # Common trainer kwargs — pulled out so the PPO / DQN branches
        # below stay readable. Both trainers now load their default
        # hyperparameters from the JSON config and accept these kwargs
        # as overrides on top.
        common_kwargs = dict(
            level=cfg.level,
            obs_type=cfg.obs_type,
            n_envs=cfg.parallelization_capped(get_cpu_count()),
            total_timesteps=total_timesteps,
            total_episodes=total_episodes,
            output_prefix=cfg.output_prefix or "snake",
            load_path=cfg.existing_model,
        )

        if cfg.algorithm == "PPO":
            from game.train.ppo_trainer import PPOTrainingConfig, train_ppo

            config = PPOTrainingConfig.from_json_dict(
                "ppo",
                **common_kwargs,
            )
            model, path = train_ppo(config)
        else:
            from game.train.dqn_trainer import DQNTrainingConfig, train_dqn

            # When resuming, skip the warm-up exploration phase so the
            # replay buffer starts sampling from the loaded model's
            # recent experience instead of random exploration.
            config = DQNTrainingConfig.from_json_dict(
                "dqn",
                **common_kwargs,
                learning_starts=0 if cfg.existing_model else 1_000,
            )
            model, path = train_dqn(config)

        print(f"\n[OK] Training complete. Saved → {path}")

    except Exception as exc:
        import traceback
        print(f"\n[ERROR] Training failed: {exc!r}")
        traceback.print_exc()
    finally:
        print("\nPress Enter to return to the launcher...")
        try:
            input()
        except EOFError:
            time.sleep(2)

    # Restore the TUI
    curses.reset_prog_mode()
    stdscr.refresh()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _main(stdscr) -> None:
    curses.curs_set(0)
    _init_colors()
    # Paint the whole window beige/cream so erase() fills every cell with
    # the launcher background color (matches game_launcher.py).
    stdscr.bkgd(' ', curses.color_pair(PAIR_DEFAULT))
    stdscr.nodelay(False)
    stdscr.keypad(True)

    cfg = TrainConfig()
    model_paths = _discover_models()
    focus = 0
    last_rescan = time.time()

    while True:
        # Refresh the model list every 2s so newly-trained .zips show up.
        if time.time() - last_rescan > 2.0:
            model_paths = _discover_models()
            last_rescan = time.time()

        _render(stdscr, cfg, focus)
        ch = stdscr.getch()

        if ch in (ord("q"), ord("Q")):
            return
        elif ch == curses.KEY_UP:
            focus = (focus - 1) % len(ROWS)
        elif ch == curses.KEY_DOWN:
            focus = (focus + 1) % len(ROWS)
        elif ch == curses.KEY_LEFT:
            _adjust_value(cfg, focus, -1)
        elif ch == curses.KEY_RIGHT:
            _adjust_value(cfg, focus, +1)
        elif ch in (curses.KEY_ENTER, 10, 13, ord(" ")):
            if _activate_row(stdscr, cfg, focus, model_paths):
                _launch_training(stdscr, cfg)
        elif ch == 27:  # Esc — for now just go to top row
            focus = 0


if __name__ == "__main__":
    try:
        curses.wrapper(_main)
    except KeyboardInterrupt:
        sys.exit(0)