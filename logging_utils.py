"""logging_utils.py

Shared logging setup for the BSNIP pipeline — the default logging standard
for every script in this repo. Fixes/adds two things project-wide:

1. Routes logging to stdout, not stderr. `logging.basicConfig()` defaults
   to a StreamHandler on stderr, so under SLURM (`--output=...out
   --error=...err`) every informational log line — epoch metrics, "saved
   to", split sizes, etc. — was landing in the .err file even though
   nothing had gone wrong. setup_logging() attaches an explicit
   `logging.StreamHandler(sys.stdout)` instead, so .out carries the run's
   narrative and .err stays reserved for genuine problems (uncaught
   tracebacks still go to stderr via Python's default excepthook, which
   this doesn't touch).

2. Colors every log line by level and, for INFO records, by content:
   epoch/metric lines, "saved"/"created" file-output lines, "New best
   checkpoint" lines, and success/completion lines each get a distinct
   color, so a long training log scans at a glance.

Colors are emitted unconditionally — not gated on sys.stdout.isatty() —
so the same ANSI-colored output renders correctly live in a terminal AND
when redirected to a SLURM .out file and viewed later with `less -R`,
`tail -f`, or `cat`; all three pass ANSI SGR codes through untouched. Only
basic 16-color codes are used (no 256-color extended codes) for the widest
pager/terminal compatibility.

Usage (replaces `logging.basicConfig(level=..., format=...)` everywhere):
    from logging_utils import setup_logging
    setup_logging(args.log_level)
"""

from __future__ import annotations

import logging
import re
import sys

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

_RESET = "\033[0m"
_BOLD_RED = "\033[1;31m"       # ERROR / CRITICAL
_BOLD_YELLOW = "\033[1;33m"    # WARNING
_BOLD_WHITE = "\033[1;37m"     # INFO (general)
_BRIGHT_CYAN = "\033[96m"      # epoch summaries / training metrics
_BOLD_MAGENTA = "\033[1;35m"   # "New best checkpoint"
_YELLOW = "\033[33m"           # output/file creation ("saved to...", "created ...")
_BOLD_GREEN = "\033[1;32m"     # success / final completion messages

# Content-based color rules for INFO/DEBUG records, checked in order —
# first match wins. Level-based rules (WARNING/ERROR/CRITICAL) always take
# priority over these and are applied separately, before these ever run.
_CONTENT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"best checkpoint", re.IGNORECASE), _BOLD_MAGENTA),
    (re.compile(r"\b(complete|completed|done|success)\b", re.IGNORECASE), _BOLD_GREEN),
    (re.compile(r"^Epoch \[|train_loss=|val_loss=|^Test set @|Optimal threshold"), _BRIGHT_CYAN),
    (re.compile(r"\b(saved|save|created|creating)\b", re.IGNORECASE), _YELLOW),
]


class ColoredFormatter(logging.Formatter):
    """logging.Formatter that ANSI-colors each record by level, then content.

    ERROR/CRITICAL -> bold red, WARNING -> bold yellow — level always wins.
    For INFO/DEBUG, _CONTENT_RULES picks a more specific color (best
    checkpoint / success / metrics / file-output); anything else falls
    back to bold white.
    """

    def __init__(self, fmt: str = DEFAULT_LOG_FORMAT) -> None:
        super().__init__(fmt)

    def _color_for(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.ERROR:
            return _BOLD_RED
        if record.levelno == logging.WARNING:
            return _BOLD_YELLOW
        message = record.getMessage()
        for pattern, color in _CONTENT_RULES:
            if pattern.search(message):
                return color
        return _BOLD_WHITE

    def format(self, record: logging.LogRecord) -> str:
        plain = super().format(record)
        color = self._color_for(record)
        # Wrap the *whole* formatted record — embedded newlines included,
        # e.g. a multi-line classification_report — in one color/reset
        # pair, so color state can't bleed into unrelated output that
        # follows (a traceback, a plain print()) and pagers/`tail -f`
        # don't see a dangling open color code mid-stream.
        return f"{color}{plain}{_RESET}"


def setup_logging(level: str = "INFO", fmt: str = DEFAULT_LOG_FORMAT) -> None:
    """Configure root logging: colored output on a single stdout StreamHandler.

    Drop-in replacement for `logging.basicConfig(level=..., format=...)` —
    call once near the top of a script's main(). Clears any existing root
    handlers first so repeated calls (e.g. in tests) don't stack duplicate
    handlers and double-print every line.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(fmt))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level))
    root.handlers.clear()
    root.addHandler(handler)
