"""Centralized logging setup.

Call `get_logger(__name__)` at the top of any module that needs to log.
The first call in a process configures the root logging format; later
calls just return a named logger using that same configuration. This
keeps every module's log lines consistent without each one repeating
`logging.basicConfig(...)`.

Note on the module name: this file is `src/utils/logging.py`, same
name as Python's standard library `logging` module. That's safe here
because Python 3 imports are absolute by default — `import logging`
below resolves to the stdlib module, not to this file, since this
file is only ever reached via its full path `src.utils.logging`, not
by adding `src/utils/` itself to sys.path.
"""

from __future__ import annotations

import logging

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring logging on first use."""
    setup_logging()
    return logging.getLogger(name)
