"""Exception hierarchy shared by every magform module.

Library code raises these (never bare ``assert``) so the CLI can map any
expected failure to exit code 2 with a clean message.
"""

from __future__ import annotations


class MagformError(Exception):
    """Base class for all expected magform errors."""


class InputError(MagformError, ValueError):
    """A non-physical or out-of-range input (negative pressure, T <= 0, ...)."""


class UnknownMaterialError(MagformError, KeyError):
    """A material key or alias that is not in the registry."""

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0]) if self.args else "unknown material"


class ConvergenceError(MagformError, RuntimeError):
    """A root finder or iteration failed to converge."""


__all__ = ["MagformError", "InputError", "UnknownMaterialError", "ConvergenceError"]
