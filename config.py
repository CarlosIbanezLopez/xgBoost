"""Compatibility alias for the application configuration module."""

import sys

from app.core import config as _implementation

sys.modules[__name__] = _implementation
