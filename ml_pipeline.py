"""Compatibility entry point for the machine-learning pipeline."""

import sys

from app.ml import pipeline as _implementation


if __name__ == "__main__":
    _implementation.run_training()
else:
    sys.modules[__name__] = _implementation
