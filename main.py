"""Compatibility entry point.

The application lives in app/main.py; the canonical production target is
``app.main:app``. This module re-exports the exact same app object so an
existing ``uvicorn main:app`` deployment keeps working after a git pull.
"""

from app.main import app

__all__ = ["app"]
