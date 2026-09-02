"""Test-wide setup.

app/core/config.py reads GEMINI_API_KEY from the environment at import
time, so it must be set before any test module imports app.main /
app.services.* - conftest.py is loaded by pytest before test collection,
so this runs early enough.
"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test-key")
