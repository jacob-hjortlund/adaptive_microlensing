"""Repository-level pytest configuration."""

import importlib.util

# ``--doctest-modules`` imports every module under src/. The plotting module needs the
# optional ``plot`` extra, so it is skipped where matplotlib or seaborn is missing.
collect_ignore_glob: list[str] = []
if importlib.util.find_spec("matplotlib") is None or importlib.util.find_spec("seaborn") is None:
    collect_ignore_glob.append("src/adaptive_microlensing/plotting.py")
