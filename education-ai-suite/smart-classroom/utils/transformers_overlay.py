"""Side-loaded transformers builds for subprocesses needing a version other than the
pinned one: installed with ``--no-deps`` under the gitignored ``models/`` tree and
prepended to the subprocess PYTHONPATH. Must stay a subprocess -- an in-process
``sys.path`` prepend would swap the version for every other consumer.
"""
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# smart-classroom/ -- this file lives in smart-classroom/utils/
_SC_ROOT = Path(__file__).resolve().parents[1]


def overlay_dir(version: str) -> Path:
    """Directory holding a side-loaded transformers build.

    Lives under the gitignored ``models/`` tree. ``SC_EXPORT_DEPS_DIR`` overrides the
    parent so a setup script can provision overlays ahead of time (e.g. for an offline
    install).
    """
    override = os.environ.get("SC_EXPORT_DEPS_DIR")
    base = Path(override) if override else _SC_ROOT / "models" / ".export-deps"
    return base / f"transformers-{version}"


def _dist_name(spec: str) -> str:
    """The distribution name from a pip requirement string ('safetensors>=0.8.0')."""
    return re.split(r"[<>=!~\[;]", spec, maxsplit=1)[0].strip()


def _missing(overlay: Path, specs: Sequence[str]) -> list[str]:
    """Which of ``specs`` have no installed dist-info in the overlay.

    Checked per package rather than by a single transformers marker: a partial install
    (transformers lands, safetensors does not) would otherwise look complete forever,
    and the worker would fail every time with "safetensors>=0.8.0 is required" while
    provisioning kept short-circuiting.
    """
    missing = []
    for spec in specs:
        name = _dist_name(spec).replace("-", "_")
        # pip normalizes '-' to '_' in dist-info directory names.
        if not any(overlay.glob(f"{name}-*.dist-info")):
            missing.append(spec)
    return missing


def ensure_overlay(version: str, extra_specs: Sequence[str] = ()) -> Path:
    """Return the overlay dir, installing what is missing with --no-deps.

    ``extra_specs`` are pip requirement strings for packages whose installed version
    is below what this transformers build demands; they are installed into the same
    directory so they shadow the venv copy for the subprocess only.
    """
    overlay = overlay_dir(version)
    specs = [f"transformers=={version}", *extra_specs]

    missing = _missing(overlay, specs)
    if not missing:
        logger.info(f"Using transformers overlay at {overlay}")
        return overlay

    logger.info(f"Provisioning {', '.join(missing)} in {overlay}...")
    overlay.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps",
         "--target", str(overlay), *missing]
    )
    still_missing = _missing(overlay, specs)
    if completed.returncode != 0 or still_missing:
        joined = " ".join(f'"{s}"' for s in specs)
        raise RuntimeError(
            f"Could not provision {', '.join(still_missing or missing)}. "
            f"Install them manually and retry:\n"
            f'  python -m pip install --no-deps --target "{overlay}" {joined}'
        )
    return overlay


def subprocess_env(version: str, extra_specs: Sequence[str] = ()) -> dict:
    """A copy of os.environ whose PYTHONPATH resolves to the overlay build first.

    The overlay must come first: PYTHONPATH entries are searched in order, and the
    inherited sys.path entries carry the venv's own transformers.
    """
    overlay = ensure_overlay(version, extra_specs)
    search_path = [str(overlay)] + [p for p in sys.path if p]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(search_path))
    return env
