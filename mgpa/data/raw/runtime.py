"""Keep optional scientific-library caches out of read-only installations."""
from pathlib import Path
import os
import tempfile

_runtime = None


def ensure_runtime():
    """Create isolated cache locations for optional EEG library imports."""
    global _runtime
    if _runtime is None:
        _runtime = Path(tempfile.mkdtemp(prefix="mgpa_acquisition_runtime_"))
    for variable, name in (("NUMBA_CACHE_DIR", "numba"), ("MPLCONFIGDIR", "matplotlib"),
                           ("_MNE_FAKE_HOME_DIR", "mne")):
        if variable not in os.environ:
            path = _runtime/name
            path.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(path)
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    return _runtime
