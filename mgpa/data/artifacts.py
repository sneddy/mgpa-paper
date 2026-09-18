"""Small hash-bound artifact primitives for standalone experiment runners."""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile


def digest(value):
    """Hash a JSON-serializable value using a deterministic encoding."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    """Stream a file into SHA256 without loading it into memory."""
    out = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            out.update(part)
    return out.hexdigest()


def read_json(path):
    """Read a plain JSON artifact without executing code."""
    return json.loads(Path(path).read_text())


def write_json(path, value, *, replace=False):
    """Atomically create JSON and refuse changed existing scientific results."""
    path = Path(path)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"Refusing to replace scientific artifact: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".writing-", delete=False) as stream:
        tmp = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if replace:
            tmp.replace(path)
        else:
            os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def safe_id(value):
    """Validate a short path-safe experiment identifier."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,150}", value):
        raise ValueError("Use a filename-safe nonempty run identifier")
    return value


@contextmanager
def run_guard(directory):
    """Hold an OS-released exclusive lock for one experiment run."""
    import fcntl
    path = Path(directory)/"process.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this experiment run") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def runtime(directory, threads=2):
    """Fix CPU threading and record numerical dependency versions."""
    directory = Path(directory)
    for key, leaf in (("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", "xdg"), ("TORCH_HOME", "torch")):
        folder = directory/leaf
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    import torch
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=threads)
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    return dict(python=sys.version, platform=platform.platform(), threads=threads, device="cpu",
        packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "torch")})
