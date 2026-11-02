"""Content hashing for model artifacts.

A bundle that names ``sentence-transformers/all-MiniLM-L6-v2`` is not
reproducible: the same name can point at different weights over time, and a
missing extra silently changes what actually runs. The helpers here hash the
*files* that back a model so a registered bundle pins the exact weights it was
evaluated with, and hash the *definition* of an in-process algorithm so the
dependency-free backends are pinned too.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

#: Files larger than this are hashed by name and size only — a stray
#: multi-gigabyte blob in a cache directory should not stall a bundle build.
_MAX_HASHED_BYTES = 256 * 1024 * 1024


def builtin_artifact_hash(name: str, **params: object) -> str:
    """Hash the definition of an in-process algorithm.

    Built-in backends (the hashing embedder, the term-overlap reranker) have
    no files on disk; their identity *is* the algorithm plus its parameters.
    """
    parts = [name, *(f"{key}={params[key]}" for key in sorted(params))]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


@lru_cache(maxsize=32)
def hash_directory(path: Path) -> str:
    """Hash every file under *path* (relative names + contents).

    The result does not depend on the order the filesystem returns the files
    in, so the same weights always produce the same hash.
    """
    digest = hashlib.sha256()
    for file in sorted(p for p in Path(path).rglob("*") if p.is_file()):
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_digest(file).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_digest(file: Path) -> str:
    """Hash one file, degrading to its size when it is too large to read."""
    try:
        size = file.stat().st_size
    except OSError:
        return "unreadable"
    if size > _MAX_HASHED_BYTES:
        return f"large:{size}"
    digest = hashlib.sha256()
    try:
        with file.open("rb") as handle:
            while block := handle.read(1 << 20):
                digest.update(block)
    except OSError:
        return "unreadable"
    return digest.hexdigest()


def resolve_model_dir(model_name: str) -> Path | None:
    """Find the directory holding a model's files, if it is on this machine.

    Handles a plain local path and the HuggingFace cache layout
    (``models--<org>--<name>/snapshots/<revision>``).
    """
    direct = Path(model_name).expanduser()
    if direct.is_dir():
        return direct

    slug = "models--" + model_name.replace("/", "--")
    roots = [
        Path(os.environ["HF_HUB_CACHE"]) if "HF_HUB_CACHE" in os.environ else None,
        Path(os.environ["HF_HOME"]) / "hub" if "HF_HOME" in os.environ else None,
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for root in roots:
        if root is None:
            continue
        snapshots = root / slug / "snapshots"
        if not snapshots.is_dir():
            continue
        revisions = sorted(p for p in snapshots.iterdir() if p.is_dir())
        if revisions:
            return revisions[-1]
    return None


def model_artifact_hash(model_name: str) -> str:
    """Content hash of a model's files.

    Returns an empty string when the files cannot be found: an unpinned model
    is better than a fake hash, and the caller can report it.
    """
    directory = resolve_model_dir(model_name)
    if directory is None:
        return ""
    try:
        return hash_directory(directory.resolve())
    except OSError:  # pragma: no cover - defensive
        return ""


def combined_artifact_hash(*parts: str) -> str:
    """Combine several artifact hashes into one bundle-level hash.

    A bundle pins an embedder *and* a reranker; the single hash in the
    registry has to cover both.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update((part or "none").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
