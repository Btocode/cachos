"""Shared helpers: model cache directory and resumable-ish downloads."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path


def cache_dir(*parts: str) -> Path:
    """``$CACHOS_CACHE`` or ``~/.cache/cachos`` (created on demand)."""
    d = Path(os.environ.get("CACHOS_CACHE", Path.home() / ".cache" / "cachos")).joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(url: str, dest: Path, *, quiet: bool = False) -> Path:
    """Download ``url`` to ``dest`` atomically unless it already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if not quiet:
        print(f"cachos: downloading {url.rsplit('/', 1)[-1]} -> {dest}", file=sys.stderr)
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URLs
    tmp.replace(dest)
    return dest
