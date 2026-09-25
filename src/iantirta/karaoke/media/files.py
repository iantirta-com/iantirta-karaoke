# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


__all__ = ["MediaFile"]


@@dataclass(frozen=True, slots=True)
class MediaFile:
    """
    Represents a locally available media file.

    Args:
        path: Path to the local media file.
        source: Original URL or source identifier.
        title: Optional media title.
        artist: Optional artist or uploader name.
        duration: Optional duration in seconds.

    Returns:
        A structured representation of a local media file.
    """

    path: Path
    source: str
    title: str | None = None
    artist: str | None = None
    duration: float | None = None
