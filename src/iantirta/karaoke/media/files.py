# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["MediaFile"]


@dataclass(frozen=True, slots=True)
class MediaFile:
    """
    Represents a locally available media file.

    Args:
        path: Path to the primary media file.
        source: Original source URL or identifier.
        audio_path: Optional path to an extracted audio file.
        title: Optional media title.
        artist: Optional artist, uploader, or channel name.
        duration: Optional duration in seconds.

    Returns:
        A structured representation of a local media file.
    """

    path: Path
    source: str
    title: str | None = None
    artist: str | None = None
    duration: float | None = None

    @property
    def audio_path(self) -> Path:
        if self.path.suffix == "wav":
            return self.path

        audio_path = self.path.with_suffix(".wav")

        if audio_path.is_file():
            return audio_path

        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.path),
            "-vn",
            "-f", "wav",
            str(audio_path),
        ]
        
        try:            
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to extract audio from {self.path}\n"
                f"Command: {cmd}\n"
                f"Error:\n{exc.stderr}"
            ) from exc

        return audio_path
