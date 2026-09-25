# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

import json
import subprocess
import typing as t
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "AudioData",
    "AudioFile",
    "get_audio_info",
    "load_audio",
]


@dataclass(frozen=True, slots=True)
class AudioFile:
    """
    A reference to an audio file and its basic audio properties.

    Attributes:
        path: Path to the audio file.
        sample_rate: Sample rate in Hz, if known.
        channels: Number of audio channels, if known.
        duration: Duration in seconds, if known.
    """

    path: Path
    sample_rate: int | None = None
    channels: int | None = None
    duration: float | None = None


@dataclass(frozen=True, slots=True)
class AudioData:
    """
    Decoded audio samples held in memory.

    Attributes:
        samples: Audio samples with shape (channels, samples).
        sample_rate: Sample rate in Hz.
    """

    samples: np.ndarray
    sample_rate: int


def load_audio(
    source: str | Path | AudioFile,
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> AudioData:
    """
    Decode an audio file into an in-memory audio representation.

    Args:
        source:
            Audio file path or an existing AudioFile.

        sample_rate: Target sample rate in Hz. If None, preserves the
            source sample rate.

        channels: Target number of channels. If None, preserves the
            source channel count.

    Returns:
        Decoded audio samples as an AudioData instance. Samples are float32 with shape
        (channels, samples).

    Raises:
        FileNotFoundError:
            If the source file does not exist.
        
        RuntimeError:
            If the audio cannot be decoded.

        ValueError:
            If the requested audio configuration is invalid.
    """
    if isinstance(source, AudioFile):
        path = source.path
        source_sample_rate = source.sample_rate
        source_channels = source.channels
    else:
        path = Path(source)
        source_sample_rate = None
        source_channels = None

    if not path.is_file():
        raise FileNotFoundError(path)

    if sample_rate is not None and sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")

    if channels is not None and channels <= 0:
        raise ValueError("channels must be greater than zero")

    if sample_rate is None:
        if source_sample_rate is None:
            source_info = get_audio_info(path)
            source_sample_rate = source_info.sample_rate

        sample_rate = source_sample_rate

    if channels is None:
        if source_channels is None:
            source_info = get_audio_info(path)
            source_channels = source_info.channels

        channels = source_channels

    if sample_rate is None:
        raise RuntimeError(
            f"Unable to determine sample rate for: {path}"
        )

    if channels is None:
        raise RuntimeError(
            f"Unable to determine channel count for: {path}"
        )

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "panic",

        "-i", str(path),

        '-threads', '1',
        "-f", "f32le",

        "-ar", str(sample_rate),
        "-ac", str(channels),

        "-vn",
        "pipe:1",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg was not found. Please install FFmpeg."
        ) from exc
    except subprocess.CalledProcessError as exc:
        error = exc.stderr.decode(errors="replace").strip()

        raise RuntimeError(
            f"Failed to decode audio file: {path}\n"
            f"{error}"
        ) from exc

    samples = np.frombuffer(
        result.stdout,
        dtype=np.float32,
    )

    if samples.size == 0:
        raise RuntimeError(
            f"Audio file produced no samples: {path}"
        )

    if samples.size % channels != 0:
        raise RuntimeError(
            f"Decoded audio has an invalid sample count: {path}"
        )

    samples = samples.reshape(-1, channels).T

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
    )


def get_audio_info(
    source: str | Path | AudioFile,
) -> AudioFile:
    """
    Inspect an audio file without decoding its samples.

    Args:
        source:
            Audio file path or an existing AudioFile.

    Returns:
        AudioFile containing the source path and detected audio
        properties.

    Raises:
        FileNotFoundError:
            If the source file does not exist.

        RuntimeError:
            If the audio metadata cannot be read.
    """
    if isinstance(source, AudioFile):
        if (
            source.sample_rate is not None
            and source.channels is not None
            and source.duration is not None
        ):
            return source

        path = source.path
    else:
        path = Path(source)

    if not path.is_file():
        raise FileNotFoundError(path)

    cmd = [
        "ffprobe",
        "-loglevel", "panic",
        "-select_streams", "a:0",
        
        # Full Output
        # '-show_format', '-show_streams',

        # Specific
        "-show_entries",
        "stream=sample_rate,channels,duration",
        
        '-print_format', 'json',
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe was not found. Please install FFmpeg."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to inspect audio file: {path}\n"
            f"{exc.stderr.strip()}"
        ) from exc

    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"No readable audio stream found in: {path}"
        ) from exc

    sample_rate = stream.get("sample_rate")
    channels = stream.get("channels")
    duration = stream.get("duration")

    return AudioFile(
        path=path,
        sample_rate=int(sample_rate) if sample_rate else None,
        channels=int(channels) if channels else None,
        duration=float(duration) if duration else None,
    )
