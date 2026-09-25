# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from iantirta.karaoke.audio.files import AudioFile, get_audio_info, load_audio

# Helper for writing mock audio

def _write_wav(
    path,
    *,
    sample_rate=16000,
    channels=1,
    duration=1.0,
):
    t = np.arange(
        int(sample_rate * duration),
        dtype=np.float32,
    ) / sample_rate

    signal = 0.5 * np.sin(2 * np.pi * 440 * t)

    data = np.repeat(
        signal[:, None],
        channels,
        axis=1,
    )

    data = (data * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(data.tobytes())


def test_audio_file():
    path = Path("song.wav")

    audio = AudioFile(
        path=path,
        sample_rate=16000,
        channels=2,
        duration=10.5,
    )

    assert audio.path == path
    assert audio.sample_rate == 16000
    assert audio.channels == 2
    assert audio.duration == 10.5


def test_get_audio_info_missing_file(tmp_path):
    path = tmp_path / "missing.wav"

    with pytest.raises(FileNotFoundError):
        get_audio_info(path)


def test_get_audio_info(tmp_path):
    path = tmp_path / "test.wav"

    _write_wav(
        path,
        sample_rate=16000,
        channels=2,
        duration=1.0,
    )

    info = get_audio_info(path)
    
    assert info.path == path
    assert info.sample_rate == 16000
    assert info.channels == 2
    assert info.duration == pytest.approx(1.0, abs=0.01)


def test_load_audio(tmp_path):
    path = tmp_path / "test.wav"

    _write_wav(
        path,
        sample_rate=16000,
        channels=2,
        duration=1.0,
    )

    audio = load_audio(path)

    assert audio.sample_rate == 16000
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 2
    assert audio.samples.shape == (2, 16000)


def test_load_audio_resample(tmp_path):
    path = tmp_path / "test.wav"

    _write_wav(
        path,
        sample_rate=16000,
        channels=1,
        duration=1.0,
    )

    audio = load_audio(
        path,
        sample_rate=8000,
    )

    assert audio.sample_rate == 8000
    assert audio.samples.shape == (1, 8000)


def test_load_audio_channels(tmp_path):
    path = tmp_path / "test.wav"

    _write_wav(
        path,
        sample_rate=16000,
        channels=2,
        duration=1.0,
    )

    audio = load_audio(
        path,
        channels=1,
    )

    assert audio.sample_rate == 16000
    assert audio.samples.shape == (1, 16000)


def test_load_audio_decodes_signal(tmp_path):
    path = tmp_path / "test.wav"

    _write_wav(path)

    audio = load_audio(path)

    assert audio.samples.shape == (1, 16000)
    assert audio.samples.dtype == np.float32
    assert np.min(audio.samples) == pytest.approx(-0.5, abs=0.01)
    assert np.max(audio.samples) == pytest.approx(0.5, abs=0.01)


def test_get_audio_info_audio_file(tmp_path):
    path = tmp_path / "test.wav"

    _write_wav(
        path,
        sample_rate=16000,
        channels=2,
        duration=1.0,
    )

    source = AudioFile(path=path)

    info = get_audio_info(source)

    assert info.path == path
    assert info.sample_rate == 16000
    assert info.channels == 2


def test_get_audio_info_returns_complete_audio_file():
    source = AudioFile(
        path=Path("test.wav"),
        sample_rate=16000,
        channels=2,
        duration=1.0,
    )

    assert get_audio_info(source) is source


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_rate": 0}, "sample_rate"),
        ({"sample_rate": -1}, "sample_rate"),
        ({"channels": 0}, "channels"),
        ({"channels": -1}, "channels"),
    ],
)
def test_load_audio_invalid_configuration(
    tmp_path,
    kwargs,
    message,
):
    path = tmp_path / "test.wav"

    _write_wav(path)

    with pytest.raises(ValueError, match=message):
        load_audio(path, **kwargs)


def test_load_audio_resample_and_channels(tmp_path):
    path = tmp_path / "test.wav"

    _write_wav(
        path,
        sample_rate=44100,
        channels=2,
        duration=1.0,
    )

    audio = load_audio(
        path,
        sample_rate=16000,
        channels=1,
    )

    assert audio.sample_rate == 16000
    assert audio.samples.shape == (1, 16000)


def test_load_audio_preserves_stereo_channels(tmp_path):
    path = tmp_path / "test.wav"

    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate

    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.25 * np.sin(2 * np.pi * 880 * t)

    data = np.stack([left, right], axis=1)
    data = (data * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(data.tobytes())

    audio = load_audio(path)

    assert audio.samples.shape == (2, sample_rate)

    assert np.max(np.abs(audio.samples[0])) == pytest.approx(
        0.5,
        abs=0.01,
    )
    assert np.max(np.abs(audio.samples[1])) == pytest.approx(
        0.25,
        abs=0.01,
    )
