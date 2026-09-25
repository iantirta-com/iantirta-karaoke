# Part of Iantirta.com
# See LICENSE file for full copyright and licensing details.

from __future__ import annotations

from dataclasses import dataclass


__all__ = ["Lyrics", "get_lyrics"]


@dataclass(frozen=True, slots=True)
class Lyrics:
    """
    Represents lyrics retrieved from a lyrics provider.

    Attributes:
        text: Plain lyrics text.
        synced_text: Optional timestamped lyrics text.
        source: Provider or source identifier.
    """

    text: str
    synced_text: str | None = None
    source: str | None = None


def get_lyrics(
    *,
    artist: str,
    title: str,
) -> Lyrics | None:
    """
    Retrieve lyrics for a song.

    Args:
        artist: Artist or performer name.
        title: Song title.

    Returns:
        Lyrics if matching lyrics are found, otherwise None.
    """
    ...