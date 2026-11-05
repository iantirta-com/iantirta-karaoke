
import pytest

from iantirta.audio import AudioSegment
from iantirta.models import WordTiming
from iantirta.karaoke.lyric_utils import timing_overlap


@pytest.mark.parametrize(
    ('aseg', 'word', 'expected'),
    [
        # Fit in between
        ((0.0, 1.0), (0.02, 0.98), True),
        # Fully Over
        ((1.0, 2.0), (0.98, 2.1), True),
        # Left overlap
        ((1.0, 2.0), (0.5, 1.4), True),
        # Right overlap
        ((2.0, 2.5), (2.2, 2.9), True),
        # Equal
        ((2.0, 3.0), (2.0, 3.0), True)
    ]
)
def test_timing_overlap(aseg, word, expected):
    aseg = AudioSegment(
        start=aseg[0],
        end=aseg[1]
    )
    word = WordTiming(
        start=word[0],
        end=word[1],
        word="",
        score=None
    )
    assert timing_overlap(aseg, word) == expected


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(['-v', '-s', __file__]))
