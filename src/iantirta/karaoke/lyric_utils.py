from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from iantirta.audio import AudioSegment
from iantirta.models import WordTiming

if TYPE_CHECKING:
    from iantirta.models import ASRResult


def timing_overlap(
    t1, t2,
) -> bool:
    return not (t1.end < t2.start or t1.start > t2.end)


def ref2lyric(
    hypothesis: ASRResult,
    lyric: str
)-> list[WordTiming]:
    """
    Aligns ASR hypothesis words to the reference lyric using Needleman-Wunsch.
    """
    from iantirta.text import align, normalize
    
    lyric_words = [
        (word, norm)
        for line in lyric.splitlines()
        if (_line := line.strip())
        and not _line.startswith("[")
        for word in _line.split()
        if (norm := normalize(word, kaldi=True).strip())
    ]
    clean_lyric_words = [norm for _, norm in lyric_words]

    valid_hyp_words = [
        (w, norm)
        for text in hypothesis.texts
        for w in text.words
        if (norm := normalize(w.clean, kaldi=True).strip())
    ]
    clean_hypothesis = [norm for _, norm in valid_hyp_words]

    alignments = align(
        clean_lyric_words,
        clean_hypothesis
    )

    ref_tokens: list[WordTiming] = []
    hyp_idx = 0
    lyric_idx = 0

    for alignment in alignments:
        # alignment.left = reference/lyric
        # alignment.right = hypothesis/asr
        if alignment.left is not None:
            start, end, score = None, None, None

            if alignment.right is not None:
                hyp_word, _ = valid_hyp_words[hyp_idx]
                start = float(round(hyp_word.start, 2))
                end = float(round(hyp_word.end, 2))
                score = hyp_word.score

            ref_tokens.append(WordTiming(
                start=start,
                end=end,
                score=score,
                word=lyric_words[lyric_idx][0]
            ))
            lyric_idx += 1

        if alignment.right is not None:
            hyp_idx += 1

    return ref_tokens


def align2ref(
    audio_segments: list[AudioSegment],
    hypothesis: ASRResult,
    lyric: str,
):
    """
    Method that is used to align
    asr result into the actual lyric.
    """
    ref_tokens = ref2lyric(
        hypothesis=hypothesis,
        lyric=lyric
    )

    audio2refs = defaultdict(list)
    for i, aseg in enumerate(audio_segments):
        audio2refs[i] = []
        for j, token in enumerate(ref_tokens):
            if token.start is None:
                continue
            if timing_overlap(aseg, token):
                audio2refs[i].append(j)

    # Interpolation
    for audio_idx, token_ids in audio2refs.items():
        if token_ids:
            audio2refs[audio_idx] = list(range(
                min(token_ids), max(token_ids) + 1
            ))

    # Extrapolation
    unused_audio_gaps = {}
    for i, aseg in enumerate(audio_segments):
        unused_audio_gaps[i] = (None, None)
        token_ids = audio2refs[i]
        if not token_ids:
            continue

        start_gap = ref_tokens[min(token_ids)].start - aseg.start
        end_gap = aseg.end - ref_tokens[max(token_ids)].end
        unused_audio_gaps[i] = (start_gap, end_gap)

    