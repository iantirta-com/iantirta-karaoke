from __future__ import annotations

import logging
from collections import defaultdict

import iantirta.text as text_utils
from iantirta.audio import AudioSegment
from iantirta.models import ASRResult, TextTiming, WordTiming

_logger = logging.getLogger(__name__)


def timing_overlap(
    t1, t2,
) -> bool:
    return not (t1.end < t2.start or t1.start > t2.end)


def ref2lyric(
    hypothesis: ASRResult,
    lyric: str
) -> list[WordTiming]:
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
        (w, norm, text.language)
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
            start, end, score, lang = None, None, None, None

            if alignment.right is not None:
                hyp_word, _, lang = valid_hyp_words[hyp_idx]
                start = float(round(hyp_word.start, 2))
                end = float(round(hyp_word.end, 2))
                score = hyp_word.score

            ref_tokens.append(WordTiming(
                start=start,
                end=end,
                score=score,
                word=lyric_words[lyric_idx][0],
                language=lang,
            ))
            lyric_idx += 1

        if alignment.right is not None:
            hyp_idx += 1

    return ref_tokens


def align2ref(
    audio_segments: list[AudioSegment],
    hypothesis: ASRResult,
    lyric: str,
    *,
    logger: logging.Logger | None = None
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

    
    def _check_missing(max_id: int):
        return list(range(last_word_idx + 1, max_id))


    last_word_idx = -1
    last_audio_id = len(audio_segments) - 1
    for i in range(len(audio_segments)):

        # This could be defined as
        # [1, 2, 3, 4, 7] and current id is 7
        # So it is missing 5-6 idx.
        is_last_idx_and_still_missing_word = (
            i == last_audio_id and
            (missing := _check_missing(len(ref_tokens)))
        )
        if is_last_idx_and_still_missing_word:
            (logger or _logger).debug(
                "Subject already in a last segment, "
                f"but there is still missing word {missing}")

            prev_anchor_audio_id = max(
                k for k, v in audio2refs.items()
                if v
            )

            _, prev_end_gap = unused_audio_gaps[prev_anchor_audio_id]
            if prev_end_gap > 0.4:
                (logger or _logger).debug(
                    "  Previous end gap is still large, "
                    "assigned it to previous "
                    f"{audio2refs[prev_anchor_audio_id]} | {missing}"
                )
                audio2refs[prev_anchor_audio_id] = sorted(
                    audio2refs[prev_anchor_audio_id] + missing
                )

            else:
                (logger or _logger).debug(
                    "  Previous end gap insufficient, "
                    "try getting the next audio"
                )

                if prev_anchor_audio_id < i:
                    (logger or _logger).debug(
                        "  Next audio exists, "
                        f"{prev_anchor_audio_id+1} | "
                        f"{audio2refs[prev_anchor_audio_id+1]}"
                    )
                    audio2refs[prev_anchor_audio_id+1] = sorted(missing)

                else:
                    raise NotImplementedError()

        token_ids = audio2refs[i]
        if not token_ids:
            continue


        is_previous_word_still_missing = (
            min(token_ids) > last_word_idx + 1 and
            (missing := _check_missing(min(token_ids)))
        )
        if is_previous_word_still_missing:
            (logger or _logger).debug(
                "Subject still skipping words: "
                f"{audio2refs[i-1]} | {missing} | "
                f"{audio2refs[i]}"
            )

            missing_text = " ".join(ref_tokens[j].word for j in missing)
            known_text = " ".join(ref_tokens[j].word for j in token_ids)
            if prev_anchor_audio_id := next((
                j for j in range(i - 1, -1, -1)
                if audio2refs[j]
            ), None):
                (logger or _logger).debug(
                    "  Previous Anchor Audio Exists: "
                    f"{prev_anchor_audio_id}"
                )

                _, anchor_end_gap = unused_audio_gaps[prev_anchor_audio_id]
                if anchor_end_gap > 1.0:
                    (logger or _logger).debug(
                        "  Previous anchor gap still high "
                        "assigned to previous anchor "
                        f"{sorted(audio2refs[prev_anchor_audio_id] + missing)}"
                    )

                    audio2refs[prev_anchor_audio_id] = sorted(
                        audio2refs[prev_anchor_audio_id] + missing
                    )

                else:
                    (logger or _logger).debug(
                        "  Previous anchor gap insufficient, "
                        "combining to previous audio: "
                        f"{audio2refs[i-1]} | {missing} | {token_ids}"
                    )

                    audio2refs[i] = sorted(
                        token_ids + missing + audio2refs[i-1]
                    )
                    audio2refs[i-1] = sorted(
                        token_ids + missing + audio2refs[i-1]
                    )

            elif (ratio := text_utils.similarity(missing_text, known_text)) > 0.7:
                (logger or _logger).debug(
                    f"  Similarity high ({ratio}) between\n"
                    f"{missing_text} | {known_text}\n"
                    f"{sorted(token_ids + missing)}"
                )

                audio2refs[i] = sorted(token_ids + missing)

            else:
                (logger or _logger).debug(
                    "  Combining to previous Audio: "
                    f"{audio2refs[i-1]} | {missing} | {token_ids}"
                )

                audio2refs[i] = sorted(
                    token_ids + missing + audio2refs[i-1]
                )
                audio2refs[i-1] = sorted(
                    token_ids + missing + audio2refs[i-1]
                )


        token_ids = audio2refs[i]

        start_gap, end_gap = unused_audio_gaps[i]
        if i < last_audio_id and (next_token_ids := audio2refs[i+1]):
            next_start_gap, _ = (
                unused_audio_gaps[i+1]
                if i < last_audio_id
                else (None, None)
            )

            if missing := sorted(set(range(
                max(token_ids) + 1, min(next_token_ids)
            ))):
                (logger or _logger).debug(
                    "Subject is missing: "
                    f"{token_ids} | {missing} | {next_token_ids}"
                )

                if next_start_gap > end_gap:
                    (logger or _logger).debug(
                        "  Belong to the next: "
                        f"{sorted(next_token_ids + missing)}"
                    )
                    audio2refs[i+1] = sorted(next_token_ids + missing)

                else:
                    (logger or _logger).debug(
                        "  Combine Both: "
                        f"{sorted(token_ids + missing + next_token_ids)}"
                    )
                    audio2refs[i] = sorted(
                        token_ids + missing + next_token_ids
                    )

            elif end_gap > 2.0 or next_start_gap > 1.0:
                # No Unknown but the gap in either each segment is so long
                # So we combine
                (logger or _logger).debug(
                    "Subject for combine: "
                    f"{audio2refs[i]} | {audio2refs[i+1]}"
                )
                audio2refs[i] = sorted(
                    audio2refs[i] + audio2refs[i+1]
                )
                audio2refs[i+1] = sorted(
                    audio2refs[i] + audio2refs[i+1]
                )

        token_ids = audio2refs[i]
        last_word_idx = max(token_ids)


    assigned = [
        token_id
        for token_ids in audio2refs.values()
        for token_id in token_ids
    ]
    missing = (
        set(range(len(ref_tokens))) - set(assigned)
    )
    if missing:
        (logger or _logger).warning(f"Missing: {missing}")

    def should_merge(i, j) -> bool:
        a_tokens = audio2refs[i]
        b_tokens = audio2refs[j]
        if not a_tokens or not b_tokens:
            return False
        return bool(not set(a_tokens).isdisjoint(set(b_tokens)))

    # Clustering
    clusters, current = [], None
    for i in range(len(audio_segments)):
        if not audio2refs[i]:
            continue

        if current is None:
            current = {
                "audio_ids": [i],
                "token_ids": list(audio2refs[i])
            }
            continue

        prev_i = current["audio_ids"][-1]
        if should_merge(prev_i, i):
            current["audio_ids"].append(i)
            current["token_ids"].extend(audio2refs[i])
            current["token_ids"] = sorted(set(current["token_ids"]))
        else:
            clusters.append(current)
            current = {
                "audio_ids": [i],
                "token_ids": list(audio2refs[i])
            }

    if current:
        clusters.append(current)

    new_audio_segments = []
    new_texts: list[TextTiming] = []
    for i, cluster in enumerate(clusters):
        audio_ids = cluster["audio_ids"]
        token_ids = cluster["token_ids"]

        first: AudioSegment = audio_segments[audio_ids[0]]
        last: AudioSegment = audio_segments[audio_ids[-1]]
        new_audio_segments.append(
            AudioSegment(
                start=first.start, end=last.end,
            )
        )

        words, langs = [], []
        for token_id in token_ids:
            token: WordTiming = ref_tokens[token_id]

            langs.append(token.language)
            words.append(WordTiming(
                start=token.start,
                end=token.end,
                score=token.score,
                word=token.word,
            ))

        if not langs:
            langs = ["English"]

        new_texts.append(TextTiming(words=words, language=langs[0]))

    assert len(new_texts) == len(new_audio_segments)

    new_words = [w for text in new_texts for w in text.words]
    assert len(new_words) == len(ref_tokens)

    return ASRResult(texts=new_texts), new_audio_segments
