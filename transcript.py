"""Human-readable transcripts, split whenever the word-level speaker changes."""
from __future__ import annotations

import math
import re
import unicodedata


def timestamp(seconds: float) -> str:
    millis = max(0, round(float(seconds) * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    seconds, millis = divmod(millis, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"


def safe_title(title: str, limit: int = 100) -> str:
    title = unicodedata.normalize("NFC", title)
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title)
    title = re.sub(r"\s+", " ", title).strip(" .")[:limit].rstrip(" .")
    if not title:
        title = "Untitled video"
    if title.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        title = "_" + title
    return title


def finite_time(value, fallback: float) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else fallback
    except (TypeError, ValueError):
        return fallback


def join_words(words: list[str], language: str) -> str:
    # WhisperX alignment removes leading spaces, including for non-space languages.
    if language in {"zh", "ja", "th", "lo", "my", "yue"}:
        return "".join(words).strip()
    result = " ".join(word.strip() for word in words if word.strip())
    result = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", result)
    return result.strip()


def speaker_turns(segments: list[dict], language: str = "en", max_seconds: float = 30) -> list[dict]:
    turns: list[dict] = []
    for segment in segments:
        segment_language = segment.get("language", language)
        start = finite_time(segment.get("start"), 0)
        end = max(start, finite_time(segment.get("end"), start))
        words = segment.get("words") or []
        # An alignment failure must never silently discard the original text.
        if not words or not any(str(word.get("word", "")).strip() for word in words):
            if str(segment.get("text", "")).strip():
                turns.append({"start": start, "end": end, "speaker": segment.get("speaker", "UNKNOWN"), "text": segment["text"].strip()})
            continue
        group = None
        previous_end = start
        for word in words:
            text = str(word.get("word", ""))
            if not text.strip():
                continue
            word_start = max(start, finite_time(word.get("start"), previous_end))
            word_end = max(word_start, finite_time(word.get("end"), word_start))
            # Untimed words (numbers, symbols) inherit the segment's label and bounds.
            speaker = word.get("speaker") or segment.get("speaker") or "UNKNOWN"
            if group is None or speaker != group["speaker"] or word_end - group["start"] > max_seconds or word_start - group["end"] > 2:
                if group is not None:
                    group["text"] = join_words(group.pop("words"), segment_language)
                    turns.append(group)
                group = {"start": word_start, "end": word_end, "speaker": speaker, "words": [text]}
            else:
                group["end"] = max(group["end"], word_end)
                group["words"].append(text)
            previous_end = word_end
        if group is not None:
            group["text"] = join_words(group.pop("words"), segment_language)
            turns.append(group)
    return turns


def render_transcript(info: dict, result: dict, model: str, url: str, preview_seconds: int | None = None) -> str:
    language = result.get("language", "unknown")
    turns = speaker_turns(result.get("segments", []), language)
    labels: dict[str, str] = {}
    for turn in turns:
        speaker = turn["speaker"]
        if speaker != "UNKNOWN" and speaker not in labels:
            labels[speaker] = f"SPEAKER {len(labels) + 1:02}"
    lines = [str(info.get("title") or "Untitled video"), "", f"Source: {url}", f"Language: {language}", f"Speakers detected: {len(labels)}", f"Transcription: WhisperX / {model}"]
    if language == "mixed":
        lines.insert(4, "Languages detected: " + ", ".join(result.get("languages") or dict.fromkeys(s.get("language", "unknown") for s in result.get("segments", []))))
    if preview_seconds:
        lines.append(f"PREVIEW ONLY: first {preview_seconds} seconds (or video end, if shorter).")
    for warning in result.get("alignment_warnings", []):
        lines.append(f"Timing note: {warning}")
    lines.extend(["Speaker numbers identify voices within this video; they are not verified names.", ""])
    for turn in turns:
        label = labels.get(turn["speaker"], "SPEAKER UNKNOWN")
        lines.append(f"[{timestamp(turn['start'])} --> {timestamp(turn['end'])}] {label}: {turn['text']}")
    if not turns:
        lines.append("No speech detected.")
    return "\n".join(lines) + "\n"
