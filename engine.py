"""One WhisperX stage per process, so model memory is released between stages."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from languages import ROMANIAN_ALIGNMENT_MODEL

SAMPLE_RATE = 16000
LANGUAGE_REGION_SECONDS = 30


def speech_windows(regions, audio_length):
    """Merge pause-separated fragments into bounded Whisper context windows."""
    first = last = None
    for region in regions:
        start = max(0, round(region.start * SAMPLE_RATE))
        end = min(audio_length, round(region.end * SAMPLE_RATE))
        if end <= start:
            continue
        if first is not None and end - first > LANGUAGE_REGION_SECONDS * SAMPLE_RATE:
            yield first, last
            first = None
        if first is None:
            first = start
        last = end
    if first is not None:
        yield first, last


def transcribe_block(model, audio, language: str | None, batch_size: int) -> list[dict]:
    """Detect language per speech window, avoiding padded passes for every pause."""
    if language is not None:
        result = model.transcribe(audio, batch_size=batch_size, language=language, task="transcribe", print_progress=True)
        return [dict(segment, language=language) for segment in result["segments"]]

    regions = model.vad_model({"waveform": model.vad_model.preprocess_audio(audio), "sample_rate": SAMPLE_RATE})
    segments = []
    detected = None
    for first, last in speech_windows(regions, len(audio)):
        clip = audio[first:last]
        # A trailing interjection/noise burst is poor evidence of a language switch.
        if detected is None or len(clip) >= 3 * SAMPLE_RATE:
            detected = model.detect_language(clip)
        print(f"Speech {first / SAMPLE_RATE:.2f}-{last / SAMPLE_RATE:.2f}s: {detected}", flush=True)
        result = model.transcribe(clip, batch_size=batch_size, language=detected, task="transcribe", print_progress=False)
        for segment in result["segments"]:
            segments.append(dict(segment, start=segment["start"] + first / SAMPLE_RATE,
                                 end=segment["end"] + first / SAMPLE_RATE, language=detected))
    return segments


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_progress(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def release_model(torch, device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    config_path, stage = Path(sys.argv[1]), sys.argv[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["OMP_NUM_THREADS"] = str(config["threads"])
    os.environ["MKL_NUM_THREADS"] = str(config["threads"])
    os.environ["TORCH_HOME"] = str(Path(config["model_dir"]) / "torch")
    import numpy as np
    import torch
    import whisperx

    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    # The media was decoded by FFmpeg. Passing arrays to WhisperX/pyannote avoids
    # TorchCodec's Windows shared-FFmpeg dependency and redundant audio decoding.
    audio = np.memmap(config["audio"], dtype=np.float32, mode="c")
    work = config_path.parent
    device = config["device"]
    cache = config["model_dir"]
    block_samples = config["chunk_minutes"] * 60 * SAMPLE_RATE
    duration = len(audio) / SAMPLE_RATE
    print(f"{stage}: {duration / 60:.1f} minutes of audio", flush=True)

    if stage == "transcribe":
        model = whisperx.load_model(
            config["model"], device, compute_type=config["compute_type"],
            language=config["language"], vad_method="silero", threads=config["threads"],
            task="transcribe",
            vad_options={"chunk_size": LANGUAGE_REGION_SECONDS if config["language"] is None else 30},
            download_root=str(Path(cache) / "whisper"),
        )
        # Save at most a minute of ASR work at a time, including silent blocks.
        step = min(block_samples, 60 * SAMPLE_RATE)
        progress_path = work / "transcribe-progress.json"
        progress = read_progress(progress_path)
        signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if progress.get("signature") != signature or progress.get("samples") != len(audio):
            progress = {}
        segments = progress.get("segments", [])
        resume = progress.get("next_sample", 0)
        for start in range(resume, len(audio), step):
            offset = start / SAMPLE_RATE
            print(f"Transcribing {offset / 60:.1f} / {duration / 60:.1f} min", flush=True)
            local = transcribe_block(model, audio[start:start + step], config["language"], config["batch_size"])
            for segment in local:
                segment["start"] += offset
                segment["end"] += offset
                segments.append(segment)
            save(progress_path, {"signature": signature, "samples": len(audio),
                                 "next_sample": min(start + step, len(audio)), "segments": segments})
        languages = list(dict.fromkeys(segment["language"] for segment in segments))
        language = languages[0] if len(languages) == 1 else ("mixed" if languages else config["language"] or "unknown")
        save(work / "transcribed.json", {"segments": segments, "language": language, "languages": languages, "duration": duration})
        progress_path.unlink(missing_ok=True)

    elif stage == "align":
        source = json.loads((work / "transcribed.json").read_text(encoding="utf-8"))
        if not source["segments"]:
            save(work / "aligned.json", source)
            return 0
        from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH

        progress_path = work / "alignment-progress.json"
        signature = hashlib.sha256((work / "transcribed.json").read_bytes()).hexdigest()
        progress = read_progress(progress_path)
        if progress.get("signature") != signature:
            progress = {"signature": signature, "blocks": {}, "warnings": []}
        segments = []
        languages = list(dict.fromkeys(s.get("language", source["language"]) for s in source["segments"]))
        # Process one language at a time to avoid holding multiple large aligners
        # in memory. Restore chronological order before whole-video diarization.
        for language in languages:
            selected = [s for s in source["segments"] if s.get("language", source["language"]) == language]
            speech_seconds = sum(max(0, s["end"] - s["start"]) for s in selected)
            if len(languages) > 1 and speech_seconds < min(30, duration * 0.01):
                # A few short misdetections used to trigger gigabytes of model
                # downloads. Preserve these fragments, with explicit coarse timing.
                warning = f"Brief detected language {language} ({speech_seconds:.1f}s); original speech timestamps retained."
                print(warning, flush=True)
                if warning not in progress["warnings"]:
                    progress["warnings"].append(warning)
                segments.extend(selected)
                continue
            if language not in DEFAULT_ALIGN_MODELS_HF and language not in DEFAULT_ALIGN_MODELS_TORCH:
                print(f"No default aligner for {language}; retaining speech-region timestamps", flush=True)
                segments.extend(selected)
                continue
            align_model = ROMANIAN_ALIGNMENT_MODEL if language == "ro" else None
            blocks = []
            for start in range(0, len(audio), block_samples):
                offset = start / SAMPLE_RATE
                stop = min(start + block_samples, len(audio)) / SAMPLE_RATE
                local = [dict(s, start=s["start"] - offset, end=s["end"] - offset) for s in selected if offset <= s["start"] < stop]
                if local:
                    blocks.append((start, offset, local))
            if all(f"{language}:{start}" in progress["blocks"] for start, _, _ in blocks):
                for start, _, _ in blocks:
                    segments.extend(progress["blocks"][f"{language}:{start}"])
                continue
            print(f"Alignment language: {language}; model: {align_model or 'WhisperX default'}", flush=True)
            try:
                model, metadata = whisperx.load_align_model(language_code=language, device=device, model_name=align_model, model_dir=str(Path(cache) / "alignment"))
            except (OSError, ValueError, RuntimeError) as error:
                warning = f"Word alignment unavailable for {language} ({type(error).__name__}); original speech timestamps retained."
                print(warning, flush=True)
                progress["warnings"].append(warning)
                for start, offset, local in blocks:
                    key = f"{language}:{start}"
                    progress["blocks"].setdefault(key, [dict(s, start=s["start"] + offset, end=s["end"] + offset) for s in local])
                    segments.extend(progress["blocks"][key])
                save(progress_path, progress)
                release_model(torch, device)
                continue
            for start, offset, local in blocks:
                key = f"{language}:{start}"
                if key in progress["blocks"]:
                    segments.extend(progress["blocks"][key])
                    continue
                print(f"Aligning {language}: {offset / 60:.1f} / {duration / 60:.1f} min", flush=True)
                aligned = whisperx.align(local, model, metadata, audio[start:start + block_samples], device, return_char_alignments=False)
                for segment in aligned["segments"]:
                    segment["language"] = language
                    segment["start"] += offset
                    segment["end"] += offset
                    for word in segment.get("words", []):
                        for time_key in ("start", "end"):
                            if time_key in word:
                                word[time_key] += offset
                    segments.append(segment)
                progress["blocks"][key] = aligned["segments"]
                save(progress_path, progress)
            del model, metadata
            release_model(torch, device)
        segments.sort(key=lambda segment: (segment["start"], segment["end"]))
        result = {**source, "segments": segments}
        if progress["warnings"]:
            result["alignment_warnings"] = progress["warnings"]
        save(work / "aligned.json", result)
        progress_path.unlink(missing_ok=True)

    elif stage == "diarize":
        source = json.loads((work / "aligned.json").read_text(encoding="utf-8"))
        if not source["segments"]:
            save(work / "diarized.json", source)
            return 0
        from huggingface_hub import get_token
        from whisperx.diarize import DiarizationPipeline
        model = DiarizationPipeline(token=get_token(), device=device, cache_dir=str(Path(cache) / "diarization"))
        # Pyannote defaults to batches of 32, which cause heavy paging on 4 GB PCs.
        model.model.segmentation_batch_size = config["batch_size"]
        model.model.embedding_batch_size = config["batch_size"]
        # A single whole-video clustering pass preserves global speaker identities.
        print("Detecting speakers across the entire video...", flush=True)
        last_percent = [-10]
        last_report = [time.monotonic()]

        def progress(value):
            if value >= last_percent[0] + 5 or value >= 100 or time.monotonic() - last_report[0] >= 60:
                print(f"Speaker detection: {value:.0f}%", flush=True)
                last_percent[0] = value
                last_report[0] = time.monotonic()

        speakers = model(audio, num_speakers=config["speakers"], min_speakers=config["min_speakers"], max_speakers=config["max_speakers"], progress_callback=progress)
        if speakers.empty:
            raise RuntimeError("Speech was transcribed but diarization found no speakers. No completed transcript was saved.")
        result = whisperx.assign_word_speakers(speakers, source, fill_nearest=False)
        save(work / "diarized.json", result)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    print(f"{stage}: complete", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
