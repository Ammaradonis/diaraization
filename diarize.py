"""Local YouTube/Rumble -> WhisperX -> timestamped speaker transcripts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse

from transcript import render_transcript, safe_title
from languages import ROMANIAN_ALIGNMENT_MODEL, language_code

ROOT = Path(__file__).resolve().parent
MODEL_REPO = "pyannote/speaker-diarization-community-1"
ENGINE_VERSION = 2  # Keep compatible completed stages reusable after recovery fixes.
STOP = threading.Event()
PRINT_LOCK = threading.RLock()


class JobError(Exception):
    pass


def say(message: str) -> None:
    with PRINT_LOCK:
        print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_credentials(path: Path) -> None:
    """Read a local token without executing the file or persisting its contents."""
    if os.environ.get("HF_TOKEN") or not path.is_file():
        return
    content = path.read_text(encoding="utf-8-sig")
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export\s+)?(?:HF_TOKEN|HUGGING_FACE_TOKEN|HUGGINGFACE_TOKEN|HUGGINGFACE_HUB_TOKEN)\s*=\s*['\"]?(hf_[A-Za-z0-9]+)['\"]?\s*(?:#.*)?", line)
        if match:
            os.environ["HF_TOKEN"] = match.group(1)
            return
        if re.fullmatch(r"hf_[A-Za-z0-9]+", line):
            os.environ["HF_TOKEN"] = line
            return
    raise JobError("No valid Hugging Face token found in the credential file. Use HF_TOKEN=hf_... or HUGGING_FACE_TOKEN=hf_... . Its contents were not logged.")


def canonical_url(value: str) -> str:
    value = value.strip().strip('\"\'')
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise ValueError("Expected an http(s) YouTube or Rumble video URL")
    host = (parsed.hostname or "").lower().removeprefix("www.")
    parts = parsed.path.strip("/").split("/")
    if host == "youtu.be":
        video_id = parts[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif len(parts) == 2 and parts[0] in {"live", "shorts", "embed"}:
            video_id = parts[1]
        else:
            raise ValueError("Use a single YouTube video URL, not a channel or playlist")
    elif host == "rumble.com":
        if (len(parts) == 1 and re.fullmatch(r"v[\w-]+\.html", parts[0])) or (len(parts) == 2 and parts[0] == "embed" and re.fullmatch(r"v[\w-]+", parts[1])):
            return "https://rumble.com/" + "/".join(parts) + ("/" if parts[0] == "embed" else "")
        raise ValueError("Use a Rumble video (.html) or /embed/video-id/ URL")
    else:
        raise ValueError("Only YouTube and Rumble video URLs are supported")
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("The YouTube video ID must contain 11 characters")
    return f"https://www.youtube.com/watch?v={video_id}"


def read_jobs(path: Path | None, urls: list[str] | None = None) -> list[str]:
    lines = urls if urls else path.read_text(encoding="utf-8-sig").splitlines()
    jobs, seen, errors = [], set(), []
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            url = canonical_url(line)
        except ValueError as error:
            errors.append(f"Line {number}: {error}: {line}")
            continue
        if url not in seen:
            seen.add(url)
            jobs.append(url)
    if errors:
        raise JobError("Fix these jobs.txt entries before running:\n" + "\n".join(errors))
    if not jobs:
        raise JobError("No video URLs found. Add one URL per line to jobs.txt.")
    return jobs


def key_for(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--jobs", type=Path, default=ROOT / "jobs.txt", help="UTF-8 file with one video URL per line")
    p.add_argument("--url", action="append", help="Process this URL instead of jobs.txt; repeat for multiple URLs")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "out", help="Transcript directory")
    p.add_argument("--workers", type=positive, default=3, help="Maximum concurrent video jobs")
    p.add_argument("--inference-workers", type=positive, default=None, help="Concurrent model processes; auto: one below 12 GB RAM, otherwise up to workers (one on CUDA)")
    p.add_argument("--model", default="base", help="Whisper model: tiny, base, small, medium, large-v3, etc.")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Inference device; CUDA requires a separate compatible PyTorch installation")
    p.add_argument("--compute-type", choices=["int8", "float32", "float16", "int8_float16"], default=None, help="Default: int8 on CPU, float16 on CUDA")
    p.add_argument("--batch-size", type=positive, default=1, help="Transcription and speaker detection batch size per model process")
    p.add_argument("--threads", type=positive, default=None, help="CPU threads per model process; auto divides available cores")
    p.add_argument("--language", type=language_code, default=None, help="Auto detects language in speech windows of up to 30 seconds; ro, en, etc. forces one language for all speech")
    speakers = p.add_mutually_exclusive_group()
    speakers.add_argument("--speakers", type=positive, help="Exact speaker count, only if known")
    speakers.add_argument("--min-speakers", type=positive, help="Minimum speaker count, e.g. 5")
    p.add_argument("--max-speakers", type=positive, help="Maximum speaker count; no tool-imposed cap by default")
    p.add_argument("--chunk-minutes", type=positive, default=10, help="ASR/alignment chunk size; diarization still covers the whole video")
    p.add_argument("--limit", type=positive, help="Use only the first N unique jobs")
    p.add_argument("--preview-seconds", type=positive, help="Transcribe only the start; writes to out/preview and never marks the full video done")
    p.add_argument("--cookies", type=Path, help="Optional Netscape cookies file for videos your account can access")
    p.add_argument("--env-file", type=Path, default=ROOT / "env.txt", help="Optional local token file; HF_TOKEN or HUGGING_FACE_TOKEN is recognized")
    p.add_argument("--ffmpeg-location", type=Path, help="Folder containing ffmpeg and ffprobe")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache", help="Downloaded audio and restart checkpoints")
    p.add_argument("--model-dir", type=Path, default=ROOT / "data" / "models", help="Downloaded model cache")
    p.add_argument("--log-dir", type=Path, default=ROOT / "data" / "logs", help="Per-job stage logs and batch summaries")
    p.add_argument("--retries", type=int, default=3, help="Downloader retries; must be 0 or greater")
    p.add_argument("--stall-timeout-minutes", type=positive, default=30, help="Stop an inference process after this many minutes without log activity; completed checkpoints are retained")
    p.add_argument("--keep-audio", action="store_true", help="Keep decoded audio after successful transcription")
    p.add_argument("--force", action="store_true", help="Recompute finished inference stages and replace this video's transcript")
    p.add_argument("--download-only", action="store_true", help="Cache audio without loading models or requiring Hugging Face login")
    p.add_argument("--dry-run", action="store_true", help="Validate and list jobs without downloading or loading models")
    p.add_argument("--check", action="store_true", help="Check dependencies, model access, and available memory; no video processing")
    return p


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def child(command: list[str], log_path: Path, label: str, stall_seconds: float | None = None) -> None:
    if STOP.is_set():
        raise JobError("Cancelled")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] Starting {label}\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        started = last_report = last_activity = time.monotonic()
        last_size = log_path.stat().st_size
        try:
            while process.poll() is None:
                if STOP.wait(0.5):
                    raise JobError("Cancelled")
                now = time.monotonic()
                size = log_path.stat().st_size
                if size != last_size:
                    last_activity, last_size = now, size
                if stall_seconds and now - last_activity >= stall_seconds:
                    raise JobError(f"{label} stopped: no log activity for {stall_seconds / 60:g} minutes. Checkpoints retained; see {log_path}")
                if now - last_report >= 60:
                    progress = ""
                    if stall_seconds:
                        with log_path.open("rb") as reader:
                            reader.seek(max(0, size - 4096))
                            lines = reader.read().decode("utf-8", errors="replace").splitlines()
                        progress = next((line for line in reversed(lines) if line.startswith(("Speaker detection:", "Transcribing ", "Aligning ", "Speech ", "Progress:"))), "")
                    say(f"{label}: {(now - started) / 60:.0f} min elapsed; {progress or 'running'}; log: {log_path.name}")
                    last_report = now
            if process.returncode:
                raise JobError(f"{label} failed (exit {process.returncode}). See {log_path}")
        finally:
            terminate(process)
            log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] {label}: exit {process.returncode}; elapsed {time.monotonic() - started:.1f}s\n")


@contextmanager
def job_lock(directory: Path):
    """OS advisory lock: automatically released even after a crash."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as handle:
        if os.fstat(handle.fileno()).st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise JobError("This video is already being processed in another terminal") from error
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise JobError("This video is already being processed in another terminal") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def completed_path(out: Path, job_id: str, fingerprint: str) -> Path | None:
    record = read_json(out / ".state" / f"{job_id}.json")
    if record.get("fingerprint") != fingerprint:
        return None
    name = record.get("filename", "")
    if not name or Path(name).name != name:
        return None
    candidate = out / name
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate
    return None


def download(url: str, work: Path, args, job_id: str) -> dict:
    audio = work / "audio.f32"
    info_path = work / "video.json"
    cached = read_json(info_path)
    if audio.is_file() and audio.stat().st_size > 0 and audio.stat().st_size % 4 == 0 and cached.get("url") == url:
        say(f"{job_id}: reusing downloaded audio")
        return cached
    cookie_copy = work / "cookies.private.txt"
    command = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-playlist", "--no-progress", "--no-colors", "--newline", "--no-simulate", "--write-info-json", "--no-write-playlist-metafiles", "--retries", str(args.retries), "--fragment-retries", str(args.retries), "--extractor-retries", str(args.retries), "--socket-timeout", "30", "--match-filter", "!is_live & live_status!=is_upcoming", "--format", "bestaudio/best[height<=480]/worst", "--output", str(work / "media.%(ext)s")]
    if urlparse(url).hostname == "rumble.com":
        # Rumble's public embed endpoint fingerprints TLS clients (yt-dlp #17496).
        command.extend(["--impersonate", "chrome"])
    if shutil.which("deno"):
        command.extend(["--js-runtimes", "deno"])
    elif shutil.which("node"):
        command.extend(["--js-runtimes", "node"])
    if args.ffmpeg_location:
        command.extend(["--ffmpeg-location", str(args.ffmpeg_location)])
    if args.cookies:
        shutil.copyfile(args.cookies, cookie_copy)
        command.extend(["--cookies", str(cookie_copy)])
    if args.preview_seconds:
        command.extend(["--download-sections", f"*0-{args.preview_seconds}"])
    command.append(url)
    say(f"{job_id}: downloading {url}")
    try:
        child(command, args.log_dir / f"{job_id}.download.log", f"{job_id} download")
    finally:
        cookie_copy.unlink(missing_ok=True)
    metadata = read_json(work / "media.info.json")
    if not metadata or metadata.get("is_live") or metadata.get("live_status") in {"is_live", "is_upcoming"}:
        raise JobError("No downloadable finished video. Live/upcoming streams are excluded; retry after the recording is published.")
    media = [file for file in work.glob("media.*") if file.suffix.lower() in {".webm", ".m4a", ".mp4", ".mp3", ".ogg", ".opus", ".mkv", ".ts", ".flv", ".aac", ".wav", ".mov"}]
    if len(media) != 1:
        raise JobError(f"Expected one downloaded media file in {work}; found {len(media)}")
    info = {"url": url, "title": metadata.get("title") or metadata.get("id") or "Untitled video", "video_id": metadata.get("id"), "extractor": metadata.get("extractor_key"), "duration": metadata.get("duration")}
    temporary = work / "audio.partial.f32"
    decode = [args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(media[0]), "-vn", "-ac", "1", "-ar", "16000", "-threads", "1"]
    if args.preview_seconds:
        decode.extend(["-t", str(args.preview_seconds)])
    decode.extend(["-f", "f32le", str(temporary)])
    child(decode, args.log_dir / f"{job_id}.decode.log", f"{job_id} audio conversion")
    if not temporary.is_file() or temporary.stat().st_size == 0 or temporary.stat().st_size % 4:
        raise JobError("FFmpeg produced empty or invalid audio")
    os.replace(temporary, audio)
    atomic_json(info_path, info)
    media[0].unlink()
    # yt-dlp's full metadata contains expiring media URLs; retain only useful fields.
    (work / "media.info.json").unlink(missing_ok=True)
    return info


def inference_settings(args) -> dict:
    return {name: getattr(args, name) for name in ("model", "device", "compute_type", "language", "batch_size", "speakers", "min_speakers", "max_speakers", "chunk_minutes", "preview_seconds")}


def run_job(url: str, args, semaphore: threading.Semaphore, fingerprint: str) -> dict:
    job_id = key_for(url)
    suffix = f"-preview-{args.preview_seconds}" if args.preview_seconds else ""
    work = args.cache_dir / (job_id + suffix)
    record = {"url": url, "id": job_id, "status": "failed"}
    try:
        with job_lock(work):
            done = completed_path(args.out, job_id, fingerprint)
            if done and not args.force and not args.download_only:
                say(f"{job_id}: already complete -> {done.name}")
                return {**record, "status": "skipped", "output": str(done)}
            if args.force and not args.download_only:
                state_path = args.out / ".state" / f"{job_id}.json"
                atomic_json(state_path, {**read_json(state_path), "fingerprint": None})
            stages = work / fingerprint
            # A crash between diarization and text export must not require audio,
            # network access, or another model slot to finish the saved result.
            ready = read_json(stages / "diarized.json") if not args.force else {}
            info = read_json(work / "video.json")
            export_ready = "segments" in ready and "language" in ready and info.get("url") == url
            if args.download_only or not export_ready:
                info = download(url, work, args, job_id)
            if args.download_only:
                return {**record, "status": "downloaded", "title": info["title"]}
            stages.mkdir(parents=True, exist_ok=True)
            config = {**inference_settings(args), "threads": args.threads, "audio": str(work / "audio.f32"), "model_dir": str(args.model_dir)}
            atomic_json(stages / "config.json", config)
            if not export_ready:
                say(f"{job_id}: waiting for a model slot - {info['title']}")
                while not semaphore.acquire(timeout=0.5):
                    if STOP.is_set():
                        raise JobError("Cancelled")
            try:
                stage_files = [("transcribe", "transcribed.json"), ("align", "aligned.json"), ("diarize", "diarized.json")]
                for index, (stage, checkpoint) in enumerate(stage_files if not export_ready else []):
                    if STOP.is_set():
                        raise JobError("Cancelled")
                    existing = read_json(stages / checkpoint)
                    if not args.force and "segments" in existing and "language" in existing:
                        say(f"{job_id}: reusing {stage} checkpoint")
                        continue
                    # Recomputed upstream data makes every downstream checkpoint stale.
                    # Invalidate before starting so a crash cannot reuse older results.
                    for _, stale in stage_files[index:]:
                        (stages / stale).unlink(missing_ok=True)
                    if stage == "transcribe":
                        (stages / "alignment-progress.json").unlink(missing_ok=True)
                        if args.force:
                            (stages / "transcribe-progress.json").unlink(missing_ok=True)
                    if args.force and stage == "align":
                        (stages / "alignment-progress.json").unlink(missing_ok=True)
                    say(f"{job_id}: {stage} - {info['title']}")
                    child([sys.executable, str(ROOT / "engine.py"), str(stages / "config.json"), stage], args.log_dir / f"{job_id}.{stage}.log", f"{job_id} {stage}", stall_seconds=getattr(args, "stall_timeout_minutes", 30) * 60)
            finally:
                if not export_ready:
                    semaphore.release()
            result = read_json(stages / "diarized.json")
            if "segments" not in result:
                raise JobError("Diarization returned no valid result")
            filename = f"{safe_title(info['title'])} [{job_id}].txt"
            output = args.out / filename
            previous = read_json(args.out / ".state" / f"{job_id}.json")
            atomic_text(output, render_transcript(info, result, args.model, url, args.preview_seconds))
            atomic_json(args.out / ".state" / f"{job_id}.json", {"url": url, "filename": filename, "fingerprint": fingerprint, "finished": datetime.now(timezone.utc).isoformat()})
            old_name = previous.get("filename")
            if old_name and old_name != filename and Path(old_name).name == old_name:
                (args.out / old_name).unlink(missing_ok=True)
            if not args.keep_audio:
                (work / "audio.f32").unlink(missing_ok=True)
            say(f"{job_id}: saved {output.name}")
            return {**record, "status": "completed", "output": str(output), "title": info["title"]}
    except Exception as error:
        message = str(error)
        say(f"{job_id}: {message}")
        return {**record, "status": "cancelled" if STOP.is_set() else "failed", "error": message}


def preflight(args, require_models: bool) -> None:
    missing = []
    for package in (["yt-dlp", "curl-cffi", "psutil", "whisperx", "torch", "torchaudio", "transformers", "pyannote.audio"] if require_models else ["yt-dlp", "curl-cffi", "psutil"]):
        try:
            installed = version(package)
            if args.check:
                say(f"{package}: {installed}")
        except PackageNotFoundError:
            missing.append(package)
    if missing:
        raise JobError("Missing dependencies: " + ", ".join(missing) + ". Run setup.ps1.")
    if args.ffmpeg_location:
        os.environ["PATH"] = str(args.ffmpeg_location) + os.pathsep + os.environ.get("PATH", "")
    args.ffmpeg = shutil.which("ffmpeg")
    if not args.ffmpeg or not shutil.which("ffprobe"):
        raise JobError("FFmpeg and ffprobe are required. Install with: winget install --id Gyan.FFmpeg -e; then reopen PowerShell.")
    if args.cookies and not args.cookies.is_file():
        raise JobError(f"Cookies file does not exist: {args.cookies}")
    if not shutil.which("deno") and not shutil.which("node"):
        say("YouTube needs Node.js 22+ or Deno 2.3+; Rumble can work without it.")
    if require_models:
        from huggingface_hub import get_token, hf_hub_download
        if not get_token():
            raise JobError("Hugging Face login required for speaker detection.\n1. Accept access at https://huggingface.co/" + MODEL_REPO + "\n2. Run: .\\.venv\\Scripts\\hf.exe auth login\n3. Run: .\\run.ps1 --check\nUse a read token; do not paste it into jobs.txt. See guide.md.")
        try:
            hf_hub_download(MODEL_REPO, "config.yaml", token=get_token(), cache_dir=str(args.model_dir / "diarization"))
        except Exception as error:
            raise JobError("Cannot access the speaker model. Accept its conditions at https://huggingface.co/" + MODEL_REPO + " and check your token/network. " + type(error).__name__) from error
        if args.language == "ro":
            try:
                hf_hub_download(ROMANIAN_ALIGNMENT_MODEL, "config.json", cache_dir=str(args.model_dir / "alignment"))
            except Exception as error:
                raise JobError("Cannot access the Romanian alignment model at https://huggingface.co/" + ROMANIAN_ALIGNMENT_MODEL + ". Check your network and retry. " + type(error).__name__) from error
            say(f"Romanian alignment: {ROMANIAN_ALIGNMENT_MODEL}; weights are downloaded on first use")
        # Keep native/ML imports out of the queue process to conserve memory.
        child([sys.executable, "-c", "import torch; import whisperx.asr; import whisperx.alignment; from whisperx.diarize import DiarizationPipeline; " + ("assert torch.cuda.is_available(), 'CUDA is unavailable in this PyTorch installation'" if args.device == "cuda" else "print('WhisperX imports OK')")], args.log_dir / "runtime-check.log", "WhisperX runtime check")


def main(argv: list[str] | None = None) -> int:
    STOP.clear()
    p = parser()
    args = p.parse_args(argv)
    if args.min_speakers and args.max_speakers and args.min_speakers > args.max_speakers:
        p.error("--min-speakers cannot exceed --max-speakers")
    if args.speakers and args.max_speakers:
        p.error("Use --speakers alone, or --min-speakers/--max-speakers")
    if args.retries < 0:
        p.error("--retries must be 0 or greater")
    if args.inference_workers and args.inference_workers > args.workers:
        p.error("--inference-workers cannot exceed --workers")
    if args.device == "cpu" and args.compute_type in {"float16", "int8_float16"}:
        p.error("Use int8 or float32 on CPU")
    if args.model.endswith(".en") and args.language != "en":
        p.error("Automatic multilingual detection and Romanian require a model without .en, such as base or small. For English-only .en models, explicitly set --language en")
    for name in ("jobs", "out", "cache_dir", "model_dir", "log_dir", "cookies", "ffmpeg_location", "env_file"):
        if getattr(args, name):
            setattr(args, name, getattr(args, name).expanduser().resolve())
    args.compute_type = args.compute_type or ("int8" if args.device == "cpu" else "float16")
    if args.preview_seconds:
        args.out = args.out / "preview"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    try:
        jobs = [] if args.check else read_jobs(args.jobs, args.url)
        if args.limit:
            jobs = jobs[:args.limit]
        if args.dry_run:
            for number, url in enumerate(jobs, 1):
                print(f"{number:3}. {url}")
            say(f"{len(jobs)} unique video(s); workers={args.workers}; language={args.language or 'auto'}; output={args.out}")
            return 0
        if not args.download_only:
            load_credentials(args.env_file)
        import psutil
        ram_gb = psutil.virtual_memory().total / 1024**3
        args.inference_workers = args.inference_workers or (1 if ram_gb < 12 or args.device == "cuda" else min(args.workers, max(1, int(ram_gb // 8))))
        args.threads = args.threads or max(1, (os.cpu_count() or 2) // args.inference_workers)
        say(f"RAM: {ram_gb:.1f} GB; job workers: {args.workers}; model slots: {args.inference_workers}; {args.device}/{args.compute_type}; model: {args.model}")
        say(f"Language: {args.language or 'automatic per 30-second speech window (supports mixed-language videos)'}")
        if ram_gb < 8:
            say("Low-memory PC: base/tiny, batch size 1, and one model slot are recommended. Long podcasts can still use substantial memory and take hours.")
        preflight(args, require_models=not args.download_only)
        if args.check:
            say("Checks passed. Models are downloaded when first used; full transcription has not been tested by this check.")
            return 0
        args.out.mkdir(parents=True, exist_ok=True)
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        args.model_dir.mkdir(parents=True, exist_ok=True)
        args.log_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = hashlib.sha256(json.dumps({**inference_settings(args), "engine_version": ENGINE_VERSION, "whisperx": version("whisperx") if not args.download_only else "download-only"}, sort_keys=True).encode()).hexdigest()[:16]
        semaphore = threading.Semaphore(args.inference_workers)
        results = []
        report = args.log_dir / f"batch-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}.json"
        started = datetime.now(timezone.utc).isoformat()
        atomic_json(report, {"started": started, "status": "running", "jobs": jobs, "results": results})
        say(f"Live batch summary: {report}")

        def interrupt(signum, frame):
            if not STOP.is_set():
                say("Stopping workers and saving the batch summary. Finished stages can be reused on the next run.")
            STOP.set()

        previous_signal = signal.signal(signal.SIGINT, interrupt)
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(run_job, url, args, semaphore, fingerprint): url for url in jobs}
                for future in as_completed(futures):
                    results.append(future.result())
                    atomic_json(report, {"started": started, "status": "running", "jobs": jobs, "results": results})
        finally:
            signal.signal(signal.SIGINT, previous_signal)
        atomic_json(report, {"started": started, "status": "finished", "finished": datetime.now(timezone.utc).isoformat(), "results": results})
        failures = [result for result in results if result["status"] in {"failed", "cancelled"}]
        if failures:
            retry_path = report.with_name(report.stem + "-retry.txt")
            atomic_text(retry_path, "\n".join(result["url"] for result in failures) + "\n")
            say(f"Retry list: {retry_path}")
        counts = {status: sum(result["status"] == status for result in results) for status in ("completed", "skipped", "downloaded", "failed", "cancelled")}
        say("Batch finished: " + ", ".join(f"{number} {status}" for status, number in counts.items() if number))
        say(f"Summary: {report}")
        return 130 if STOP.is_set() else (1 if failures else 0)
    except (JobError, OSError, ImportError) as error:
        say(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
