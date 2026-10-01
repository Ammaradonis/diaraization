# Local video diarization tool

This tool downloads YouTube and Rumble recordings, transcribes them with **WhisperX**, aligns the words, detects speakers with **pyannote**, and writes a UTF-8 `.txt` file for each video. The first line is the actual video title. Each speech turn has start/end timestamps and a speaker label.

## Main command

Your existing `env.txt` is loaded automatically. Open PowerShell and paste:

```powershell
cd "C:\Users\FingerWeg\CascadeProjects\diaraization"
.\run.ps1 --workers 3
```

This reads `jobs.txt` and writes to `data\out`. The explicit equivalent is:

```powershell
.\run.ps1 --jobs ".\jobs.txt" --workers 3 --out ".\data\out"
```

**Automatic language switching is enabled by default**, including English and Romanian within the same video. Nearby speech fragments are grouped into windows of up to 30 seconds for language detection and transcription, then sent to the matching word aligner. Speaker detection still covers the whole video so a language switch does not restart speaker numbering. No language flag is needed for mixed recordings.

Default paths are anchored to the tool's folder. Explicit relative paths are relative to your current terminal folder. Paths with spaces must be quoted.

**On this PC:** about 4 GB RAM, an Intel i3, and a GeForce 830M were detected. Defaults are `base`, CPU, `int8`, batch size 1. Three video jobs can download/wait concurrently, while **one model job runs at a time** to limit memory use. Three simultaneous WhisperX model jobs are available with `--inference-workers 3`, but are unsuitable for this PC's memory. Multi-hour podcasts may take many hours and can still exceed available memory. No real-time speed is promised.

## One-time installation and model access

The installer creates a private Python 3.12 environment in `.venv`, with its Python runtime in `.tools`. It installs CPU PyTorch and pinned WhisperX dependencies. Your system Python is not changed.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

Internet access and several GB of disk space are needed. The first transcription also downloads models. Re-running setup safely retries an interrupted installation.

Your `env.txt` already contains a `HUGGING_FACE_TOKEN` entry, which the tool recognizes. Its value is passed to model processes in their environment, never written to job settings or printed. `env.txt` is excluded by `.gitignore`. No separate CLI login is needed when that token has model access. Credential precedence is: an existing `HF_TOKEN` environment variable, then `env.txt`, then a saved Hugging Face login. Use `--env-file PATH` to choose another file.

Speaker detection requires access to the [pyannote community model](https://huggingface.co/pyannote/speaker-diarization-community-1). On a fresh setup, or if the existing token lacks access:

1. Sign in to Hugging Face and accept the conditions on that model page.
2. Create a **read** token in [Hugging Face token settings](https://huggingface.co/settings/tokens). A fine-grained token must permit reading the gated model.
3. Update the token in `env.txt`. Alternatively, rename `env.txt` to `env.txt.bak` and use the local login command with its hidden prompt:

   ```powershell
   .\.venv\Scripts\hf.exe auth login
   ```

   You do not need to add the token as a Git credential. Do not put it in `jobs.txt` or commit it to this project. An existing `HF_TOKEN` environment variable is also supported and takes precedence over the saved login.

4. Check the installation:

   ```powershell
   .\run.ps1 --check
   ```

The check verifies packages, FFmpeg, model configuration access, and WhisperX imports. It does not transcribe a video or download every model weight. Your account must complete the model access step; the tool cannot accept those conditions for you.

FFmpeg and a suitable Node.js runtime were already present on this PC. If setting this up elsewhere, install missing tools and reopen PowerShell:

```powershell
winget install --id Gyan.FFmpeg -e
winget install --id OpenJS.NodeJS.LTS -e
```

YouTube's downloader uses Node.js 22+ or Deno 2.3+. Setup includes yt-dlp's JavaScript solver package. Rumble uses its own yt-dlp extractor with Chrome request impersonation through `curl-cffi`, enabled automatically for Rumble URLs. See the official [yt-dlp JavaScript setup](https://github.com/yt-dlp/yt-dlp/wiki/EJS) and [browser request support](https://github.com/yt-dlp/yt-dlp#impersonation).

If PowerShell blocks scripts, use the following invocation; it changes policy for that process only:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 --workers 3
```

## jobs.txt

Put one video URL per line. Blank lines and lines starting with `#` are ignored. YouTube watch, youtu.be, shorts, and recorded live URLs are accepted. Rumble video pages and embed URLs are accepted. Duplicate YouTube links with different sharing parameters are processed once.

```text
# A YouTube interview
https://youtu.be/3V33MIIcXjM

# A Rumble interview
https://rumble.com/v4cw3gr-the-king-of-toxic-masculinity-a-conversation-with-cobra-tate-in-warsaw-pola.html
```

Use individual videos. Channels, playlists, current live streams, and upcoming streams are excluded. A recorded YouTube `/live/` URL works after its recording becomes downloadable. Restricted, removed, or region-blocked videos may fail; other queued jobs continue.

## Examples

Validate the queue without network access or models:

```powershell
.\run.ps1 --dry-run
```

Start with a short preview of the first video:

```powershell
.\run.ps1 --limit 1 --preview-seconds 120
```

Previews go into `data\out\preview`, are clearly labeled, and never mark the full video as complete. Speaker detection on a preview only sees the speakers who appear during that preview.

### Romanian recordings

For mixed Romanian/English recordings, use the main command unchanged:

```powershell
.\run.ps1 --workers 3
```

Force Romanian only when every recording is entirely Romanian:

```powershell
.\run.ps1 --workers 3 --language ro
```

`--language Romanian`, `--language romana`, and `--language "română"` also select Romanian. Use a multilingual Whisper model (`base` is the default); English-only `.en` models are rejected with Romanian. Text stays in Romanian, including `ă â î ș ț`, with the same speaker labels and timestamps.

Romanian uses [`gigant/romanian-wav2vec2`](https://huggingface.co/gigant/romanian-wav2vec2), the [WhisperX default Romanian alignment model](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py). Its weights download on first use and are reused from `data\models\alignment`. Check model configuration access or run a short preview:

```powershell
.\run.ps1 --check --language ro
.\run.ps1 --language ro --limit 1 --preview-seconds 120
```

Omitting `--language` or using `--language auto` enables detection within each recording, as well as across the queue. Silero finds speech regions and groups nearby fragments into windows of up to 30 seconds. Each window gets a fresh language detection; trailing fragments shorter than three seconds reuse the previous window's language. The first detected language is never locked for the whole video. Grouping avoids running padded Whisper inference separately on every short pause. English regions use English alignment and Romanian regions use Romanian alignment. Aligners load one at a time to limit memory use, and transcript segments return to chronological order afterward. Mixed transcripts list their detected languages in the header.

An explicit `--language ro` or `--language en` disables switching and applies that language to every recording. Very short utterances, accents, overlapping voices, and switches inside one speech region can still be misrecognized. Detection operates on speech windows; a language switch inside a window can be missed. Detection adds processing time. If a detected language has no default aligner or its model cannot load, original text and speech timestamps are retained. Alignment-model failures are noted in the output. In mixed recordings, a language occupying less than both 30 seconds and 1% of the recording also retains speech timestamps, avoiding large model downloads for isolated misdetections; the output explains this choice.

Process a specific Rumble video:

```powershell
.\run.ps1 --url "https://rumble.com/v4cw3gr-the-king-of-toxic-masculinity-a-conversation-with-cobra-tate-in-warsaw-pola.html" --language en
```

For a batch where every recording really has at least five speakers:

```powershell
.\run.ps1 --workers 3 --min-speakers 5
```

For a known seven-person discussion:

```powershell
.\run.ps1 --url "YOUR_VIDEO_URL" --speakers 7
```

Use a larger model on a computer with sufficient memory:

```powershell
.\run.ps1 --model small --workers 3 --inference-workers 1
```

For **three concurrent model jobs on a sufficiently powerful PC**:

```powershell
.\run.ps1 --workers 3 --inference-workers 3 --model small
```

Cache audio before logging into Hugging Face:

```powershell
.\run.ps1 --download-only --limit 1
```

Supply an exported Netscape cookies file when needed for your own account's access:

```powershell
.\run.ps1 --cookies "C:\private\cookies.txt"
```

The downloader uses a temporary copy for each job and removes that copy afterward. Cookies do not guarantee that a site's access restrictions or bot checks will allow the download.

## Flags

Use `.\run.ps1 --help` for built-in help.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--jobs PATH` | `jobs.txt` | Read the batch file. |
| `--url URL` | None | Process a URL instead of the file; repeat for multiple URLs. |
| `--out PATH` | `data\out` | Transcript output folder. |
| `--workers N` | `3` | Maximum active video jobs, including downloads and jobs awaiting a model slot. |
| `--inference-workers N` | Automatic | Maximum simultaneous model jobs. Automatic is one below 12 GB RAM or on CUDA; otherwise up to one per 8 GB, capped by `--workers`. This is a heuristic, not a memory guarantee. |
| `--model NAME` | `base` | Whisper model, such as `tiny`, `base`, `small`, `medium`, or `large-v3`. English-only `.en` variants require explicit `--language en`. |
| `--device cpu/cuda` | `cpu` | Device used by WhisperX. Setup installs CPU PyTorch; CUDA requires a compatible GPU, PyTorch build, and CUDA libraries. |
| `--compute-type TYPE` | CPU: `int8`; CUDA: `float16` | Also supports `float32` and CUDA `int8_float16`. |
| `--batch-size N` | `1` | Transcription and speaker detection batch size; higher values need more memory. |
| `--threads N` | Automatic | CPU threads per model process, divided among model slots. |
| `--language CODE` | Auto-detect per 30-second speech window | Omit for mixed recordings. `ro`, `en`, etc. force one language throughout; `auto` restores detection within each video. Romanian name aliases are accepted. |
| `--speakers N` | Automatic | Exact speaker count. Cannot combine with minimum/maximum. |
| `--min-speakers N` | Unset | Lower bound for speaker detection. |
| `--max-speakers N` | Unset | Upper bound; the tool does not impose a five-speaker limit. |
| `--chunk-minutes N` | `10` | ASR/alignment audio chunks. Does not split global speaker detection. |
| `--limit N` | All | First N unique input URLs, including any already completed URLs. |
| `--preview-seconds N` | Full video | Process only the start and save under `out\preview`. |
| `--cookies PATH` | None | Netscape cookies file for the downloader. |
| `--env-file PATH` | `env.txt` | Optional local token file. Recognizes `HF_TOKEN`, `HUGGING_FACE_TOKEN`, `HUGGINGFACE_TOKEN`, or `HUGGINGFACE_HUB_TOKEN`; never prints their values. |
| `--ffmpeg-location PATH` | System PATH | Directory containing both FFmpeg and ffprobe. |
| `--stall-timeout-minutes N` | `30` | Stop inference after N minutes without log activity; retain completed checkpoints. Increase for exceptionally slow model loading or clustering. |
| `--cache-dir PATH` | `data\cache` | Downloaded audio and per-stage checkpoints. |
| `--model-dir PATH` | `data\models` | Whisper, alignment, diarization, and Torch model caches. |
| `--log-dir PATH` | `data\logs` | Downloader, model, and summary logs. |
| `--retries N` | `3` | Network/extractor/fragment retries; `0` disables retries. Model failures are not automatically rerun. |
| `--keep-audio` | Off | Retain decoded audio after successful transcription. |
| `--force` | Off | Recompute all inference stages and replace the transcript for matching inputs/settings. |
| `--download-only` | Off | Download/decode audio without models or Hugging Face login. |
| `--dry-run` | Off | Validate/list inputs, then exit. |
| `--check` | Off | Verify dependencies and model access, then exit. Combine with `--download-only` to check download prerequisites only. |
| `--help` / `-h` | Off | Print all options. |

## Output and restarting

A filename looks like `Actual Video Title [54554ac23934].txt`. The title is sanitized only for Windows filenames; the headline keeps the supplied video title. The stable URL hash prevents collisions between different videos with identical titles.

Example format (illustrative):

```text
Actual Video Title

Source: https://www.youtube.com/watch?v=...
Language: en
Speakers detected: 6
Transcription: WhisperX / base
Speaker numbers identify voices within this video; they are not verified names.

[00:00:01.120 --> 00:00:04.530] SPEAKER 01: Welcome to the discussion.
[00:00:04.610 --> 00:00:07.240] SPEAKER 02: Thank you for having me.
```

Speaker labels follow first appearance and restart for each video. Word-level speaker changes split the text into separate turns. Unassigned speech is labeled `SPEAKER UNKNOWN`. Alignment failures retain the original text with coarser timestamps.

Rerun the same command to resume. A matching successful transcript is skipped. Completed stages are checkpointed separately. Transcription also saves each minute of audio, and alignment saves each completed language/chunk, so an interrupted stage can resume. Diarization still requires a complete whole-video pass. If diarization finished but text export was interrupted, the saved result can be exported without audio or another model pass. Model/language/speaker/preview setting changes invalidate previous completion records. Output text is replaced atomically only after the full pipeline succeeds. There is no completed transcript for a failed download/model stage.

The October 1 recovery fixes preserve existing compatible checkpoints and completed transcripts. Rerun the same command without `--force` to reuse the last run's work. Use `--force` only if you deliberately want to recompute transcription and speaker detection.

Press **Ctrl+C once** to stop active subprocesses and save a summary. The next run can reuse complete checkpoints and saved transcription/alignment chunks. Logs retain previous attempts and include start/end times. The live batch summary is written at startup and updated after every job. Console updates show elapsed time and the latest inference progress. A model process with no log activity for 30 minutes is terminated with a resumable failure instead of waiting indefinitely. Do not run the same batch in several terminals to increase speed; per-video locks reject simultaneous writers with the same cache directory. Use `--workers` instead.

Files and folders:

| Location | Contents |
| --- | --- |
| `data\out\*.txt` | Finished full-video transcripts. |
| `data\out\.state` | Small completion records; retain them to skip finished jobs. |
| `data\out\preview` | Preview transcripts and separate completion records. |
| `data\cache` | Decoded audio and JSON checkpoints. Audio is removed after success unless `--keep-audio` is set; failed-job audio is retained. |
| `data\models` | Reusable downloaded models. NLTK may additionally use its normal user data directory. |
| `data\logs\VIDEO_ID.STAGE.log` | Detailed stage logs, overwritten on the next attempt for that video/stage. |
| `data\logs\batch-*.json` | Per-run success/failure summaries. |
| `data\logs\batch-*-retry.txt` | Only failed/cancelled URLs; created when needed. Pass this file to `--jobs`. |

Exit codes: `0` success/all skipped/downloaded; `1` at least one failed video; `2` input/setup error; `130` cancelled. PowerShell exposes the code as `$LASTEXITCODE`.

## Best practices and troubleshooting

- **Try a preview first.** Review names, timestamps, and speaker changes before starting all 54 existing jobs. Short previews cannot validate every speaker in a long video.
- **Leave speaker counts automatic when unsure.** Set `--min-speakers 5` only if each selected recording has at least five audible speakers. Forcing too many speakers can split one voice into multiple labels. Exact counts include hosts, guests, narration, and voices in inserted clips.
- **Leave language automatic for mixed recordings.** The default detects language in grouped speech windows and skips silence. Set `--language en` or `--language ro` only to force a single language throughout. Review short utterances and rapid language switches against the audio.
- **For this 4 GB PC, start small.** Keep one model slot, `--batch-size 1`, and `base` or `tiny`. Close memory-heavy apps, keep Windows' pagefile enabled, and leave free disk space. A smaller Whisper model does not reduce all of pyannote's memory needs. If whole-video diarization still runs out of memory, use a machine with more RAM.
- **Preserve full-video speaker identities.** ASR/alignment use bounded audio chunks; pyannote clusters speakers across the whole recording. Cutting long recordings into unrelated jobs will restart speaker labels in each part.
- **Expect corrections.** Automatic labels are not real names. Crosstalk, music, similar voices, weak microphones, and inserted clips can cause wrong words or speaker assignments. Review important passages against the original recording. [WhisperX documents these limitations](https://github.com/m-bain/whisperX#limitations-).
- **Allow disk space for audio and models.** Decoded float32 mono audio uses roughly 230 MB per hour, plus temporary downloaded media. A Rumble source without a separate audio stream may require downloading a video container first; the tool prefers up to 480p in that case and discards that container after decoding.
- **Model access errors:** accept the model conditions using the account that owns your token and update `env.txt` if needed. If using a saved login instead, rename `env.txt` to `env.txt.bak` and run `hf.exe auth login` again. A stale `HF_TOKEN` environment variable overrides both. Keep credentials out of logs and source files, including renamed credential backups.
- **Download errors:** read the video's `.download.log`. Confirm that the URL still plays, use cookies when appropriate, and update the downloader if a site has changed:

  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -UpdateDownloader
  ```

- **Model errors:** read `.transcribe.log`, `.align.log`, or `.diarize.log`. For initial download errors, fix connectivity and retry; downloaded model files are cached. Do not independently upgrade Torch/Transformers without checking WhisperX compatibility.
- **TorchCodec warning on Windows:** this tool decodes audio with FFmpeg and supplies arrays directly to pyannote. A warning about unavailable TorchCodec shared libraries can be harmless for this path; an actual import failure or a nonzero stage exit is a failure and is recorded in the log.

The implementation follows [WhisperX's Python pipeline](https://github.com/m-bain/whisperX#python-usage-), uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) for downloads, and runs the [community diarization model](https://huggingface.co/pyannote/speaker-diarization-community-1) locally. No paid transcription API is used. Internet is needed for video/model downloads and initial access checks; downloaded audio is processed on this computer.

## Verified on this PC

On 29 September 2026, installation compatibility and WhisperX runtime/model access checks passed, along with 17 offline tests. A real three-worker queue completed 60-second previews of one YouTube video and both Rumble URLs from `jobs.txt`, including download, transcription, word alignment, speaker detection, and TXT output.

| Preview | Detected speakers |
| --- | --- |
| Adin Ross Introduces Adam22 & Lena The Plug To ANDREW TATE | 3 |
| The KING of Toxic Masculinity - a Conversation with Cobra Tate in Warsaw Poland | 2 |
| M2 discuss their recent excursion to Eastern Europe and visiting their friends Cobra & Tristan Tate | 4 |

The transcripts are in `data\out\preview`. The successful batch report is `data\logs\batch-20260929-193157-14428.json`. These counts describe the previews and are not verified counts for the complete videos. The full 54-video queue has not been processed or benchmarked. Automatic words and labels still need review.

The base Whisper model, English alignment model, Silero voice detector, and community speaker model are cached locally. Speaker batch sizes were reduced to follow `--batch-size` after the first preview showed heavy memory pressure with pyannote's larger default. The second Rumble preview and the YouTube preview completed with the final batch-size setting of 1.

## Development checks

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_tool
```

The offline regression suite covers input parsing, seven-speaker formatting, word-level speaker changes, checkpoint recovery, completed-job skipping, and file locks. Language checks cover the main command's automatic mode, repeated English/Romanian switches within a video, alignment routing and chronological timestamps across chunks, silence, unsupported aligner fallback, forced-language overrides, aliases, and UTF-8 diacritics. A main-command regression check verifies that legacy checkpoints are recomputed and new results subsequently resume normally. Worker routing tests mock the speech models; they do not measure speech recognition or diarization accuracy. Mixed Romanian/English audio has not yet been evaluated end to end.

A real 12-second cached English clip passed the previous September 29 automatic transcription and alignment path: five speech regions were detected independently and became six aligned sentences. Those historical artifacts are in `data\cache\language-smoke-20260929`.

The October 1 recovery fixes passed 32 offline regression tests, including alignment-download failure, chunk recovery, sparse language detections, checkpoint export without audio, and stalled-child termination. A real 60-second smoke check was stopped during dependency imports on the memory-constrained machine; end-to-end completion and performance of this update have not been verified. See `data\logs\recovery-20261001.md` for the evidence, recovered output, and remaining work.
