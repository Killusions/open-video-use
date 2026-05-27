"""Transcribe a video with Whisper via OpenAI-compatible API.

Extracts mono 16kHz mp3 audio via ffmpeg (configurable bitrate, default 32k),
uploads to the Whisper API with word-level timestamps, writes the full response
to <edit_dir>/transcripts/<video_stem>.json.

Auto-splitting: if the extracted mp3 exceeds WHISPER_MAX_SIZE (default ~4.6 MB)
or WHISPER_MAX_DURATION, the audio is automatically split into overlapping
segments, each transcribed separately, then merged into a single unified
transcript with correct absolute timestamps.

Cached: if the output file already exists, the upload is skipped.

Supports any OpenAI-compatible API endpoint (OpenAI, vLLM, faster-whisper-server,
whisper.cpp, etc.) via --base-url or OPENAI_BASE_URL env var.

Environment variables / .env keys:
    OPENAI_API_KEY       — required
    OPENAI_BASE_URL      — API base URL (default: https://api.openai.com/v1)
    WHISPER_MODEL        — model name (default: whisper-large-v3-turbo)
    WHISPER_MAX_SIZE     — max upload size in bytes (default: 4800000, ~4.6 MB)
    WHISPER_MAX_DURATION — max segment duration in seconds (default: none)
    WHISPER_OVERLAP      — overlap between segments in seconds (default: 30)
    WHISPER_BITRATE      — mp3 bitrate for extraction (default: 32k)

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --model whisper-large-v3-turbo
    python helpers/transcribe.py <video_path> --base-url http://localhost:8000/v1
    python helpers/transcribe.py <video_path> --bitrate 64k
    python helpers/transcribe.py <video_path> --max-size 4800000
    WHISPER_MAX_SIZE=4800000 python helpers/transcribe.py <video_path>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "whisper-large-v3-turbo"
DEFAULT_MAX_SIZE = 4_800_000
DEFAULT_OVERLAP = 30.0
DEFAULT_BITRATE = "32k"


def _load_env_vars() -> dict[str, str]:
    """Load key=value pairs from .env files.

    Priority: .env at skill root > .env in cwd. First-found wins per key.
    """
    env_vars: dict[str, str] = {}
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and v:
                    env_vars.setdefault(k, v)
    return env_vars


def _env(key: str, default: str, env_vars: dict[str, str]) -> str:
    """Get a config value: .env file > environment > default."""
    return env_vars.get(key) or os.environ.get(key, "") or default


def load_config() -> tuple[str, str, str]:
    """Return (api_key, base_url, model) from .env files or environment.

    Priority: .env at repo root > .env in cwd > environment variables.
    """
    env_vars = _load_env_vars()

    api_key = _env("OPENAI_API_KEY", "", env_vars)
    if not api_key:
        sys.exit("OPENAI_API_KEY not found in .env or environment")

    base_url = _env("OPENAI_BASE_URL", DEFAULT_BASE_URL, env_vars)
    model = _env("WHISPER_MODEL", DEFAULT_MODEL, env_vars)

    return api_key, base_url, model


def load_split_config() -> tuple[int, float | None, float, str]:
    """Return (max_size, max_duration, overlap, bitrate) from .env or environment."""
    env_vars = _load_env_vars()
    max_size = int(_env("WHISPER_MAX_SIZE", str(DEFAULT_MAX_SIZE), env_vars))
    max_dur_raw = _env("WHISPER_MAX_DURATION", "", env_vars)
    max_duration = float(max_dur_raw) if max_dur_raw else None
    overlap = float(_env("WHISPER_OVERLAP", str(DEFAULT_OVERLAP), env_vars))
    bitrate = _env("WHISPER_BITRATE", DEFAULT_BITRATE, env_vars)
    return max_size, max_duration, overlap, bitrate


def extract_audio(video_path: Path, dest: Path, bitrate: str = DEFAULT_BITRATE) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-b:a", bitrate,
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _get_audio_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def call_whisper(
    audio_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    language: str | None = None,
) -> dict:
    url = f"{base_url.rstrip('/')}/audio/transcriptions"

    data: list[tuple[str, str]] = [
        ("model", model),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "word"),
    ]
    if language:
        data.append(("language", language))

    with open(audio_path, "rb") as f:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (audio_path.name, f, "audio/mpeg")},
            data=data,
            timeout=1800,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Whisper API returned {resp.status_code}: {resp.text[:500]}")

    return resp.json()


def _split_audio(
    full_mp3: Path,
    tmp_dir: Path,
    chunk_dur: float,
    overlap: float,
    total_dur: float,
) -> list[tuple[Path, float]]:
    """Split mp3 into overlapping segments. Returns [(path, start_time), ...]."""
    segments: list[tuple[Path, float]] = []
    start = 0.0
    i = 0
    while start < total_dur:
        seg_path = tmp_dir / f"segment_{i:03d}.mp3"
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-t", str(chunk_dur),
            "-i", str(full_mp3), "-c", "copy", str(seg_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if seg_path.stat().st_size == 0:
            break
        segments.append((seg_path, start))
        next_start = start + chunk_dur - overlap
        if next_start + overlap >= total_dur:
            break  # current segment already covers the rest
        start = next_start
        i += 1
    return segments


def _compute_drift_factors(
    results: list[tuple[dict, float]],
) -> list[float]:
    """Compute per-segment linear time-stretch factors using overlap text matching.

    Whisper segment-level timestamps drift linearly (~1s per 100s of audio).
    For each consecutive pair we find matching text in the overlap region.
    Segment i+1 is near its start so its timestamps are accurate; segment i
    is near its end where drift is worst.  The ratio gives us the correction
    factor for segment i: corrected_local_t = local_t * factor.
    """
    n = len(results)
    factors = [1.0] * n

    for i in range(n - 1):
        payload_i, start_i = results[i]
        payload_j, start_j = results[i + 1]

        segs_i = payload_i.get("segments", [])
        segs_j = payload_j.get("segments", [])
        if not segs_i or not segs_j:
            continue

        # Try to match text from the end of seg_i with the start of seg_j
        # seg_j is fresh so its early timestamps are accurate
        best_whisper_t = None
        best_actual_t = None
        for si in reversed(segs_i):
            si_text = si.get("text", "").strip().lower()
            if len(si_text) < 10:
                continue
            needle = si_text[:30]
            for sj in segs_j:
                sj_text = sj.get("text", "").strip().lower()
                if needle in sj_text or sj_text[:30] in si_text:
                    # Match: seg_j says this content is at local time sj["start"],
                    # which corresponds to absolute time (start_j + sj["start"]).
                    # seg_i says the same content is at local time si["start"].
                    # The true local time in seg_i's frame is:
                    #   start_j + sj["start"] - start_i
                    best_whisper_t = si["start"]
                    best_actual_t = start_j + sj["start"] - start_i
                    break
            if best_whisper_t is not None:
                break

        if best_whisper_t and best_whisper_t > 0:
            factors[i] = best_actual_t / best_whisper_t

    return factors


def _correct_time(local_t: float, factor: float, seg_start: float) -> float:
    """Apply linear drift correction and offset to a local timestamp."""
    return round(local_t * factor + seg_start, 3)


def _merge_transcripts(
    results: list[tuple[dict, float]],
    chunk_dur: float,
    overlap: float,
    total_dur: float,
) -> dict:
    """Merge overlapping transcript segments into a single unified transcript.

    Uses overlap regions to measure and correct Whisper timestamp drift
    (which accumulates linearly within each segment).
    """
    if len(results) == 1:
        return results[0][0]

    factors = _compute_drift_factors(results)

    merged_words: list[dict] = []
    merged_segments: list[dict] = []
    n = len(results)

    for i, (payload, seg_start) in enumerate(results):
        factor = factors[i]

        # Local time bounds: keep the middle portion, discard overlap edges
        left_cut = 0.0 if i == 0 else overlap / 2
        right_cut = float("inf") if i == n - 1 else chunk_dur - overlap / 2

        if payload.get("words"):
            for w in payload["words"]:
                t = w.get("start", 0.0)
                if t < left_cut or t >= right_cut:
                    continue
                merged_words.append({
                    **w,
                    "start": _correct_time(t, factor, seg_start),
                    "end": _correct_time(w.get("end", t), factor, seg_start),
                })

        if payload.get("segments"):
            for s in payload["segments"]:
                t = s.get("start", 0.0)
                if t < left_cut or t >= right_cut:
                    continue
                merged_segments.append({
                    **s,
                    "start": _correct_time(s["start"], factor, seg_start),
                    "end": _correct_time(s["end"], factor, seg_start),
                })

    # Re-index segment IDs
    for idx, s in enumerate(merged_segments):
        s["id"] = idx

    # Reconstruct full text
    if merged_segments:
        text = " ".join(s.get("text", "").strip() for s in merged_segments)
    elif merged_words:
        text = " ".join(w.get("word", "").strip() for w in merged_words)
    else:
        text = " ".join(r[0].get("text", "") for r in results)

    merged: dict = {"text": text, "duration": total_dur}
    # Preserve top-level metadata from first segment
    first = results[0][0]
    for key in ("task", "language"):
        if key in first:
            merged[key] = first[key]
    if merged_segments:
        merged["segments"] = merged_segments
    if merged_words:
        merged["words"] = merged_words

    return merged


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    verbose: bool = True,
    max_size: int | None = None,
    max_duration: float | None = None,
    overlap: float | None = None,
    bitrate: str | None = None,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    Auto-splits long files into overlapping segments when they exceed max_size
    or max_duration, then merges the results.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    # Load split config from env for any unset params
    _max_size, _max_dur, _overlap, _bitrate = load_split_config()
    if max_size is None:
        max_size = _max_size
    if max_duration is None:
        max_duration = _max_dur
    if overlap is None:
        overlap = _overlap
    if bitrate is None:
        bitrate = _bitrate

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        audio = tmp_dir / f"{video.stem}.mp3"
        extract_audio(video, audio, bitrate=bitrate)
        file_size = audio.stat().st_size
        size_mb = file_size / (1024 * 1024)
        total_dur = _get_audio_duration(audio)

        needs_split = file_size > max_size or (max_duration is not None and total_dur > max_duration)

        if not needs_split:
            if verbose:
                print(f"  uploading {video.stem}.mp3 ({size_mb:.1f} MB)", flush=True)
            payload = call_whisper(audio, api_key, base_url, model, language)
        else:
            # Chunk duration from observed bitrate (actual bytes/sec, not nominal)
            observed_bps = file_size / total_dur
            chunk_dur = max_size / observed_bps
            if max_duration is not None:
                chunk_dur = min(chunk_dur, max_duration)

            # Sanity: overlap must be less than chunk duration
            if overlap >= chunk_dur:
                overlap = max(chunk_dur / 4, 1.0)

            step = chunk_dur - overlap
            n_segments = 1 + int((total_dur - chunk_dur + step - 0.001) / step) if total_dur > chunk_dur else 1

            if verbose:
                print(
                    f"  file {size_mb:.1f} MB / {total_dur:.0f}s exceeds limits, "
                    f"splitting into ~{n_segments} segments "
                    f"({chunk_dur:.0f}s each, {overlap:.0f}s overlap)",
                    flush=True,
                )

            segments = _split_audio(audio, tmp_dir, chunk_dur, overlap, total_dur)
            results: list[tuple[dict, float]] = []
            for j, (seg_path, seg_start) in enumerate(segments):
                seg_size_mb = seg_path.stat().st_size / (1024 * 1024)
                if verbose:
                    print(
                        f"  uploading segment {j + 1}/{len(segments)} "
                        f"({seg_size_mb:.1f} MB, offset {seg_start:.1f}s)",
                        flush=True,
                    )
                seg_payload = call_whisper(seg_path, api_key, base_url, model, language)
                results.append((seg_payload, seg_start))

            payload = _merge_transcripts(results, chunk_dur, overlap, total_dur)

    out_path.write_text(json.dumps(payload, indent=2))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        if isinstance(payload, dict) and payload.get("words"):
            print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video with Whisper via OpenAI-compatible API")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Whisper model name (default: WHISPER_MODEL env or '{DEFAULT_MODEL}')",
    )
    ap.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=f"OpenAI-compatible API base URL (default: OPENAI_BASE_URL env or '{DEFAULT_BASE_URL}')",
    )
    ap.add_argument(
        "--max-size",
        type=int,
        default=None,
        help=f"Max upload file size in bytes (default: WHISPER_MAX_SIZE env or {DEFAULT_MAX_SIZE})",
    )
    ap.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Max segment duration in seconds (default: WHISPER_MAX_DURATION env or no limit)",
    )
    ap.add_argument(
        "--overlap",
        type=float,
        default=None,
        help=f"Overlap between segments in seconds (default: WHISPER_OVERLAP env or {DEFAULT_OVERLAP})",
    )
    ap.add_argument(
        "--bitrate",
        type=str,
        default=None,
        help=f"MP3 bitrate for extraction (default: WHISPER_BITRATE env or '{DEFAULT_BITRATE}')",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    api_key, base_url, model = load_config()

    # CLI args override env/config
    if args.base_url:
        base_url = args.base_url
    if args.model:
        model = args.model

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        api_key=api_key,
        base_url=base_url,
        model=model,
        language=args.language,
        max_size=args.max_size,
        max_duration=args.max_duration,
        overlap=args.overlap,
        bitrate=args.bitrate,
    )


if __name__ == "__main__":
    main()
