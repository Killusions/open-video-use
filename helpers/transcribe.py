"""Transcribe a video with Whisper via OpenAI-compatible API.

Extracts mono 16kHz audio via ffmpeg, uploads to the Whisper API with
word-level timestamps, writes the full response to
<edit_dir>/transcripts/<video_stem>.json.

Cached: if the output file already exists, the upload is skipped.

Supports any OpenAI-compatible API endpoint (OpenAI, vLLM, faster-whisper-server,
whisper.cpp, etc.) via --base-url or OPENAI_BASE_URL env var.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --model whisper-large-v3-turbo
    python helpers/transcribe.py <video_path> --base-url http://localhost:8000/v1
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


def load_config() -> tuple[str, str, str]:
    """Return (api_key, base_url, model) from .env files or environment.

    Priority: .env at repo root > .env in cwd > environment variables.
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

    api_key = env_vars.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        sys.exit("OPENAI_API_KEY not found in .env or environment")

    base_url = env_vars.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "") or DEFAULT_BASE_URL
    model = env_vars.get("WHISPER_MODEL") or os.environ.get("WHISPER_MODEL", "") or DEFAULT_MODEL

    return api_key, base_url, model


def extract_audio(video_path: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def call_whisper(
    audio_path: Path,
    api_key: str,
    base_url: str,
    model: str,
    language: str | None = None,
) -> dict:
    url = f"{base_url.rstrip('/')}/audio/transcriptions"

    # Use list-of-tuples for form data to handle array-style params
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
            files={"file": (audio_path.name, f, "audio/wav")},
            data=data,
            timeout=1800,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Whisper API returned {resp.status_code}: {resp.text[:500]}")

    return resp.json()


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    verbose: bool = True,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio)
        size_mb = audio.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"  uploading {video.stem}.wav ({size_mb:.1f} MB)", flush=True)
        payload = call_whisper(audio, api_key, base_url, model, language)

    out_path.write_text(json.dumps(payload, indent=2))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        if isinstance(payload, dict) and "words" in payload:
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
    )


if __name__ == "__main__":
    main()
