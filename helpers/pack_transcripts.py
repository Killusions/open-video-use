"""Pack all Whisper transcripts in <edit>/transcripts/ into one readable markdown.

Groups word-level entries into phrase-level lines, breaking on any silence
>= 0.5s between consecutive words. Each phrase gets a [start-end] prefix.
This is the PRIMARY artifact the editor sub-agent reads to pick cuts -- it
fits one hour of takes in a tenth the tokens of raw Whisper JSON and gives
word-boundary precision from text alone.

Output: <edit>/takes_packed.md

Usage:
    python helpers/pack_transcripts.py --edit-dir <edit_dir>
    python helpers/pack_transcripts.py --edit-dir <edit_dir> --silence-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def format_time(seconds: float) -> str:
    """Format a time in seconds as "NNN.NN" with fixed 6-char width for alignment."""
    return f"{seconds:06.2f}"


def format_duration(seconds: float) -> str:
    """Format a duration as "Ms" or "Mm SSs"."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}m {s:04.1f}s"


def group_into_phrases(
    words: list[dict],
    silence_threshold: float = 0.5,
) -> list[dict]:
    """Walk a Whisper word list, break into phrases on silence >= threshold.
    Returns list of {start, end, text}.

    Whisper word entries have: {word, start, end}.
    Gaps are computed directly between consecutive words.
    """
    phrases: list[dict] = []
    current_words: list[dict] = []
    current_start: float | None = None

    def flush() -> None:
        nonlocal current_words, current_start
        if not current_words:
            return
        text_parts = [w.get("word", "").strip() for w in current_words]
        text_parts = [t for t in text_parts if t]
        if not text_parts:
            current_words = []
            current_start = None
            return
        text = " ".join(text_parts)
        text = text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
        end_time = current_words[-1].get("end", current_start or 0.0)
        phrases.append({
            "start": current_start,
            "end": end_time,
            "text": text,
        })
        current_words = []
        current_start = None

    prev_end: float | None = None

    for w in words:
        word_text = w.get("word", "").strip()
        start = w.get("start")
        if start is None or not word_text:
            continue

        # Flush on a long gap from the previous word
        if prev_end is not None and start - prev_end >= silence_threshold:
            flush()

        if current_start is None:
            current_start = start
        current_words.append(w)
        prev_end = w.get("end", start)

    flush()
    return phrases


def pack_one_file(json_path: Path, silence_threshold: float) -> tuple[str, float, list[dict]]:
    """Return (header_name, duration, phrases) for one transcript file."""
    data = json.loads(json_path.read_text())
    words = data.get("words", [])
    phrases = group_into_phrases(words, silence_threshold)
    if phrases:
        duration = phrases[-1]["end"] - phrases[0]["start"]
    else:
        duration = data.get("duration", 0.0)
    return json_path.stem, duration, phrases


def render_markdown(entries: list[tuple[str, float, list[dict]]], silence_threshold: float) -> str:
    lines: list[str] = []
    lines.append("# Packed transcripts")
    lines.append("")
    lines.append(f"Phrase-level, grouped on silences >= {silence_threshold:.1f}s.")
    lines.append("Use `[start-end]` ranges to address cuts in the EDL.")
    lines.append("")
    for name, duration, phrases in entries:
        lines.append(f"## {name}  (duration: {format_duration(duration)}, {len(phrases)} phrases)")
        if not phrases:
            lines.append("  _no speech detected_")
            lines.append("")
            continue
        for p in phrases:
            lines.append(f"  [{format_time(p['start'])}-{format_time(p['end'])}] {p['text']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack Whisper transcripts into takes_packed.md")
    ap.add_argument("--edit-dir", type=Path, required=True, help="Edit directory containing transcripts/")
    ap.add_argument(
        "--silence-threshold",
        type=float,
        default=0.5,
        help="Break phrases on silences >= this (seconds). Default 0.5.",
    )
    ap.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path (default: <edit-dir>/takes_packed.md)",
    )
    args = ap.parse_args()

    edit_dir = args.edit_dir.resolve()
    transcripts_dir = edit_dir / "transcripts"
    if not transcripts_dir.is_dir():
        sys.exit(f"no transcripts directory at {transcripts_dir}")

    json_files = sorted(transcripts_dir.glob("*.json"))
    if not json_files:
        sys.exit(f"no .json files in {transcripts_dir}")

    entries = [pack_one_file(p, args.silence_threshold) for p in json_files]
    markdown = render_markdown(entries, args.silence_threshold)

    out_path = args.output or (edit_dir / "takes_packed.md")
    out_path.write_text(markdown, encoding="utf-8")

    total_phrases = sum(len(e[2]) for e in entries)
    total_duration = sum(e[1] for e in entries)
    kb = out_path.stat().st_size / 1024
    print(f"packed {len(entries)} transcripts -> {out_path}")
    print(f"  {total_phrases} phrases, {format_duration(total_duration)} total runtime")
    print(f"  {kb:.1f} KB")


if __name__ == "__main__":
    main()
