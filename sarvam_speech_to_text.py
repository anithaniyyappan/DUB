from __future__ import annotations

import argparse
import subprocess
import json
import mimetypes
import os
import tempfile
import sys
from pathlib import Path
from typing import Any

import requests


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
API_URL = "https://api.sarvam.ai/speech-to-text"
AUDIO_DIR = BASE_DIR / "downloads" / "audio"
CAPTION_DIR = BASE_DIR / "downloads" / "captions"
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".amr", ".wma", ".mp4"}
MAX_REST_SECONDS = 30


def safe_console(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


def read_env_value(env_path: Path, key: str) -> str:
    env_direct = os.getenv(key, "").strip()
    if env_direct:
        return env_direct.strip('"').strip("'")

    if not env_path.exists():
        raise FileNotFoundError(f"{env_path} not found and environment variable {key} is not set.")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return ""


def find_latest_audio(audio_dir: Path) -> Path:
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio folder not found: {audio_dir}")

    candidates = [
        p for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_caption_text(response_json: Any) -> str:
    if isinstance(response_json, dict):
        for key in ("text", "transcript", "transcription"):
            value = response_json.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        results = response_json.get("results")
        if isinstance(results, dict):
            for key in ("text", "transcript"):
                value = results.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        segments = response_json.get("segments")
        if isinstance(segments, list):
            lines = []
            for seg in segments:
                if isinstance(seg, dict):
                    text = seg.get("text")
                    if isinstance(text, str) and text.strip():
                        lines.append(text.strip())
            if lines:
                return "\n".join(lines)

    return json.dumps(response_json, ensure_ascii=False, indent=2)


def save_outputs(audio_path: Path, response_json: Any, caption_text: str) -> tuple[Path, Path]:
    CAPTION_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = audio_path.stem
    text_out = CAPTION_DIR / f"{safe_stem}.txt"
    json_out = CAPTION_DIR / f"{safe_stem}.json"
    last_caption_ptr = CAPTION_DIR / ".last_caption_path"

    text_out.write_text(caption_text + "\n", encoding="utf-8")
    json_out.write_text(json.dumps(response_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    last_caption_ptr.write_text(str(text_out.resolve()), encoding="utf-8")
    return text_out, json_out


def transcribe(audio_path: Path, api_key: str) -> dict[str, Any]:
    headers = {"api-subscription-key": api_key}
    guessed_mime, _ = mimetypes.guess_type(audio_path.name)
    mime_type = guessed_mime or "application/octet-stream"
    session = requests.Session()
    session.trust_env = False
    with audio_path.open("rb") as audio_file:
        files = {"file": (audio_path.name, audio_file, mime_type)}
        response = session.post(API_URL, headers=headers, files=files, timeout=180)

    if response.status_code != 200:
        raise RuntimeError(f"Sarvam API failed ({response.status_code}): {response.text}")

    try:
        return response.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Expected JSON response but got: {response.text}") from exc


def sanitize_audio_path(raw_audio_arg: str) -> Path:
    raw = raw_audio_arg.strip().strip('"').strip("'")
    marker = ":\\"
    marker_pos = raw.find(marker)
    if marker_pos > 1:
        # Handle accidental prefixes like ".\downloads\audio\C:\..."
        drive_start = marker_pos - 1
        raw = raw[drive_start:]
    return Path(raw)


def ffmpeg_available() -> bool:
    return shutil_which("ffmpeg") is not None


def shutil_which(cmd: str) -> str | None:
    # Local helper to avoid importing large modules for one call path.
    from shutil import which
    return which(cmd)


def split_audio_to_chunks(audio_path: Path, chunk_seconds: int = MAX_REST_SECONDS) -> list[Path]:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required to split long audio but was not found in PATH.")

    temp_dir = Path(tempfile.mkdtemp(prefix="sarvam_chunks_"))
    out_pattern = temp_dir / "chunk_%03d.mp3"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-c",
        "copy",
        str(out_pattern),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg split failed: {completed.stderr}")

    chunks = sorted(temp_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise RuntimeError("No chunks were produced from audio splitting.")
    return chunks


def transcribe_with_auto_chunking(audio_path: Path, api_key: str) -> dict[str, Any]:
    try:
        return transcribe(audio_path, api_key)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "maximum limit of 30 seconds" not in msg:
            raise

    chunks = split_audio_to_chunks(audio_path, MAX_REST_SECONDS)
    full_text_parts: list[str] = []
    raw_parts: list[Any] = []

    for idx, chunk in enumerate(chunks, start=1):
        part_json = transcribe(chunk, api_key)
        raw_parts.append({"chunk": idx, "file": chunk.name, "response": part_json})
        text = extract_caption_text(part_json).strip()
        if text:
            full_text_parts.append(text)

    return {
        "mode": "chunked_rest",
        "source_audio": str(audio_path),
        "chunks": len(chunks),
        "text": "\n".join(full_text_parts).strip(),
        "parts": raw_parts,
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Transcribe audio from downloads/audio using Sarvam API.")
    parser.add_argument("--audio", type=str, default="", help="Optional path to audio file. Defaults to latest in downloads/audio.")
    args = parser.parse_args()

    api_key = read_env_value(ENV_PATH, "SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY is empty in .env")

    audio_path = sanitize_audio_path(args.audio) if args.audio else find_latest_audio(AUDIO_DIR)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    result_json = transcribe_with_auto_chunking(audio_path, api_key)
    caption_text = extract_caption_text(result_json)
    text_out, json_out = save_outputs(audio_path, result_json, caption_text)

    print("Audio used:", safe_console(str(audio_path)))
    print("Caption text file:", safe_console(str(text_out)))
    print("Raw response file:", safe_console(str(json_out)))
    print("\nTranscribed Text:\n")
    print(safe_console(caption_text))


if __name__ == "__main__":
    main()
