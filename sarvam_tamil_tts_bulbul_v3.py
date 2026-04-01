from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import wave
from pathlib import Path
from typing import Any

import requests


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
CAPTION_DIR = BASE_DIR / "downloads" / "captions"
OUTPUT_DIR = BASE_DIR / "downloads" / "tts"
TTS_URL = "https://api.sarvam.ai/text-to-speech"
MAX_CHARS_V3 = 2500
SAFE_CHUNK_SIZE = 2000
DEFAULT_PAUSE_MS = 350
DEFAULT_BLANK_LINE_PAUSE_MS = 650
DEFAULT_MAX_GAP_MS = 4000


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
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return ""


def find_latest_tamil_caption(caption_dir: Path) -> Path:
    if not caption_dir.exists():
        raise FileNotFoundError(f"Caption folder not found: {caption_dir}")

    candidates = [
        p for p in caption_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt" and p.name.endswith(".ta-IN.txt")
    ]
    if not candidates:
        raise FileNotFoundError(f"No Tamil caption .ta-IN.txt file found in {caption_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def chunk_text(text: str, chunk_size: int = SAFE_CHUNK_SIZE) -> list[str]:
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current = ""

    for line in lines:
        if len(line) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(line):
                chunks.append(line[start:start + chunk_size])
                start += chunk_size
            continue

        if len(current) + len(line) <= chunk_size:
            current += line
        else:
            if current:
                chunks.append(current)
            current = line

    if current:
        chunks.append(current)

    if not chunks and text:
        chunks = [text[:chunk_size]]

    return [c for c in chunks if c.strip()]


def load_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_line_segments(text: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    blank_lines_before = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            blank_lines_before += 1
            continue

        segments.append(
            {
                "index": len(segments) + 1,
                "text": line,
                "blank_lines_before": blank_lines_before,
            }
        )
        blank_lines_before = 0

    if not segments and text.strip():
        segments.append({"index": 1, "text": text.strip(), "blank_lines_before": 0})

    return segments


def build_sentence_segments(text: str) -> list[dict[str, Any]]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    return [{"index": idx, "text": part, "blank_lines_before": 0} for idx, part in enumerate(parts, start=1)]


def align_text_to_source_segments(text: str, source_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sentence_segments = build_sentence_segments(text)
    if len(sentence_segments) < 2 or not source_segments:
        return []

    target_count = len(source_segments)
    sentence_count = len(sentence_segments)
    if sentence_count < target_count:
        return []

    aligned: list[dict[str, Any]] = []
    start = 0

    for idx, source_segment in enumerate(source_segments):
        remaining_targets = target_count - idx
        boundary = round((idx + 1) * sentence_count / target_count)
        end = max(start + 1, boundary)
        max_end = sentence_count - (remaining_targets - 1)
        end = min(end, max_end)

        chunk_text = " ".join(segment["text"] for segment in sentence_segments[start:end]).strip()
        if not chunk_text:
            return []

        aligned.append(
            {
                "index": len(aligned) + 1,
                "text": chunk_text,
                "blank_lines_before": int(source_segment.get("blank_lines_before", 0) or 0),
            }
        )
        start = end

    return aligned if start == sentence_count else []


def coerce_time_ms(raw_value: Any, key_name: str) -> float | None:
    if raw_value in (None, ""):
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None

    if key_name.endswith("_ms") or abs(value) >= 10000:
        return value
    return value * 1000.0


def extract_time_ms(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in payload:
            value = coerce_time_ms(payload.get(key), key)
            if value is not None:
                return value
    return None


def infer_pause_ms(
    current_text: str,
    next_blank_lines: int,
    pause_ms: int,
    blank_line_pause_ms: int,
) -> int:
    inferred = pause_ms + max(0, next_blank_lines) * blank_line_pause_ms
    trimmed = current_text.rstrip()

    if trimmed.endswith(("?", "!")):
        inferred += 150
    elif trimmed.endswith((",", ";", ":")):
        inferred += 75

    return inferred


def load_tts_segments(
    caption_path: Path,
    pause_ms: int,
    blank_line_pause_ms: int,
    max_gap_ms: int,
) -> list[dict[str, Any]]:
    text = caption_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Caption file is empty.")

    json_payload = load_json_if_exists(caption_path.with_suffix(".json"))
    segments: list[dict[str, Any]] = []

    if isinstance(json_payload, dict) and isinstance(json_payload.get("segments"), list):
        for item in json_payload["segments"]:
            if not isinstance(item, dict):
                continue

            segment_text = str(
                item.get("final_text")
                or item.get("text")
                or item.get("translated_text")
                or item.get("styled_text")
                or item.get("source_text")
                or ""
            ).strip()
            if not segment_text:
                continue

            segment: dict[str, Any] = {
                "index": len(segments) + 1,
                "text": segment_text,
                "blank_lines_before": int(item.get("blank_lines_before", 0) or 0),
            }

            explicit_pause = coerce_time_ms(item.get("pause_after_ms"), "pause_after_ms")
            if explicit_pause is not None:
                segment["pause_after_ms"] = max(0, int(round(explicit_pause)))

            start_ms = extract_time_ms(item, ("start_ms", "start", "start_time", "startTime"))
            end_ms = extract_time_ms(item, ("end_ms", "end", "end_time", "endTime"))
            if start_ms is not None:
                segment["start_ms"] = round(start_ms, 3)
            if end_ms is not None:
                segment["end_ms"] = round(end_ms, 3)

            segments.append(segment)

    if not segments and isinstance(json_payload, dict):
        source_file = str(json_payload.get("source_file", "")).strip()
        source_path = Path(source_file) if source_file else None
        if source_path and source_path.exists():
            source_segments = build_line_segments(source_path.read_text(encoding="utf-8"))
            aligned_segments = align_text_to_source_segments(text, source_segments)
            if aligned_segments:
                segments = aligned_segments

    if not segments:
        segments = build_line_segments(text)

    if not segments:
        raise ValueError("No valid caption segments found.")

    for idx, segment in enumerate(segments):
        if idx == len(segments) - 1:
            segment["pause_after_ms"] = 0
            continue

        if "pause_after_ms" in segment:
            continue

        next_segment = segments[idx + 1]
        current_end = segment.get("end_ms")
        next_start = next_segment.get("start_ms")

        if current_end is not None and next_start is not None:
            gap_ms = max(0, int(round(next_start - current_end)))
            segment["pause_after_ms"] = min(gap_ms, max_gap_ms)
            continue

        segment["pause_after_ms"] = infer_pause_ms(
            current_text=segment["text"],
            next_blank_lines=int(next_segment.get("blank_lines_before", 0) or 0),
            pause_ms=pause_ms,
            blank_line_pause_ms=blank_line_pause_ms,
        )

    return segments


def tts_request(
    api_key: str,
    text: str,
    speaker: str,
    pace: float,
    temperature: float,
    sample_rate: int,
) -> dict[str, Any]:
    if len(text) > MAX_CHARS_V3:
        raise ValueError(f"TTS chunk exceeds {MAX_CHARS_V3} characters.")

    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "target_language_code": "ta-IN",
        "speaker": speaker,
        "model": "bulbul:v3",
        "pace": pace,
        "temperature": temperature,
        "speech_sample_rate": sample_rate,
        "output_audio_codec": "wav",
    }
    session = requests.Session()
    session.trust_env = False
    response = session.post(TTS_URL, headers=headers, json=payload, timeout=180)
    if response.status_code != 200:
        raise RuntimeError(f"Sarvam TTS failed ({response.status_code}): {response.text}")
    return response.json()


def decode_audio_from_response(response_json: dict[str, Any]) -> bytes:
    audios = response_json.get("audios")
    if not isinstance(audios, list) or not audios:
        raise RuntimeError(f"Invalid TTS response (missing audios): {json.dumps(response_json, ensure_ascii=False)}")
    audio_b64 = audios[0]
    if not isinstance(audio_b64, str) or not audio_b64.strip():
        raise RuntimeError("Empty audio payload received from TTS.")
    return base64.b64decode(audio_b64)


def read_wav_clip(audio_bytes: bytes) -> tuple[wave._wave_params, bytes]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_in:
        params = wav_in.getparams()
        frames = wav_in.readframes(wav_in.getnframes())
    return params, frames


def validate_wav_params(reference: wave._wave_params, current: wave._wave_params) -> None:
    if (
        reference.nchannels != current.nchannels
        or reference.sampwidth != current.sampwidth
        or reference.framerate != current.framerate
        or reference.comptype != current.comptype
    ):
        raise RuntimeError("TTS returned incompatible WAV chunks; cannot stitch them safely.")


def build_silence_frames(duration_ms: int, params: wave._wave_params) -> bytes:
    if duration_ms <= 0:
        return b""
    frame_count = int(params.framerate * (duration_ms / 1000.0))
    bytes_per_frame = params.nchannels * params.sampwidth
    return b"\x00" * frame_count * bytes_per_frame


def stitch_wav_clips(clips: list[dict[str, Any]]) -> bytes:
    if not clips:
        raise ValueError("No audio clips to stitch.")

    output_buffer = io.BytesIO()
    reference_params: wave._wave_params | None = None

    with wave.open(output_buffer, "wb") as wav_out:
        for clip in clips:
            params, frames = read_wav_clip(clip["audio_bytes"])
            if reference_params is None:
                reference_params = params
                wav_out.setnchannels(params.nchannels)
                wav_out.setsampwidth(params.sampwidth)
                wav_out.setframerate(params.framerate)
            else:
                validate_wav_params(reference_params, params)

            wav_out.writeframes(frames)

            pause_after_ms = max(0, int(round(float(clip.get("pause_after_ms", 0) or 0))))
            if pause_after_ms and reference_params is not None:
                wav_out.writeframes(build_silence_frames(pause_after_ms, reference_params))

    return output_buffer.getvalue()


def synthesize_segment_audio(
    api_key: str,
    text: str,
    speaker: str,
    pace: float,
    temperature: float,
    sample_rate: int,
) -> tuple[bytes, int]:
    clips: list[dict[str, Any]] = []
    chunks = chunk_text(text, SAFE_CHUNK_SIZE)

    for chunk in chunks:
        response_json = tts_request(
            api_key=api_key,
            text=chunk,
            speaker=speaker,
            pace=pace,
            temperature=temperature,
            sample_rate=sample_rate,
        )
        clips.append({"audio_bytes": decode_audio_from_response(response_json), "pause_after_ms": 0})

    if not clips:
        raise ValueError("No valid text chunks found for TTS.")

    if len(clips) == 1:
        return clips[0]["audio_bytes"], 1

    return stitch_wav_clips(clips), len(clips)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Convert Tamil caption text to speech using Sarvam Bulbul v3.")
    parser.add_argument("--caption", type=str, default="", help="Path to source Tamil caption (.ta-IN.txt).")
    parser.add_argument("--speaker", type=str, default="shubh", help="Bulbul v3 speaker (lowercase).")
    parser.add_argument("--pace", type=float, default=1.0, help="Speech pace for bulbul:v3 (0.5 to 2.0).")
    parser.add_argument("--temperature", type=float, default=0.4, help="TTS temperature (0.01 to 2.0).")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Sample rate, e.g. 24000.")
    parser.add_argument("--pause-ms", type=int, default=DEFAULT_PAUSE_MS, help="Fallback pause after each caption line in milliseconds.")
    parser.add_argument("--blank-line-pause-ms", type=int, default=DEFAULT_BLANK_LINE_PAUSE_MS, help="Extra fallback pause for blank lines between caption segments.")
    parser.add_argument("--max-gap-ms", type=int, default=DEFAULT_MAX_GAP_MS, help="Maximum silence inserted from timing metadata, in milliseconds.")
    args = parser.parse_args()

    api_key = read_env_value(ENV_PATH, "SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY is empty in .env")

    caption_path = Path(args.caption) if args.caption else find_latest_tamil_caption(CAPTION_DIR)
    if not caption_path.exists():
        raise FileNotFoundError(f"Caption file not found: {caption_path}")

    segments = load_tts_segments(
        caption_path=caption_path,
        pause_ms=args.pause_ms,
        blank_line_pause_ms=args.blank_line_pause_ms,
        max_gap_ms=args.max_gap_ms,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{caption_path.stem}.bulbul-v3.wav"

    audio_clips: list[dict[str, Any]] = []
    total_api_chunks = 0
    total_inserted_pause_ms = 0

    for segment in segments:
        audio_bytes, chunk_count = synthesize_segment_audio(
            api_key=api_key,
            text=segment["text"],
            speaker=args.speaker,
            pace=args.pace,
            temperature=args.temperature,
            sample_rate=args.sample_rate,
        )
        pause_after_ms = max(0, int(segment.get("pause_after_ms", 0) or 0))
        audio_clips.append({"audio_bytes": audio_bytes, "pause_after_ms": pause_after_ms})
        total_api_chunks += chunk_count
        total_inserted_pause_ms += pause_after_ms

    out_path.write_bytes(stitch_wav_clips(audio_clips))

    print("Source caption:", safe_console(str(caption_path)))
    print("Output audio:", safe_console(str(out_path)))
    print("Segments used:", len(segments))
    print("TTS API chunks:", total_api_chunks)
    print("Inserted pause (ms):", total_inserted_pause_ms)


if __name__ == "__main__":
    main()
