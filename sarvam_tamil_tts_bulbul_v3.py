from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import requests


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
CAPTION_DIR = BASE_DIR / "downloads" / "captions"
OUTPUT_DIR = BASE_DIR / "downloads" / "tts"
SOURCE_AUDIO_DIR = BASE_DIR / "downloads" / "audio"
TTS_URL = "https://api.sarvam.ai/text-to-speech"
MAX_CHARS_V3 = 2500
SAFE_CHUNK_SIZE = 2000
DEFAULT_PAUSE_MS = 350
DEFAULT_BLANK_LINE_PAUSE_MS = 650
DEFAULT_MAX_GAP_MS = 4000
DEFAULT_SYNC_TOLERANCE_SEC = 0.15


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


def stem_without_tamil_suffix(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".ta-IN"):
        return stem[:-6]
    return stem


def extract_video_id_from_name(name: str) -> str | None:
    match = re.search(r"\[([^\[\]]+)\]$", name.strip())
    if match:
        return match.group(1)
    return None


def find_matching_source_audio(caption_path: Path) -> Path | None:
    if not SOURCE_AUDIO_DIR.exists():
        return None

    base_stem = stem_without_tamil_suffix(caption_path)
    direct_matches = [p for p in SOURCE_AUDIO_DIR.iterdir() if p.is_file() and p.stem == base_stem]
    if direct_matches:
        return max(direct_matches, key=lambda p: p.stat().st_mtime)

    video_id = extract_video_id_from_name(base_stem)
    if video_id:
        id_matches = [p for p in SOURCE_AUDIO_DIR.iterdir() if p.is_file() and f"[{video_id}]" in p.stem]
        if id_matches:
            return max(id_matches, key=lambda p: p.stat().st_mtime)

    return None


def ffprobe_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not parse duration from ffprobe output: {result.stdout}") from exc


def build_atempo_chain(speed_factor: float) -> str:
    if speed_factor <= 0:
        raise ValueError("Invalid speed factor.")

    factors: list[float] = []
    remaining = speed_factor

    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0

    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5

    if abs(remaining - 1.0) > 1e-4:
        factors.append(remaining)

    if not factors:
        return "atempo=1.0"

    return ",".join(f"atempo={f:.6f}" for f in factors)


def sync_wav_to_target_duration(
    wav_path: Path,
    target_seconds: float,
    tolerance_seconds: float = DEFAULT_SYNC_TOLERANCE_SEC,
) -> dict[str, float | bool]:
    current_seconds = ffprobe_duration_seconds(wav_path)
    delta = current_seconds - target_seconds

    if abs(delta) <= tolerance_seconds or target_seconds <= 0:
        return {
            "synced": False,
            "before_seconds": current_seconds,
            "after_seconds": current_seconds,
            "target_seconds": target_seconds,
            "speed_factor": 1.0,
        }

    speed_factor = current_seconds / target_seconds
    atempo_chain = build_atempo_chain(speed_factor)

    with tempfile.TemporaryDirectory(prefix="sarvam_sync_") as tmp:
        temp_out = Path(tmp) / "synced.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(wav_path),
            "-filter:a",
            atempo_chain,
            str(temp_out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg sync failed: {result.stderr}")

        wav_path.write_bytes(temp_out.read_bytes())

    after_seconds = ffprobe_duration_seconds(wav_path)
    return {
        "synced": True,
        "before_seconds": current_seconds,
        "after_seconds": after_seconds,
        "target_seconds": target_seconds,
        "speed_factor": speed_factor,
    }


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


def detect_speech_windows_ms(
    audio_path: Path,
    silence_noise_db: str = "-32dB",
    silence_min_sec: float = 0.25,
    min_speech_ms: int = 120,
) -> list[tuple[int, int]]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={silence_noise_db}:d={silence_min_sec}",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg silencedetect failed: {result.stderr}")

    stderr = result.stderr
    silence_start_re = re.compile(r"silence_start:\s*([0-9.]+)")
    silence_end_re = re.compile(r"silence_end:\s*([0-9.]+)")

    events: list[tuple[str, float]] = []
    for line in stderr.splitlines():
        m_start = silence_start_re.search(line)
        if m_start:
            events.append(("start", float(m_start.group(1))))
            continue
        m_end = silence_end_re.search(line)
        if m_end:
            events.append(("end", float(m_end.group(1))))

    total_sec = ffprobe_duration_seconds(audio_path)
    speech_start_sec = 0.0
    windows: list[tuple[int, int]] = []

    for event_type, timestamp_sec in events:
        if event_type == "start":
            speech_end_sec = max(0.0, timestamp_sec)
            if speech_end_sec > speech_start_sec:
                start_ms = int(round(speech_start_sec * 1000.0))
                end_ms = int(round(speech_end_sec * 1000.0))
                if end_ms - start_ms >= min_speech_ms:
                    windows.append((start_ms, end_ms))
        elif event_type == "end":
            speech_start_sec = max(0.0, timestamp_sec)

    if total_sec > speech_start_sec:
        start_ms = int(round(speech_start_sec * 1000.0))
        end_ms = int(round(total_sec * 1000.0))
        if end_ms - start_ms >= min_speech_ms:
            windows.append((start_ms, end_ms))

    return windows


def assign_segments_to_speech_windows(
    segments: list[dict[str, Any]],
    windows: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    if not segments or not windows:
        return segments

    total_speech_ms = sum(max(0, end - start) for start, end in windows)
    if total_speech_ms <= 0:
        return segments

    weights = [max(1, len(str(seg.get("text", "")))) for seg in segments]
    weight_sum = sum(weights) or len(segments)
    target_durations = [max(120, int(round(total_speech_ms * (w / weight_sum)))) for w in weights]

    # Normalize to total speech duration.
    diff = total_speech_ms - sum(target_durations)
    target_durations[-1] = max(120, target_durations[-1] + diff)

    window_idx = 0
    cursor_ms = windows[0][0]

    for segment, duration_ms in zip(segments, target_durations):
        remaining = duration_ms
        start_ms = cursor_ms
        last_pos = cursor_ms

        while remaining > 0 and window_idx < len(windows):
            win_start, win_end = windows[window_idx]
            if cursor_ms < win_start:
                cursor_ms = win_start
                if start_ms < win_start:
                    start_ms = win_start

            available = win_end - cursor_ms
            if available <= 0:
                window_idx += 1
                if window_idx < len(windows):
                    cursor_ms = windows[window_idx][0]
                continue

            take = min(remaining, available)
            cursor_ms += take
            last_pos = cursor_ms
            remaining -= take

            if remaining > 0:
                window_idx += 1
                if window_idx < len(windows):
                    cursor_ms = windows[window_idx][0]

        segment["start_ms"] = int(start_ms)
        segment["end_ms"] = int(max(start_ms + 120, last_pos))
        segment["timing_source"] = "detected_speech_windows"

    return segments


def load_tts_segments(
    caption_path: Path,
    pause_ms: int,
    blank_line_pause_ms: int,
    max_gap_ms: int,
    source_audio_path: Path | None = None,
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

    has_explicit_timing = any(("start_ms" in s and "end_ms" in s) for s in segments)
    if (not has_explicit_timing) and source_audio_path and source_audio_path.exists():
        windows = detect_speech_windows_ms(source_audio_path)
        if windows:
            segments = assign_segments_to_speech_windows(segments, windows)

    if segments:
        first_start = segments[0].get("start_ms")
        if first_start is not None:
            segments[0]["pause_before_ms"] = max(0, int(round(float(first_start))))
        else:
            segments[0]["pause_before_ms"] = 0

    for idx, segment in enumerate(segments):
        start_ms = segment.get("start_ms")
        end_ms = segment.get("end_ms")
        if start_ms is not None and end_ms is not None and end_ms > start_ms:
            segment["target_duration_ms"] = max(100, int(round(float(end_ms) - float(start_ms))))

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


def wav_duration_ms(audio_bytes: bytes) -> float:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_in:
        frames = wav_in.getnframes()
        framerate = wav_in.getframerate()
    if framerate <= 0:
        return 0.0
    return (frames / float(framerate)) * 1000.0


def fit_clip_to_target_duration(audio_bytes: bytes, target_ms: float) -> bytes:
    target_ms = max(100.0, float(target_ms))
    current_ms = wav_duration_ms(audio_bytes)
    if current_ms <= 0:
        return audio_bytes

    ratio = current_ms / target_ms
    if 0.98 <= ratio <= 1.02:
        return audio_bytes

    atempo_chain = build_atempo_chain(ratio)
    target_sec = target_ms / 1000.0

    with tempfile.TemporaryDirectory(prefix="sarvam_fit_") as tmp:
        in_path = Path(tmp) / "in.wav"
        out_path = Path(tmp) / "out.wav"
        in_path.write_bytes(audio_bytes)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(in_path),
            "-filter:a",
            f"{atempo_chain},apad=pad_dur={target_sec:.3f},atrim=duration={target_sec:.3f}",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg duration-fit failed: {result.stderr}")

        return out_path.read_bytes()


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

            pause_before_ms = max(0, int(round(float(clip.get("pause_before_ms", 0) or 0))))
            if pause_before_ms and reference_params is not None:
                wav_out.writeframes(build_silence_frames(pause_before_ms, reference_params))

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
    parser.add_argument("--no-sync-original", action="store_true", help="Disable duration sync with original source audio.")
    parser.add_argument("--force-global-sync", action="store_true", help="Apply whole-file duration sync even when segment timestamps are used.")
    parser.add_argument("--sync-tolerance-sec", type=float, default=DEFAULT_SYNC_TOLERANCE_SEC, help="Skip sync if duration difference is below this value (seconds).")
    args = parser.parse_args()

    api_key = read_env_value(ENV_PATH, "SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY is empty in .env")

    caption_path = Path(args.caption) if args.caption else find_latest_tamil_caption(CAPTION_DIR)
    if not caption_path.exists():
        raise FileNotFoundError(f"Caption file not found: {caption_path}")

    source_audio = find_matching_source_audio(caption_path)

    segments = load_tts_segments(
        caption_path=caption_path,
        pause_ms=args.pause_ms,
        blank_line_pause_ms=args.blank_line_pause_ms,
        max_gap_ms=args.max_gap_ms,
        source_audio_path=source_audio,
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
        target_duration_ms = segment.get("target_duration_ms")
        if target_duration_ms is not None:
            audio_bytes = fit_clip_to_target_duration(audio_bytes, float(target_duration_ms))

        pause_before_ms = max(0, int(segment.get("pause_before_ms", 0) or 0))
        pause_after_ms = max(0, int(segment.get("pause_after_ms", 0) or 0))
        audio_clips.append(
            {
                "audio_bytes": audio_bytes,
                "pause_before_ms": pause_before_ms,
                "pause_after_ms": pause_after_ms,
            }
        )
        total_api_chunks += chunk_count
        total_inserted_pause_ms += (pause_before_ms + pause_after_ms)

    out_path.write_bytes(stitch_wav_clips(audio_clips))

    sync_info: dict[str, float | bool] | None = None
    has_segment_timing = any(
        ("start_ms" in segment and "end_ms" in segment) for segment in segments
    )
    should_global_sync = (
        (not args.no_sync_original)
        and source_audio
        and source_audio.exists()
        and (args.force_global_sync or not has_segment_timing)
    )

    if should_global_sync:
        target_seconds = ffprobe_duration_seconds(source_audio)
        sync_info = sync_wav_to_target_duration(
            wav_path=out_path,
            target_seconds=target_seconds,
            tolerance_seconds=max(0.0, args.sync_tolerance_sec),
        )

    print("Source caption:", safe_console(str(caption_path)))
    print("Output audio:", safe_console(str(out_path)))
    print("Segments used:", len(segments))
    print("TTS API chunks:", total_api_chunks)
    print("Inserted pause (ms):", total_inserted_pause_ms)
    if source_audio:
        print("Matched source audio:", safe_console(str(source_audio)))
        print("Segment timestamp mode:", has_segment_timing)
        if sync_info:
            print("Sync applied:", bool(sync_info.get("synced", False)))
            print("Source duration (s):", round(float(sync_info.get("target_seconds", 0.0)), 3))
            print("TTS before sync (s):", round(float(sync_info.get("before_seconds", 0.0)), 3))
            print("TTS after sync (s):", round(float(sync_info.get("after_seconds", 0.0)), 3))
            print("Speed factor:", round(float(sync_info.get("speed_factor", 1.0)), 4))
        else:
            print("Sync applied: False (segment timestamps prioritized)")
    else:
        print("Matched source audio: none (sync skipped)")


if __name__ == "__main__":
    main()
