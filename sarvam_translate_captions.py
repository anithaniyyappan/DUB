from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
CAPTION_DIR = BASE_DIR / "downloads" / "captions"
SOURCE_AUDIO_DIR = BASE_DIR / "downloads" / "audio"
DEFAULT_SLANG_DICT_PATH = BASE_DIR / "tamil_slang_dictionary.json"
API_URL = "https://api.sarvam.ai/translate"
CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"
MAX_INPUT_CHARS = 2000
SAFE_CHUNK_SIZE = 1800
STYLE_CHUNK_SIZE = 1400
MIN_LENGTH_RATIO = 0.7
CHUNKED_STT_SECONDS = 30


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

    silence_start_re = re.compile(r"silence_start:\s*([0-9.]+)")
    silence_end_re = re.compile(r"silence_end:\s*([0-9.]+)")
    events: list[tuple[str, float]] = []

    for line in result.stderr.splitlines():
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


def assign_segments_by_weight(
    segments: list[dict[str, Any]],
    ranges: list[tuple[int, int]],
    timing_source: str,
) -> list[dict[str, Any]]:
    if not segments or not ranges:
        return segments

    total_ms = sum(max(0, end - start) for start, end in ranges)
    if total_ms <= 0:
        return segments

    weights = [max(1, len(str(seg.get("source_text", "")))) for seg in segments]
    weight_sum = sum(weights) or len(segments)
    target_durations = [max(120, int(round(total_ms * (w / weight_sum)))) for w in weights]

    diff = total_ms - sum(target_durations)
    target_durations[-1] = max(120, target_durations[-1] + diff)

    range_idx = 0
    cursor_ms = ranges[0][0]

    for segment, duration_ms in zip(segments, target_durations):
        remaining = duration_ms
        start_ms = cursor_ms
        last_pos = cursor_ms

        while remaining > 0 and range_idx < len(ranges):
            win_start, win_end = ranges[range_idx]
            if cursor_ms < win_start:
                cursor_ms = win_start
                if start_ms < win_start:
                    start_ms = win_start

            available = win_end - cursor_ms
            if available <= 0:
                range_idx += 1
                if range_idx < len(ranges):
                    cursor_ms = ranges[range_idx][0]
                continue

            take = min(remaining, available)
            cursor_ms += take
            last_pos = cursor_ms
            remaining -= take

            if remaining > 0:
                range_idx += 1
                if range_idx < len(ranges):
                    cursor_ms = ranges[range_idx][0]

        segment["start_ms"] = int(start_ms)
        segment["end_ms"] = int(max(start_ms + 120, last_pos))
        segment["timing_source"] = timing_source

    return segments


def apply_timing_to_segments(segments: list[dict[str, Any]], source_audio: Path | None) -> list[dict[str, Any]]:
    has_explicit_timing = any(("start_ms" in s and "end_ms" in s) for s in segments)
    if has_explicit_timing:
        return segments

    if source_audio and source_audio.exists():
        windows = detect_speech_windows_ms(source_audio)
        if windows:
            return assign_segments_by_weight(segments, windows, "detected_speech_windows")

        total_ms = int(round(ffprobe_duration_seconds(source_audio) * 1000.0))
        if total_ms > 0:
            return assign_segments_by_weight(segments, [(0, total_ms)], "equal_split_fallback")

    return segments


def format_ms(ms: float | int) -> str:
    total_ms = max(0, int(round(float(ms))))
    h = total_ms // 3_600_000
    m = (total_ms % 3_600_000) // 60_000
    s = (total_ms % 60_000) // 1_000
    ms_part = total_ms % 1_000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms_part:03d}"


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


def find_latest_caption(caption_dir: Path) -> Path:
    if not caption_dir.exists():
        raise FileNotFoundError(f"Caption folder not found: {caption_dir}")

    last_caption_ptr = caption_dir / ".last_caption_path"
    if last_caption_ptr.exists():
        candidate = Path(last_caption_ptr.read_text(encoding="utf-8").strip())
        if candidate.exists() and candidate.suffix.lower() == ".txt" and not candidate.name.endswith(".ta-IN.txt"):
            return candidate

    candidates = [p for p in caption_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt" and not p.name.endswith(".ta-IN.txt")]
    if not candidates:
        raise FileNotFoundError(f"No source caption .txt file found in {caption_dir}")
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

    return chunks


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

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            blank_lines_before += 1
            continue

        segments.append(
            {
                "index": len(segments) + 1,
                "line_number": line_number,
                "blank_lines_before": blank_lines_before,
                "source_text": line,
            }
        )
        blank_lines_before = 0

    if not segments and text.strip():
        segments.append(
            {
                "index": 1,
                "line_number": 1,
                "blank_lines_before": 0,
                "source_text": text.strip(),
            }
        )

    return segments


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


def normalize_raw_segments(raw_segments: Any, chunk_index: int = 0) -> list[dict[str, Any]]:
    if not isinstance(raw_segments, list):
        return []

    chunk_offset_ms = (chunk_index - 1) * CHUNKED_STT_SECONDS * 1000.0 if chunk_index else 0.0
    normalized: list[dict[str, Any]] = []

    for item in raw_segments:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text") or item.get("transcript") or item.get("utterance") or "").strip()
        if not text:
            continue

        record: dict[str, Any] = {
            "index": len(normalized) + 1,
            "source_text": text,
        }
        if chunk_index:
            record["source_chunk"] = chunk_index

        start_ms = extract_time_ms(item, ("start_ms", "start", "start_time", "startTime"))
        end_ms = extract_time_ms(item, ("end_ms", "end", "end_time", "endTime"))
        if start_ms is not None:
            record["start_ms"] = round(start_ms + chunk_offset_ms, 3)
        if end_ms is not None:
            record["end_ms"] = round(end_ms + chunk_offset_ms, 3)

        normalized.append(record)

    return normalized


def merge_segments_with_source_text(
    caption_segments: list[dict[str, Any]],
    raw_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not raw_segments:
        return caption_segments

    if len(caption_segments) != len(raw_segments):
        has_timing = any("start_ms" in segment or "end_ms" in segment for segment in raw_segments)
        return raw_segments if has_timing else caption_segments

    merged: list[dict[str, Any]] = []
    for caption_segment, raw_segment in zip(caption_segments, raw_segments):
        merged_segment = dict(raw_segment)
        merged_segment["index"] = caption_segment["index"]
        merged_segment["line_number"] = caption_segment["line_number"]
        merged_segment["blank_lines_before"] = caption_segment["blank_lines_before"]
        merged_segment["source_text"] = caption_segment["source_text"]
        merged.append(merged_segment)
    return merged


def load_source_segments(caption_path: Path, input_text: str) -> list[dict[str, Any]]:
    caption_segments = build_line_segments(input_text)
    raw_json = load_json_if_exists(caption_path.with_suffix(".json"))
    if not isinstance(raw_json, dict):
        return caption_segments

    raw_segments = normalize_raw_segments(raw_json.get("segments"))
    if raw_segments:
        return merge_segments_with_source_text(caption_segments, raw_segments)

    parts = raw_json.get("parts")
    if isinstance(parts, list):
        combined_segments: list[dict[str, Any]] = []
        for chunk_index, part in enumerate(parts, start=1):
            if not isinstance(part, dict):
                continue
            response = part.get("response")
            if not isinstance(response, dict):
                continue

            nested_segments = normalize_raw_segments(response.get("segments"), chunk_index=chunk_index)
            if nested_segments:
                combined_segments.extend(nested_segments)
                continue

            transcript = str(response.get("transcript") or response.get("text") or "").strip()
            if not transcript:
                continue

            for segment in build_line_segments(transcript):
                segment["source_chunk"] = chunk_index
                combined_segments.append(segment)

        if combined_segments:
            return merge_segments_with_source_text(caption_segments, combined_segments)

    return caption_segments


def render_segments_text(segments: list[dict[str, Any]], text_key: str) -> str:
    lines: list[str] = []
    for segment in segments:
        text = str(segment.get(text_key, "")).strip()
        if not text:
            continue
        if lines:
            lines.extend([""] * int(segment.get("blank_lines_before", 0)))
        lines.append(text)
    return "\n".join(lines).strip()


def render_timed_segments_text(segments: list[dict[str, Any]], text_key: str) -> str:
    lines: list[str] = []
    for segment in segments:
        text = str(segment.get(text_key, "")).strip()
        if not text:
            continue
        start_ms = segment.get("start_ms")
        end_ms = segment.get("end_ms")
        if start_ms is not None and end_ms is not None:
            lines.append(f"[{format_ms(start_ms)} --> {format_ms(end_ms)}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines).strip()


def translate_text_chunks(
    api_key: str,
    text: str,
    source_language_code: str,
    target_language_code: str,
) -> tuple[str, list[dict[str, Any]]]:
    translated_parts: list[str] = []
    raw_parts: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunk_text(text, SAFE_CHUNK_SIZE), start=1):
        result = translate_chunk(
            api_key=api_key,
            text=chunk,
            source_language_code=source_language_code,
            target_language_code=target_language_code,
        )
        translated = str(result.get("translated_text", "")).strip() or chunk.strip()
        translated_parts.append(translated)
        raw_parts.append({"chunk": idx, "input": chunk, "response": result})

    return "".join(translated_parts).strip(), raw_parts


def style_text_chunks(
    api_key: str,
    text: str,
    model: str,
    slang_guidance: str,
) -> tuple[str, list[dict[str, Any]]]:
    styled_parts: list[str] = []
    raw_parts: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunk_text(text, STYLE_CHUNK_SIZE), start=1):
        styled = style_tamil_chunk(api_key=api_key, text=chunk, model=model, slang_guidance=slang_guidance)
        styled_parts.append(styled.strip() or chunk.strip())
        raw_parts.append({"chunk": idx, "input": chunk, "output": styled})

    return "".join(styled_parts).strip(), raw_parts


def spell_fix_text_chunks(
    api_key: str,
    text: str,
    model: str,
) -> tuple[str, list[dict[str, Any]]]:
    corrected_parts: list[str] = []
    raw_parts: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunk_text(text, STYLE_CHUNK_SIZE), start=1):
        corrected = fix_tamil_spelling_chunk(api_key=api_key, text=chunk, model=model)
        corrected_parts.append(corrected.strip() or chunk.strip())
        raw_parts.append({"chunk": idx, "input": chunk, "output": corrected})

    return "".join(corrected_parts).strip(), raw_parts


def load_slang_dictionary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"replacements": [], "preferred_phrases": [], "avoid_phrases": []}

    if not path.exists():
        raise FileNotFoundError(f"Slang dictionary file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    replacements = data.get("replacements", [])
    preferred = data.get("preferred_phrases", [])
    avoid = data.get("avoid_phrases", [])

    if not isinstance(replacements, list) or not isinstance(preferred, list) or not isinstance(avoid, list):
        raise ValueError("Invalid slang dictionary format.")

    return {
        "replacements": replacements,
        "preferred_phrases": preferred,
        "avoid_phrases": avoid,
    }


def build_slang_guidance(slang_dict: dict[str, Any]) -> str:
    preferred = [str(x).strip() for x in slang_dict.get("preferred_phrases", []) if str(x).strip()]
    avoid = [str(x).strip() for x in slang_dict.get("avoid_phrases", []) if str(x).strip()]
    replacements = slang_dict.get("replacements", [])

    lines: list[str] = []
    if preferred:
        lines.append("Prefer these colloquial phrases when natural:")
        for phrase in preferred[:12]:
            lines.append(f"- {phrase}")
    if avoid:
        lines.append("Avoid these phrases:")
        for phrase in avoid[:12]:
            lines.append(f"- {phrase}")
    if replacements:
        lines.append("Use these wording preferences when context matches:")
        for item in replacements[:20]:
            if isinstance(item, dict):
                src = str(item.get("from", "")).strip()
                dst = str(item.get("to", "")).strip()
                if src and dst:
                    lines.append(f"- {src} -> {dst}")
    return "\n".join(lines).strip()


def apply_slang_replacements(text: str, slang_dict: dict[str, Any]) -> str:
    replacements = slang_dict.get("replacements", [])
    pairs: list[tuple[str, str]] = []
    for item in replacements:
        if not isinstance(item, dict):
            continue
        src = str(item.get("from", "")).strip()
        dst = str(item.get("to", "")).strip()
        if src and dst:
            pairs.append((src, dst))

    # Apply longer matches first to reduce overlap issues.
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    out = text
    for src, dst in pairs:
        out = out.replace(src, dst)
    return out


def translate_chunk(
    api_key: str,
    text: str,
    source_language_code: str = "auto",
    target_language_code: str = "ta-IN",
) -> dict[str, Any]:
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"Chunk exceeds API max input size ({MAX_INPUT_CHARS}).")

    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "input": text,
        "source_language_code": source_language_code,
        "target_language_code": target_language_code,
    }
    session = requests.Session()
    session.trust_env = False
    response = session.post(API_URL, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Sarvam Translate failed ({response.status_code}): {response.text}")
    return response.json()


def style_tamil_chunk(api_key: str, text: str, model: str = "sarvam-m", slang_guidance: str = "") -> str:
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    prompt = (
        "You are a Tamil caption polisher for short videos.\n"
        "Task: rewrite the given text in lively, emotional, natural spoken Tamil.\n"
        "Rules:\n"
        "- Keep meaning intact.\n"
        "- Use colloquial Tamil naturally, with friendly human slang.\n"
        "- Avoid robotic/textbook Tamil.\n"
        "- Keep it clean (family-safe), concise, and easy to read.\n"
        "- Keep scientific terms in English only when unavoidable.\n"
        "- Output only rewritten Tamil text.\n\n"
        "Style examples (follow this vibe):\n"
        "Formal: \"இதற்கு காரணம் என்ன?\"\n"
        "Spoken: \"இதுக்குத்தான் காரணம் என்ன தெரியுமா?\"\n"
        "Formal: \"நீங்கள் கருத்துகளில் பதிவு செய்யலாம்.\"\n"
        "Spoken: \"கமெண்ட்ல சொல்லுங்க பார்ப்போம்!\"\n\n"
        "Now rewrite:\n"
        f"{text}"
    )
    if slang_guidance:
        prompt = prompt + "\n\nCustom slang dictionary guidance:\n" + slang_guidance
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1200,
        "temperature": 0.2,
    }
    session = requests.Session()
    session.trust_env = False
    response = session.post(CHAT_URL, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Sarvam Chat style rewrite failed ({response.status_code}): {response.text}")

    data = response.json()
    try:
        raw = data["choices"][0]["message"]["content"].strip()
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
        lower = cleaned.lower()
        if "<think>" in lower:
            idx = lower.find("<think>")
            cleaned = cleaned[:idx].strip()
        if not cleaned:
            return text
        if len(cleaned) < int(len(text) * MIN_LENGTH_RATIO):
            return text
        return cleaned
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unexpected chat response format: {json.dumps(data, ensure_ascii=False)}") from exc


def fix_tamil_spelling_chunk(api_key: str, text: str, model: str = "sarvam-m") -> str:
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }
    prompt = (
        "Correct spelling and minor grammar in this Tamil text.\n"
        "Rules:\n"
        "- Preserve current colloquial/slang tone.\n"
        "- Do not make it formal.\n"
        "- Keep meaning exactly the same.\n"
        "- Keep scientific terms as-is when needed.\n"
        "- Output only corrected Tamil text.\n\n"
        f"Text:\n{text}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1200,
        "temperature": 0,
    }
    session = requests.Session()
    session.trust_env = False
    response = session.post(CHAT_URL, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"Sarvam spelling pass failed ({response.status_code}): {response.text}")

    data = response.json()
    try:
        raw = data["choices"][0]["message"]["content"].strip()
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
        lower = cleaned.lower()
        if "<think>" in lower:
            idx = lower.find("<think>")
            cleaned = cleaned[:idx].strip()
        if not cleaned:
            return text
        if len(cleaned) < int(len(text) * MIN_LENGTH_RATIO):
            return text
        return cleaned
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unexpected spelling response format: {json.dumps(data, ensure_ascii=False)}") from exc


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Translate caption text to Tamil using Sarvam Translate API.")
    parser.add_argument("--caption", type=str, default="", help="Optional source .txt path. Defaults to latest .txt in downloads/captions.")
    parser.add_argument("--source-lang", type=str, default="auto", help="Source language code (default: auto).")
    parser.add_argument("--target-lang", type=str, default="ta-IN", help="Target language code (default: ta-IN for Tamil).")
    parser.add_argument("--style", type=str, default="colloquial", choices=["formal", "colloquial"], help="Tamil output style (default: colloquial).")
    parser.add_argument("--chat-model", type=str, default="sarvam-m", help="Sarvam chat model for style rewrite.")
    parser.add_argument("--spell-fix", action="store_true", help="Run final Tamil spelling cleanup pass.")
    parser.add_argument("--slang-dict", type=str, default="", help="Optional slang dictionary JSON path.")
    args = parser.parse_args()

    api_key = read_env_value(ENV_PATH, "SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY is empty in .env")

    caption_path = Path(args.caption) if args.caption else find_latest_caption(CAPTION_DIR)
    if not caption_path.exists():
        raise FileNotFoundError(f"Caption file not found: {caption_path}")

    input_text = caption_path.read_text(encoding="utf-8")
    if not input_text.strip():
        raise ValueError(f"Caption file is empty: {caption_path}")

    slang_dict_path = Path(args.slang_dict) if args.slang_dict else (DEFAULT_SLANG_DICT_PATH if DEFAULT_SLANG_DICT_PATH.exists() else None)
    slang_dict = load_slang_dictionary(slang_dict_path) if slang_dict_path else {"replacements": [], "preferred_phrases": [], "avoid_phrases": []}
    slang_guidance = build_slang_guidance(slang_dict)

    source_segments = load_source_segments(caption_path, input_text)
    source_audio = find_matching_source_audio(caption_path)
    source_segments = apply_timing_to_segments(source_segments, source_audio)
    translated_groups: list[dict[str, Any]] = []
    style_groups: list[dict[str, Any]] = []
    spell_groups: list[dict[str, Any]] = []
    output_segments: list[dict[str, Any]] = []

    for segment in source_segments:
        translated_text, translate_parts = translate_text_chunks(
            api_key=api_key,
            text=segment["source_text"],
            source_language_code=args.source_lang,
            target_language_code=args.target_lang,
        )
        if not translated_text:
            translated_text = segment["source_text"]

        final_segment_text = translated_text
        styled_text = ""
        style_parts: list[dict[str, Any]] = []
        if args.style == "colloquial":
            styled_text, style_parts = style_text_chunks(
                api_key=api_key,
                text=translated_text,
                model=args.chat_model,
                slang_guidance=slang_guidance,
            )
            if styled_text:
                final_segment_text = styled_text

        spell_parts: list[dict[str, Any]] = []
        if args.spell_fix:
            corrected_text, spell_parts = spell_fix_text_chunks(
                api_key=api_key,
                text=final_segment_text,
                model=args.chat_model,
            )
            if corrected_text:
                final_segment_text = corrected_text

        final_segment_text = apply_slang_replacements(final_segment_text, slang_dict).strip()
        if not final_segment_text:
            final_segment_text = translated_text

        translated_groups.append(
            {
                "segment": segment["index"],
                "source_text": segment["source_text"],
                "parts": translate_parts,
            }
        )
        if style_parts:
            style_groups.append(
                {
                    "segment": segment["index"],
                    "source_text": segment["source_text"],
                    "parts": style_parts,
                }
            )
        if spell_parts:
            spell_groups.append(
                {
                    "segment": segment["index"],
                    "source_text": segment["source_text"],
                    "parts": spell_parts,
                }
            )

        segment_record = dict(segment)
        segment_record["translated_text"] = translated_text
        if styled_text:
            segment_record["styled_text"] = styled_text
        segment_record["final_text"] = final_segment_text
        output_segments.append(segment_record)

    final_text = render_timed_segments_text(output_segments, "final_text")
    if not final_text.strip():
        final_text = render_timed_segments_text(output_segments, "translated_text")

    out_txt = caption_path.with_name(f"{caption_path.stem}.{args.target_lang}.txt")
    out_json = caption_path.with_name(f"{caption_path.stem}.{args.target_lang}.json")

    out_txt.write_text(final_text + "\n", encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "source_file": str(caption_path),
                "source_json": str(caption_path.with_suffix(".json")),
                "source_audio": str(source_audio) if source_audio else "",
                "style": args.style,
                "slang_dict": str(slang_dict_path) if slang_dict_path else "",
                "translated_text": final_text,
                "segments": output_segments,
                "translate_parts": translated_groups,
                "style_parts": style_groups,
                "spell_fix": args.spell_fix,
                "spell_parts": spell_groups,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("Source caption:", safe_console(str(caption_path)))
    print("Translated caption:", safe_console(str(out_txt)))
    print("Raw translation JSON:", safe_console(str(out_json)))
    print("\nTranslated Text:\n")
    print(safe_console(final_text))


if __name__ == "__main__":
    main()
