"""Real-time Form-Filling Assistant for Thai customer conversations.

Always-listening mode: captures speech via VAD, transcribes with Typhoon ASR,
extracts form fields with Typhoon LLM, and displays a live-updating form.

Usage:
    python scripts/form_assistant.py
    python scripts/form_assistant.py --api-key YOUR_KEY
    python scripts/form_assistant.py --output forms/customer.json

Press q+ENTER to finish and save the form.
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import sounddevice as sd
from pydantic import BaseModel, Field

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024

# Extract every N speech segments
EXTRACT_EVERY_N = 2


# ── Form Schema ──────────────────────────────────────────────────

class CustomerForm(BaseModel):
    """แบบฟอร์มข้อมูลลูกค้า — Thai customer information form."""
    full_name: Optional[str] = Field(None, description="ชื่อ-นามสกุล")
    phone: Optional[str] = Field(None, description="เบอร์โทรศัพท์")
    email: Optional[str] = Field(None, description="อีเมล")
    address: Optional[str] = Field(None, description="ที่อยู่")
    date_of_birth: Optional[str] = Field(None, description="วันเกิด")
    occupation: Optional[str] = Field(None, description="อาชีพ")
    company: Optional[str] = Field(None, description="บริษัท/สถานที่ทำงาน")
    notes: Optional[str] = Field(None, description="หมายเหตุเพิ่มเติม")


FORM_LABELS = {
    "full_name": "ชื่อ-นามสกุล",
    "phone": "เบอร์โทร",
    "email": "อีเมล",
    "address": "ที่อยู่",
    "date_of_birth": "วันเกิด",
    "occupation": "อาชีพ",
    "company": "บริษัท",
    "notes": "หมายเหตุ",
}

FIELD_ORDER = [
    "full_name", "phone", "email", "address",
    "date_of_birth", "occupation", "company", "notes",
]


# ── VAD (simple energy-based) ────────────────────────────────────

class SimpleVAD:
    """Energy-based voice activity detector for continuous listening."""

    def __init__(
        self,
        energy_threshold: float = 0.015,
        silence_duration: float = 1.0,
        min_speech_duration: float = 0.5,
    ):
        self._energy_threshold = energy_threshold
        self._silence_duration = silence_duration
        self._min_speech_duration = min_speech_duration

        self._speech_frames: List[np.ndarray] = []
        self._silence_frames = 0
        self._is_speaking = False
        self._speech_start_time = 0.0

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a frame. Returns complete speech segment or None."""
        rms = np.sqrt(np.mean(frame ** 2))
        is_speech = rms >= self._energy_threshold

        if is_speech:
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start_time = time.perf_counter()
                self._speech_frames = []
            self._speech_frames.append(frame)
            self._silence_frames = 0
        else:
            if self._is_speaking:
                self._speech_frames.append(frame)
                self._silence_frames += 1
                silence_sec = self._silence_frames * len(frame) / SAMPLE_RATE
                if silence_sec >= self._silence_duration:
                    self._is_speaking = False
                    audio = np.concatenate(self._speech_frames)
                    duration = len(audio) / SAMPLE_RATE
                    if duration >= self._min_speech_duration:
                        return audio
                    self._speech_frames = []
        return None


# ── Form Display ─────────────────────────────────────────────────

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def draw_form(known_fields: Dict[str, str], segments: int, listening: bool = True):
    """Draw the live form in terminal."""
    lines = []
    lines.append("╔══════════════════════════════════════════════════╗")
    lines.append("║       📋 ข้อมูลลูกค้า (Customer Information)        ║")
    lines.append("╠══════════════════════════════════════════════════╣")

    filled = 0
    total = len(FIELD_ORDER)

    for field_key in FIELD_ORDER:
        label = FORM_LABELS[field_key]
        value = known_fields.get(field_key)
        if value:
            filled += 1
            status = "✅"
            display_val = value
        else:
            status = "  "
            display_val = "___"
        lines.append(f"║  {status} {label:12s} {display_val:24s} ║")

    lines.append("╠══════════════════════════════════════════════════╣")

    if listening:
        lines.append(f"║  🎧 กำลังฟัง... ({filled}/{total} fields, {segments} segments)    ║")
    else:
        lines.append(f"║  ⏹️  หยุดแล้ว ({filled}/{total} fields filled)               ║")

    lines.append("╚══════════════════════════════════════════════════╝")
    lines.append("")
    lines.append("  Press q+ENTER to finish and save")

    clear_screen()
    print("\n".join(lines))


# ── Audio Helpers ────────────────────────────────────────────────

def save_wav(audio: np.ndarray, path: str):
    """Save float32 audio to WAV file."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def extract_text(result: dict) -> str:
    """Extract text from typhoon_asr result."""
    raw = result.get("text", "")
    if hasattr(raw, "text"):
        return raw.text
    return str(raw)


def segment_thai(text: str) -> str:
    """Segment Thai words with spaces."""
    if not text:
        return text
    try:
        from pythainlp.tokenize import word_tokenize
        return " ".join(word_tokenize(text, engine="newmm"))
    except ImportError:
        return text


# ── LLM Extraction ───────────────────────────────────────────────

EXTRACT_SYSTEM_PROMPT = """\
You are extracting customer information from a Thai conversation transcript.
Fill in any fields you can identify from the transcript below.

Rules:
- Return ONLY valid JSON with these fields: full_name, phone, email, address, date_of_birth, occupation, company, notes
- Use null for fields you cannot determine
- Extract names, phone numbers, emails, addresses, dates, jobs, companies from the conversation
- Phone numbers: include dashes if present (e.g. "081-234-5678")
- If a field was already known and the transcript has new/corrected info, use the new value
- Respond ONLY with the JSON object, nothing else
"""

EXTRACT_USER_TEMPLATE = """\
Current known fields:
{known_fields}

Conversation transcript:
{transcript}

Updated fields (JSON):
"""


def extract_form_fields(
    client,
    model: str,
    transcript: str,
    known_fields: Dict[str, str],
) -> Dict[str, str]:
    """Call LLM to extract form fields from transcript."""
    import re

    known_json = json.dumps(known_fields, ensure_ascii=False, indent=2)
    user_msg = EXTRACT_USER_TEMPLATE.format(
        known_fields=known_json,
        transcript=transcript,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=256,
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON from response
    # Try direct parse
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting from ```json ... ``` blocks
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                return known_fields
        else:
            # Try finding first { ... }
            match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return known_fields
            else:
                return known_fields

    # Merge: only overwrite with non-null values
    updated = dict(known_fields)
    for key, value in data.items():
        if key in FORM_LABELS and value is not None and value != "":
            updated[key] = str(value)
    return updated


# ── Main Loop ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time Form-Filling Assistant (Thai customer conversations)"
    )
    parser.add_argument("--api-key", default=None, help="Typhoon API key (or set TYPHOON_API_KEY)")
    parser.add_argument(
        "--model", default="typhoon-v2.5-30b-a3b-instruct",
        help="Typhoon LLM model ID",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file path (default: forms/YYYY-MM-DD_HHMMSS.json)",
    )
    parser.add_argument(
        "--energy-threshold", type=float, default=0.015,
        help="VAD energy threshold (default: 0.015)",
    )
    args = parser.parse_args()

    # ── Setup API client ──
    from openai import OpenAI

    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get("TYPHOON_API_KEY", "")
    if not api_key:
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.strip().startswith("TYPHOON_API_KEY="):
                        api_key = line.strip().split("=", 1)[1]
    if not api_key:
        print("❌ No API key. Use --api-key or set TYPHOON_API_KEY")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.opentyphoon.ai/v1")

    # ── State ──
    known_fields: Dict[str, str] = {}
    full_transcript: List[str] = []
    segment_count = 0
    segments_since_extract = 0

    # ── Audio capture ──
    audio_frames: List[np.ndarray] = []
    audio_lock = threading.Lock()
    stop_event = threading.Event()

    def capture_audio():
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS,
            blocksize=BLOCK_SIZE, dtype="float32",
        ) as stream:
            while not stop_event.is_set():
                block, _ = stream.read(BLOCK_SIZE)
                with audio_lock:
                    audio_frames.append(block.flatten())

    # ── Start capture ──
    print("🌪️  Form-Filling Assistant")
    print("📋 Listening for customer conversation...")
    print("   Press q+ENTER to finish and save\n")

    # ── Pre-load ASR model (avoid reloading on each segment) ──
    print("  🔄 Loading Typhoon ASR model...")
    from typhoon_asr import transcribe as _transcribe
    # Warm up with silent audio so model loads once
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
    clipped = np.clip(silent, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(CHANNELS); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    _transcribe(tmp_path, device="auto")
    os.unlink(tmp_path)
    print("  ✅ ASR model loaded\n")

    capturer = threading.Thread(target=capture_audio, daemon=True)
    capturer.start()

    vad = SimpleVAD(
        energy_threshold=args.energy_threshold,
        silence_duration=1.0,
        min_speech_duration=0.5,
    )

    draw_form(known_fields, 0)

    # ── Check for quit input (non-blocking) ──
    quit_event = threading.Event()

    def check_quit():
        while not stop_event.is_set():
            try:
                line = input()
                if line.strip().lower() == "q":
                    quit_event.set()
                    return
            except EOFError:
                quit_event.set()
                return

    quit_thread = threading.Thread(target=check_quit, daemon=True)
    quit_thread.start()

    # ── Main processing loop ──
    frame_idx = 0
    try:
        while not quit_event.is_set():
            # Get latest frames
            with audio_lock:
                new_frames = audio_frames[frame_idx:]
                frame_idx = len(audio_frames)

            if not new_frames:
                time.sleep(0.05)
                continue

            # Process each frame through VAD
            for frame in new_frames:
                speech = vad.process_frame(frame)
                if speech is None:
                    continue

                segment_count += 1
                segments_since_extract += 1
                duration = len(speech) / SAMPLE_RATE

                # Transcribe
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    save_wav(speech, tmp_path)
                    result = _transcribe(tmp_path, device="auto")
                    text = extract_text(result)
                finally:
                    os.unlink(tmp_path)

                if not text:
                    draw_form(known_fields, segment_count)
                    continue

                display = segment_thai(text)
                full_transcript.append(text)

                # Redraw with new transcript line
                draw_form(known_fields, segment_count)
                print(f"\n  📝 [{segment_count}] {display}")

                # Extract form fields every N segments
                if segments_since_extract >= EXTRACT_EVERY_N:
                    print(f"\n  🔄 Extracting form fields...")
                    try:
                        transcript_text = "\n".join(
                            f"[{i+1}] {t}" for i, t in enumerate(full_transcript)
                        )
                        known_fields = extract_form_fields(
                            client, args.model, transcript_text, known_fields,
                        )
                        segments_since_extract = 0
                    except Exception as e:
                        print(f"  ⚠️ Extraction error: {e}")

                    draw_form(known_fields, segment_count)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()

    # ── Final extraction (catch anything missed) ──
    if full_transcript:
        print("\n  🔄 Final extraction...")
        transcript_text = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(full_transcript))
        try:
            known_fields = extract_form_fields(
                client, args.model, transcript_text, known_fields,
            )
        except Exception as e:
            print(f"  ⚠️ Final extraction error: {e}")

    # ── Save form ──
    draw_form(known_fields, segment_count, listening=False)

    # Print full transcript
    print("\n  📝 Full Transcript:")
    for i, t in enumerate(full_transcript):
        print(f"    [{i+1}] {segment_thai(t)}")

    # Save
    output_path = args.output
    if not output_path:
        forms_dir = Path("forms")
        forms_dir.mkdir(exist_ok=True)
        name = known_fields.get("full_name", "unknown").replace(" ", "_")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_path = str(forms_dir / f"{timestamp}_{name}.json")

    output_data = {
        "form": known_fields,
        "transcript": full_transcript,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "segments": segment_count,
            "model": args.model,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    filled = sum(1 for v in known_fields.values() if v)
    total = len(FIELD_ORDER)
    print(f"\n  💾 Form saved to: {output_path}")
    print(f"  ✅ {filled}/{total} fields filled\n")


if __name__ == "__main__":
    main()
