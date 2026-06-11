# 📋 Real-time Thai Form-Filling Assistant

POC สำหรับกรอกแบบฟอร์มข้อมูลลูกค้าอัตโนมัติจากการสนทนาภาษาไทยแบบเรียลไทม์

Real-time form-filling assistant that listens to Thai conversations, transcribes speech, and automatically extracts customer information into a structured form.

## Features

- 🎤 **Always-listening** — VAD auto-detects speech, no button press needed
- 🌪️ **Typhoon ASR** — local Thai speech-to-text (FastConformer, 114M params)
- 🤖 **Typhoon v2.5 API** — LLM extracts form fields from conversation
- 📝 **Thai word segmentation** — auto-spaces Thai words via pythainlp
- 📋 **Live form display** — terminal UI updates in real-time as fields are filled
- 💾 **JSON export** — saves completed forms with full transcript

## Architecture

```
Mic → VAD → Typhoon ASR → Accumulate Transcript → Typhoon LLM Extract → Live Form
```

## Prerequisites

- Python 3.12+
- macOS (Apple Silicon recommended) or Linux
- Typhoon API key ([get one free](https://opentyphoon.ai))
- Microphone

## Install

```bash
# Clone
git clone https://github.com/Napattarapong/PoC_realtime-fill.git
cd PoC_realtime-fill

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install typhoon-asr openai sounddevice pythainlp psutil

# Set API key
echo 'TYPHOON_API_KEY=your-key-here' > .env
```

## Usage

```bash
python scripts/form_assistant.py
```

Then have a conversation with customer information. The assistant will:

1. **Listen** — auto-detects speech via VAD
2. **Transcribe** — Thai ASR with word segmentation
3. **Extract** — every 2 speech segments, LLM fills form fields
4. **Display** — live terminal form updates

### Example session

```
╔══════════════════════════════════════════════════╗
║       📋 ข้อมูลลูกค้า (Customer Information)        ║
╠══════════════════════════════════════════════════╣
║  ✅ ชื่อ-นามสกุล   สมชาย ใจดี                     ║
║  ✅ เบอร์โทร       081-234-5678                   ║
║    อีเมล          ___                            ║
║    ที่อยู่         ___                            ║
║  ✅ อาชีพ         วิศวกร                         ║
║  ✅ บริษัท         บริษัท ABC                      ║
║    หมายเหตุ       ___                            ║
╠══════════════════════════════════════════════════╣
║  🎧 กำลังฟัง... (4/7 fields, 6 segments)         ║
╚══════════════════════════════════════════════════╝
```

Press **q + ENTER** to finish and save the form.

### Output

Forms are saved to `forms/` as JSON:

```json
{
  "form": {
    "full_name": "สมชาย ใจดี",
    "phone": "081-234-5678",
    "occupation": "วิศวกร",
    "company": "บริษัท ABC จำกัด"
  },
  "transcript": ["..."],
  "metadata": {
    "timestamp": "2026-06-09T17:00:00",
    "segments": 8,
    "model": "typhoon-v2.5-30b-a3b-instruct"
  }
}
```

## CLI Options

```
python scripts/form_assistant.py [OPTIONS]

Options:
  --api-key KEY         Typhoon API key (or set TYPHOON_API_KEY env var)
  --model MODEL         LLM model ID (default: typhoon-v2.5-30b-a3b-instruct)
  --output PATH         Output JSON file path (default: forms/TIMESTAMP_NAME.json)
  --energy-threshold N  VAD sensitivity (default: 0.015)
```

## Form Fields

| Field | Thai Label | Description |
|-------|-----------|-------------|
| `full_name` | ชื่อ-นามสกุล | Full name |
| `phone` | เบอร์โทร | Phone number |
| `email` | อีเมล | Email address |
| `address` | ที่อยู่ | Address |
| `date_of_birth` | วันเกิด | Date of birth |
| `occupation` | อาชีพ | Occupation |
| `company` | บริษัท | Company |
| `notes` | หมายเหตุ | Additional notes |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ASR | [Typhoon ASR Real-Time](https://github.com/scb-10x/typhoon-asr) (NeMo FastConformer) |
| LLM | [Typhoon v2.5 API](https://opentyphoon.ai) (OpenAI-compatible) |
| VAD | Energy-based voice activity detector |
| Thai NLP | [pythainlp](https://github.com/PyThaiNLP/pythainlp) |
| Audio | [sounddevice](https://python-sounddevice.readthedocs.io/) (PortAudio) |

## Project Structure

```
├── scripts/
│   └── form_assistant.py       # Main entry point
├── src/
│   ├── audio/                  # Mic capture, ring buffer, resampling
│   ├── llm/                    # Typhoon API engine
│   ├── pipeline/               # Orchestrator (STT → LLM flow)
│   ├── stt/                    # STT backends (Typhoon ASR, Whisper)
│   └── vad/                    # Voice activity detection
├── forms/                      # Saved form JSONs (gitignored)
├── .env                        # API key (gitignored)
└── pyproject.toml
```

## License

Apache-2.0
