# Multilingual Voice Calling Agent

A production-ready voice calling agent in Python built for inbound and outbound calling use cases, integrating Exotel telephony (with bidirectional audio streaming over WebSockets), Sarvam AI (for code-mixed Speech-to-Text and Text-to-Speech), and Groq (Llama 3 models) for fast conversational dialog logic.

---

## Architecture Overview

The system processes real-time bidirectional telephony audio over WebSockets:

1. **Telephony Layer (`src/telephony/`)**: FastAPI endpoints handle Exotel webhooks. When a call connects, the app returns a WebSocket stream URL (`/ws/media?call_sid=...`). Bidirectional audio packets are exchanged inside JSON frames.
2. **Audio & VAD Layer (`src/audio/`)**: A local energy-based Voice Activity Detector (VAD) monitors 16-bit mono PCM chunks to identify start/end of speech.
3. **STT & Language Profiling (`src/stt/`)**: Finished audio utterances are packaged as WAV and sent to Sarvam AI (`saaras:v3`) with `language_code="unknown"`. The STT output detects language, code-mix tokens, and updates a rolling `LanguageProfile` per session.
4. **Conversation Manager (`src/conversation/`)**: Maintains dialog history, dynamically routes inbound calls (`support` vs `lead_followup` vs `dealer_recruitment`), compiles runtime system prompts based on rolling language parameters, and calls Groq API (llama-3.3-70b-versatile primary, llama-3.1-8b-instant fallback).
5. **TTS Layer (`src/tts/`)**: Synthesizes agent response text to PCM audio in the customer's primary language via Sarvam AI (`bulbul:v3`).
6. **Barge-in handling (`src/orchestrator/`)**: If the caller interrupts while the agent is speaking, the pipeline cancels the audio playback task, sends a `clear` command to Exotel's playback buffer, and shifts instantly to listening.
7. **Storage (`src/storage/`)**: Tracks call outcomes, durations, STT duration (seconds), TTS size (characters), LLM tokens, and transcripts in a local SQLite database.

---

## Installation & Setup

1. **Clone the project & initialize requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables:**
   Copy the example environment template and fill in your API credentials:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and configure:
   - `EXOTEL_ACCOUNT_SID`, `EXOTEL_API_KEY`, `EXOTEL_API_TOKEN`
   - `SARVAM_API_KEY`
   - `GROQ_API_KEY` (get a free key at https://console.groq.com/keys)
   - `WEBSOCKET_URL` (Set to your public domain/ngrok URL in production, e.g. `wss://<subdomain>.ngrok-free.app/ws/media`)

---

## Running Local Simulations

### 1. Interactive CLI Simulator
Test the agent dialogue, database logging, dynamic prompt compiling, and cost metric tracking directly in your terminal without initiating any real telephony calls:
```bash
python scripts/local_dev_test.py
```
*Tip: You can simulate language changes by prefixing utterances with `[hi]` or `[te]`, e.g., `[te] GPS tracker install cheyyadam ela?`.*

### 2. Run Automated Unit Tests
To verify all internal layers (VAD, SQLite, Language Profile, and Inbound Routing) in isolation:
```bash
python -m unittest tests/test_call_simulation.py
```

---

## Deploying the REST API

To launch the FastAPI server locally:
```bash
python src/api/main.py
```
- **Webhooks URL for Exotel**: `https://<your-domain>/voice/inbound`
- **QA Call logs & Transcripts**: You can query the database transcript and billing logs for review using the GET endpoint:
  `GET /calls/{call_sid}`
