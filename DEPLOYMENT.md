# FuelTracks Voice Calling Agent — Production Deployment Guide

> Version: 1.0  
> Updated: June 2026

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Server Provisioning](#3-server-provisioning)
4. [Domain & TLS Setup](#4-domain--tls-setup)
5. [Environment Configuration](#5-environment-configuration)
6. [PostgreSQL Setup](#6-postgresql-setup)
7. [Deploying with Docker Compose](#7-deploying-with-docker-compose)
8. [Exotel Webhook Configuration](#8-exotel-webhook-configuration)
9. [VAD Backend Selection](#9-vad-backend-selection)
10. [Monitoring & Logging](#10-monitoring--logging)
11. [Scaling Horizontally](#11-scaling-horizontally)
12. [Backup & Recovery](#12-backup--recovery)
13. [Updating the Agent](#13-updating-the-agent)
14. [Security Checklist](#14-security-checklist)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Architecture Overview

```
Exotel ──WebSocket──► nginx (TLS termination)
                           │
                           ▼
                   FastAPI app (uvicorn)
                    ├─ VAD (energy/webrtc/silero)
                    ├─ Sarvam STT
                    ├─ Claude / Gemini LLM
                    └─ Sarvam TTS
                           │
                     PostgreSQL (asyncpg)
```

All traffic from Exotel arrives over a secure WebSocket (`wss://`).  
Nginx terminates TLS and proxies to the FastAPI app running inside Docker.

---

## 2. Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Ubuntu 22.04 LTS | Ubuntu 24.04 LTS |
| CPU | 2 vCPU | 4 vCPU |
| RAM | 2 GB | 4 GB |
| Disk | 20 GB SSD | 40 GB SSD |
| Docker | 24.x | latest |
| Docker Compose | v2.20+ | latest |
| Nginx | 1.24+ | latest |
| SSL certificate | Let's Encrypt | Let's Encrypt |

API keys required:
- Anthropic Claude (`ANTHROPIC_API_KEY`)
- Google Gemini (`GOOGLE_API_KEY`)
- Sarvam AI (`SARVAM_API_KEY`)
- Exotel (`EXOTEL_API_KEY`, `EXOTEL_API_TOKEN`, `EXOTEL_ACCOUNT_SID`)

---

## 3. Server Provisioning

### 3.1 Initial Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose plugin
sudo apt install docker-compose-plugin -y

# Install Nginx and Certbot
sudo apt install nginx certbot python3-certbot-nginx -y

# Clone the repository
git clone https://github.com/YOUR_ORG/voice-calling-agent.git /opt/fueltracks
cd /opt/fueltracks
```

---

## 4. Domain & TLS Setup

### 4.1 Point your domain to the server

Create an `A` record: `voice.fueltracks.in` → `<server-ip>`

### 4.2 Obtain a Let's Encrypt certificate

```bash
sudo certbot --nginx -d voice.fueltracks.in
```

### 4.3 Nginx configuration

Create `/etc/nginx/sites-available/fueltracks`:

```nginx
upstream fueltracks_app {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name voice.fueltracks.in;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name voice.fueltracks.in;

    ssl_certificate     /etc/letsencrypt/live/voice.fueltracks.in/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/voice.fueltracks.in/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    # WebSocket-specific settings
    proxy_read_timeout      3600s;
    proxy_send_timeout      3600s;
    proxy_connect_timeout   10s;

    location / {
        proxy_pass         http://fueltracks_app;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_buffering    off;
    }

    location /health {
        proxy_pass http://fueltracks_app/health;
        access_log off;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/fueltracks /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

---

## 5. Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` — fill in all values:

```dotenv
# ── Application ─────────────────────────────────────────────────────────────
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
SECRET_KEY=<generate: openssl rand -hex 32>

# ── Database ─────────────────────────────────────────────────────────────────
DATABASE_URL=postgresql://fueltracks:STRONG_PASSWORD@db:5432/fueltracks_prod

# ── AI APIs ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
SARVAM_API_KEY=...

# ── Exotel ───────────────────────────────────────────────────────────────────
EXOTEL_API_KEY=...
EXOTEL_API_TOKEN=...
EXOTEL_ACCOUNT_SID=...
EXOTEL_CALLER_ID=0XXXXXXXXXX
EXOTEL_APP_ID=...
BASE_URL=https://voice.fueltracks.in

# ── VAD ──────────────────────────────────────────────────────────────────────
# Options: energy | webrtc | silero
VAD_MODE=silero
```

> **Never commit `.env` to version control.**

---

## 6. PostgreSQL Setup

The Postgres service starts automatically with Docker Compose.  
You only need these one-time steps:

```bash
# Start only the database first
docker compose up -d db

# Wait for Postgres to be ready (usually < 10 seconds)
docker compose exec db pg_isready -U fueltracks

# Run initial schema migration (the app auto-creates tables on startup)
docker compose run --rm app python -c "
import asyncio
from src.storage.database import initialize_database
asyncio.run(initialize_database())
print('Database initialized.')
"
```

### 6.1 Connection pooling

The app uses **asyncpg** with a pool of 5–20 connections by default.  
Tune via `DATABASE_URL` params if needed:

```dotenv
DATABASE_URL=postgresql://user:pass@host:5432/db?min_size=5&max_size=20
```

---

## 7. Deploying with Docker Compose

```bash
# Pull latest images (for ghcr.io-based deployments)
docker compose pull

# Build locally if using local code
docker compose build

# Start all services in the background
docker compose up -d

# View live logs
docker compose logs -f app

# Confirm the app is healthy
curl https://voice.fueltracks.in/health
```

Expected health response:
```json
{"status":"healthy","version":"1.0.0","database":"connected"}
```

---

## 8. Exotel Webhook Configuration

In the **Exotel Dashboard → Apps → Your App → Passthru**:

| Setting | Value |
|---------|-------|
| **Applet URL** | `https://voice.fueltracks.in/exotel/voice` |
| **Method** | `POST` |
| **WebSocket URL (streaming)** | `wss://voice.fueltracks.in/ws/{call_sid}` |
| **Audio Format** | `pcm_16bit_8khz` |
| **Status Callback URL** | `https://voice.fueltracks.in/exotel/status` |

---

## 9. VAD Backend Selection

| Backend | Accuracy | CPU | RAM | Latency | Notes |
|---------|----------|-----|-----|---------|-------|
| `energy` | Low | Minimal | None | ~0ms | Always works, use for CI/testing |
| `webrtc` | Medium | Very low | ~1MB | ~1ms | Install: `pip install webrtcvad-wheels` |
| `silero` | High | Low | ~40MB | ~2ms | Best for Indian languages |

Set `VAD_MODE=silero` for production.  
`webrtcvad-wheels` is pre-installed in the Dockerfile.

---

## 10. Monitoring & Logging

### 10.1 Structured logs

Logs are written to **stdout** in JSON-compatible format.  
View with:

```bash
docker compose logs -f app | jq .
```

Key log fields:
- `STT latency: Xms` — speech-to-text round-trip
- `LLM first token latency: Xms` — time to first token
- `TTS latency: Xms` — TTS synthesis time

### 10.2 Latency metrics in database

Per-call latency is persisted in the `call_sessions` table:

```sql
SELECT
    call_sid,
    stt_latency_ms,
    llm_latency_ms,
    tts_latency_ms,
    duration_seconds
FROM call_sessions
ORDER BY started_at DESC
LIMIT 20;
```

### 10.3 Optional: Prometheus + Grafana

Add a `prometheus` service to `docker-compose.yml` and mount a `prometheus.yml` scraping `http://app:8000/metrics`.  
The app exposes a `/metrics` endpoint when `ENABLE_METRICS=true`.

---

## 11. Scaling Horizontally

For > 50 concurrent calls, run multiple app replicas behind Nginx:

```yaml
# docker-compose.yml
services:
  app:
    deploy:
      replicas: 3
```

> **Important**: WebSocket sessions are pinned per Exotel call. Use `ip_hash` in Nginx upstream or a sticky-session load balancer so reconnections land on the same replica.

---

## 12. Backup & Recovery

### 12.1 Automated daily backup

```bash
# Add to crontab: crontab -e
0 2 * * * docker compose -f /opt/fueltracks/docker-compose.yml exec -T db \
  pg_dump -U fueltracks fueltracks_prod | \
  gzip > /backups/fueltracks_$(date +\%Y\%m\%d).sql.gz
```

### 12.2 Restore from backup

```bash
gunzip -c /backups/fueltracks_20260626.sql.gz | \
  docker compose exec -T db psql -U fueltracks fueltracks_prod
```

---

## 13. Updating the Agent

```bash
cd /opt/fueltracks

# Pull latest code
git pull origin main

# Rebuild and restart (zero-downtime with replicas)
docker compose build app
docker compose up -d --no-deps --scale app=2 app  # bring up new replicas
sleep 10
docker compose up -d --no-deps --scale app=1 app  # remove old replica
```

---

## 14. Security Checklist

- [ ] `SECRET_KEY` is a random 32-byte hex string (`openssl rand -hex 32`)
- [ ] `.env` is not committed to git (`.gitignore` includes it)
- [ ] Postgres password is strong (> 20 chars, mixed chars)
- [ ] Nginx has TLSv1.2+ only (no TLSv1.0/1.1)
- [ ] Firewall: only ports 80, 443, 22 open to the public
- [ ] Postgres port 5432 is **NOT** exposed to the public (only internal Docker network)
- [ ] API keys are stored only in `.env`, never hard-coded
- [ ] Let's Encrypt cert auto-renewal is configured (`certbot renew --dry-run`)
- [ ] `LOG_LEVEL=WARNING` or `INFO` in production (never `DEBUG`)

---

## 15. Troubleshooting

### App won't start

```bash
docker compose logs app | tail -50
```

Common causes:
- Missing env variable → check `.env` against `.env.example`
- Database not ready → `docker compose logs db`
- Port 8000 already in use → `lsof -i :8000`

### WebSocket connection drops

- Check Nginx `proxy_read_timeout` is ≥ 3600s
- Verify Exotel is hitting the `wss://` URL (not `ws://`)
- Check `docker compose logs app` for `WebSocket disconnected` messages

### High STT latency

- Sarvam API may be experiencing slowness — check their status page
- Switch `VAD_MODE=energy` temporarily to rule out VAD overhead
- Ensure audio chunks are arriving every 20ms (check Exotel streaming docs)

### Silero VAD not loading

```bash
docker compose exec app python -c "
from src.audio.vad import VoiceActivityDetector
v = VoiceActivityDetector()
print(v._backend)
"
```

If it falls back to `energy`, install: `pip install silero-vad onnxruntime`

### Database connection pool exhausted

Increase pool size in `DATABASE_URL`:
```dotenv
DATABASE_URL=postgresql://user:pass@host:5432/db?min_size=5&max_size=30
```

Or scale the database vertically / add a PgBouncer sidecar.

---

*For support, contact the FuelTracks engineering team at info@fueltracks.in*
