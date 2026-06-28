# Week 1 — GitHub Webhook Server & Project Foundation

## What Week 1 Built

The skeleton of the entire system. Everything else in weeks 2–8 plugs into this foundation.

```
GitHub PR opened/updated
        ↓
  GitHub sends POST to /webhook
        ↓
  FastAPI server receives payload
        ↓
  Validates HMAC signature (security)
        ↓
  Queues review job (background task)
        ↓
  Returns 200 OK immediately (GitHub requires fast response)
```

---

## The Webhook Server (`main.py`)

Built with **FastAPI** + **uvicorn**. Why FastAPI?
- Automatic request validation via Pydantic
- Async support — webhook must return 200 in <10 seconds or GitHub retries
- Built-in OpenAPI docs at `/docs`

### Key endpoint

```python
@app.post("/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    # 1. Verify HMAC-SHA256 signature from GitHub
    # 2. Parse pull_request event
    # 3. Kick off review in background
    # 4. Return 200 immediately
```

### HMAC signature verification

GitHub signs every webhook payload with your secret. Without this check, anyone could send fake events to your server.

```python
import hmac, hashlib

def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

---

## GitHub Integration (`PyGithub`)

```python
from github import Github

g = Github(GITHUB_TOKEN)
repo = g.get_repo("owner/repo")
pr   = repo.get_pull(pr_number)
```

Used to fetch PR metadata, diff, and later post reviews back.

---

## Local Tunnel (ngrok)

GitHub can't send webhooks to `localhost`. ngrok creates a public HTTPS URL that forwards to your local server.

```bash
ngrok http 8000
# → https://abc123.ngrok.io → localhost:8000
```

Set `https://abc123.ngrok.io/webhook` as your GitHub webhook URL during development.

---

## Environment Config (`.env`)

```
GITHUB_TOKEN=ghp_...
GITHUB_WEBHOOK_SECRET=your_secret
GEMINI_API_KEY=AIza...
```

Loaded via `python-dotenv`. Never hardcoded. Never committed. `.gitignore` excludes `.env`.

---

## File Summary

```
main.py          ← FastAPI server, /webhook endpoint, HMAC verification
config.py        ← Load and validate all environment variables
.env.example     ← Template showing required variables (safe to commit)
.env             ← Actual secrets (NEVER commit)
requirements.txt ← PyGithub, fastapi, uvicorn, python-dotenv, tree-sitter
```

---

## Tests (`tests/test_week1.py`)

- Webhook accepts valid HMAC signatures
- Webhook rejects invalid signatures (401)
- Non-PR events return 200 but do nothing
- Config validation raises on missing env vars
