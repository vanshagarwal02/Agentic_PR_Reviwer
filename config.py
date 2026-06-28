# Central configuration — import this everywhere instead of os.getenv() scattered around
import os
from dotenv import load_dotenv

# load_dotenv() reads the .env file and puts everything into os.environ
# This must run before anything else imports config values
load_dotenv()


class Config:
    # ── GitHub ────────────────────────────────────────────────────────────────
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GITHUB_WEBHOOK_SECRET: str = os.getenv("GITHUB_WEBHOOK_SECRET", "")

    # ── Gemini (Week 5) ───────────────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # How many seconds to sleep between Gemini API calls during bulk runs.
    # This is intentional — free tier allows ~15 requests/minute for Flash.
    GEMINI_SLEEP_BETWEEN_CALLS: float = 4.0

    # ── Paths ─────────────────────────────────────────────────────────────────
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")     # trained ML models
    DATA_DIR: str = os.path.join(BASE_DIR, "data")         # datasets
    LOGS_DIR: str = os.path.join(BASE_DIR, "logs")         # agent decision logs

    # ── Diff Parser ───────────────────────────────────────────────────────────
    # How many lines above/below a change to include as context.
    # This context is sent to the LLM so it understands the surrounding code.
    CONTEXT_LINES: int = 5

    # ── Database (Week 7) ─────────────────────────────────────────────────────
    # Defaults to SQLite (zero setup). Swap to PostgreSQL via .env:
    #   DATABASE_URL=postgresql://user:pass@localhost/ai_reviewer
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./reviews.db")


config = Config()

# Create directories if they don't exist yet
for d in [config.MODELS_DIR, config.DATA_DIR, config.LOGS_DIR]:
    os.makedirs(d, exist_ok=True)
