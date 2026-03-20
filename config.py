"""
config.py — Central configuration loader.
Accepts GEMINI_API_KEY or GOOGLE_API_KEY interchangeably.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Accept either key name — GEMINI_API_KEY takes priority
    GEMINI_API_KEY: str = (
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    )
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "128"))
    DEBATE_ROUNDS: int = int(os.getenv("DEBATE_ROUNDS", "2"))
    # Set USE_MOCK_LLM=true to run without any API key
    USE_MOCK_LLM: bool = os.getenv("USE_MOCK_LLM", "false").lower() == "true"


settings = Settings()
