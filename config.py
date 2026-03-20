"""
config.py — Central configuration loader.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    _overrides: dict = {}

    def _get(self, key: str, default: str) -> str:
        if key in self._overrides:
            return str(self._overrides[key])
        return os.environ.get(key, default)

    @property
    def GEMINI_API_KEY(self) -> str:
        if "GEMINI_API_KEY" in self._overrides:
            return str(self._overrides["GEMINI_API_KEY"])
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")

    @GEMINI_API_KEY.setter
    def GEMINI_API_KEY(self, value: str):
        self._overrides["GEMINI_API_KEY"] = value

    @property
    def EMBEDDING_MODEL(self) -> str:
        return self._get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    @property
    def LLM_MODEL(self) -> str:
        return self._get("LLM_MODEL", "gemini-2.0-flash")

    @property
    def CACHE_MAX_SIZE(self) -> int:
        return int(self._get("CACHE_MAX_SIZE", "128"))

    @property
    def DEBATE_ROUNDS(self) -> int:
        return int(self._get("DEBATE_ROUNDS", "2"))


settings = Settings()