"""
config.py
---------
Central place for settings and folder paths. Everything is resolved relative
to the project folder, so the app works no matter where it is launched from.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
PANELS_DIR = STATIC_DIR / "panels"
EXPORTS_DIR = STATIC_DIR / "exports"
FONTS_DIR = STATIC_DIR / "fonts"

# Load API keys and settings from the .env file in the project folder
load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    """Read an environment variable, treating blanks and placeholders as unset."""
    value = os.getenv(name, "").strip()
    if not value or value.startswith("your_"):
        return default
    return value


GEMINI_API_KEY = _env("GEMINI_API_KEY")
HF_API_KEY = _env("HF_API_KEY")

# Gemini models. If the first one fails (quota reached, model retired, ...)
# the app automatically tries the next one in the list.
OUTLINE_MODELS = [m.strip() for m in _env("GEMINI_OUTLINE_MODELS", "gemini-3.5-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.8-flash,gemini-3-flash-preview,gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",") if m.strip()]
STORY_MODELS = [m.strip() for m in _env("GEMINI_STORY_MODELS", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash,gemini-3-flash-preview,gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",") if m.strip()]

HF_IMAGE_MODEL = _env("HF_IMAGE_MODEL", "stabilityai/stable-diffusion-3-medium-diffusers")

# Image services to try for each panel, in order (see image_generator.py)
IMAGE_PROVIDERS = [p.strip().lower() for p in _env("IMAGE_PROVIDERS", "flux_space,huggingface,pollinations,horde").split(",") if p.strip()]

NUM_PANELS = max(1, min(int(_env("NUM_PANELS", "6")), 10))
# How many panel images are requested from Hugging Face at the same time
IMAGE_WORKERS = max(1, min(int(_env("IMAGE_WORKERS", "3")), 6))

for folder in (PANELS_DIR, EXPORTS_DIR, FONTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def static_url(path: Path) -> str:
    """Turn a file inside static/ into the URL the browser can load it from."""
    return "/static/" + path.relative_to(STATIC_DIR).as_posix()


def missing_keys() -> list:
    """Names of required API keys that have not been set in .env."""
    return [name for name, value in (("GEMINI_API_KEY", GEMINI_API_KEY), ("HF_API_KEY", HF_API_KEY)) if not value]
