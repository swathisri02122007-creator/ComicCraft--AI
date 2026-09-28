"""
image_generator.py
-------------------
Step 3 of the pipeline: draws each panel with free AI image services. Panels
are generated in parallel, and each panel tries the services in order
(IMAGE_PROVIDERS in .env) until one works:

    flux_space    FLUX.1-schnell on a free Hugging Face Space (best quality, fast)
    huggingface   Stable Diffusion 3 via Hugging Face Inference (monthly free credits)
    pollinations  Pollinations.ai (free, no key, one image at a time, small watermark)
    horde         AI Horde (free, volunteer-run, no key)

If every service fails, a placeholder card is drawn so the comic still completes.
"""

import io
import logging
import os
import re
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageFont

from app.config import (FONTS_DIR, HF_API_KEY, HF_IMAGE_MODEL, IMAGE_PROVIDERS, IMAGE_WORKERS,
                        PANELS_DIR, static_url)

log = logging.getLogger("comiccraft")

NEGATIVE_PROMPT = "text, words, letters, speech bubble, watermark, signature, blurry, deformed, extra limbs"


class ProviderUnavailable(RuntimeError):
    """The service can't be used for a while (quota used up, bad key, ...)."""

    def __init__(self, message: str, retry_after: float = float("inf")):
        super().__init__(message)
        self.retry_after = retry_after


# ---------------------------------------------------------------------------
# FLUX.1-schnell on a Hugging Face Space (uses the free daily GPU allowance,
# not the monthly Inference credits; works without a token too)
# ---------------------------------------------------------------------------
FLUX_SPACE = "black-forest-labs/FLUX.1-schnell"
_flux_client = None
_flux_lock = threading.Lock()


def _flux_space(prompt: str) -> bytes:
    global _flux_client
    from gradio_client import Client  # imported here so the app starts fast

    with _flux_lock:
        if _flux_client is None:
            _flux_client = Client(FLUX_SPACE, token=HF_API_KEY or None, verbose=False)
    try:
        result = _flux_client.predict(prompt, 0, True, 1024, 1024, 4, api_name="/infer")
    except Exception as e:
        if "quota" in str(e).lower():
            raise ProviderUnavailable("free daily FLUX allowance used up", retry_after=3600) from e
        raise
    path = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(path, dict):
        path = path.get("path")
    with open(path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Hugging Face Inference API (Stable Diffusion 3)
# ---------------------------------------------------------------------------
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"


def _huggingface(prompt: str) -> bytes:
    if not HF_API_KEY:
        raise ProviderUnavailable("HF_API_KEY is missing from .env")

    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Accept": "image/png"}
    payload = {"inputs": prompt, "parameters": {"negative_prompt": NEGATIVE_PROMPT}}

    last_error = "unknown error"
    for attempt in range(4):
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)
        except requests.RequestException as e:
            last_error = f"network error: {e}"
            time.sleep(3)
            continue

        if response.status_code == 200 and response.headers.get("content-type", "").startswith("image/"):
            return response.content
        if response.status_code == 429 or response.status_code >= 500:
            # 503 = model loading, 429 = too many requests at once, other 5xx = temporary server issue
            last_error = f"Hugging Face busy ({response.status_code})"
            time.sleep(8 * (attempt + 1))
            continue
        if response.status_code in (401, 403):
            raise ProviderUnavailable("Hugging Face rejected HF_API_KEY")
        if response.status_code == 402:
            raise ProviderUnavailable("Hugging Face monthly credits used up", retry_after=24 * 3600)
        raise RuntimeError(f"Hugging Face error {response.status_code}: {response.text[:200]}")

    raise RuntimeError(last_error)


# ---------------------------------------------------------------------------
# Pollinations.ai (free, no key; only one request at a time per connection)
# ---------------------------------------------------------------------------
POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}?width=1024&height=1024&nologo=true&safe=true&seed={seed}"
_pollinations_lock = threading.Lock()


def _pollinations(prompt: str) -> bytes:
    url = POLLINATIONS_URL.format(prompt=quote(prompt[:700]), seed=int.from_bytes(os.urandom(3), "big"))
    # If another panel is already using Pollinations for a while, move on to
    # the next service instead of waiting in line.
    if not _pollinations_lock.acquire(timeout=45):
        raise RuntimeError("Pollinations busy with another panel")
    last_error = "unknown error"
    try:
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=90)
            except requests.RequestException as e:
                last_error = f"network error: {e}"
                time.sleep(3)
                continue
            if response.status_code == 200 and response.headers.get("content-type", "").startswith("image/"):
                return response.content
            last_error = f"Pollinations error {response.status_code}"
            time.sleep(4 * (attempt + 1))
    finally:
        _pollinations_lock.release()
    raise RuntimeError(last_error)


# ---------------------------------------------------------------------------
# AI Horde (free, crowd-sourced GPUs; anonymous key)
# ---------------------------------------------------------------------------
HORDE_API = "https://aihorde.net/api/v2"
HORDE_HEADERS = {"apikey": "0000000000", "Client-Agent": "ComicCraft:2.0:github"}


def _horde(prompt: str) -> bytes:
    response = requests.post(f"{HORDE_API}/generate/async", headers=HORDE_HEADERS, timeout=30, json={
        "prompt": f"{prompt[:900]} ### {NEGATIVE_PROMPT}",
        # Anonymous users are limited to about 620x620
        "params": {"width": 512, "height": 512, "steps": 20, "n": 1},
        "nsfw": False, "censor_nsfw": True, "r2": True,
    })
    if response.status_code != 202:
        raise RuntimeError(f"AI Horde error {response.status_code}: {response.text[:150]}")
    job_id = response.json()["id"]

    deadline = time.time() + 180
    while time.time() < deadline:
        time.sleep(4)
        status = requests.get(f"{HORDE_API}/generate/check/{job_id}", headers=HORDE_HEADERS, timeout=30).json()
        if status.get("faulted") or not status.get("is_possible", True):
            raise RuntimeError("AI Horde could not generate the image")
        if status.get("done"):
            result = requests.get(f"{HORDE_API}/generate/status/{job_id}", headers=HORDE_HEADERS, timeout=30).json()
            return requests.get(result["generations"][0]["img"], timeout=60).content

    requests.delete(f"{HORDE_API}/generate/status/{job_id}", headers=HORDE_HEADERS, timeout=30)
    raise RuntimeError("AI Horde took too long (busy queue)")


PROVIDERS = {
    "flux_space": ("FLUX.1 (Hugging Face Space)", _flux_space),
    "huggingface": ("Stable Diffusion 3 (Hugging Face)", _huggingface),
    "pollinations": ("Pollinations.ai", _pollinations),
    "horde": ("AI Horde", _horde),
}

# Services that are temporarily switched off: name -> (reason, time when it may be retried)
_disabled = {}
_disabled_lock = threading.Lock()


def _is_disabled(name: str) -> bool:
    with _disabled_lock:
        entry = _disabled.get(name)
        if entry and time.time() >= entry[1]:
            del _disabled[name]
            return False
        return entry is not None


def _disable(name: str, reason: str, retry_after: float) -> None:
    with _disabled_lock:
        _disabled[name] = (reason, time.time() + retry_after)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def sanitize_filename(prompt: str) -> str:
    """Convert a free-text prompt into a safe, unique filename."""
    base = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.strip().lower())[:40].strip("_") or "panel"
    return f"{base}_{os.urandom(4).hex()}.png"


def _placeholder(text: str) -> Image.Image:
    """A simple comic-style card shown when a panel image can't be generated."""
    img = Image.new("RGB", (1024, 1024), "#ece4d0")
    draw = ImageDraw.Draw(img)
    draw.rectangle([24, 24, 1000, 1000], outline="#1b1f3b", width=8)
    font_path = FONTS_DIR / "DejaVuSans.ttf"
    font = ImageFont.truetype(str(font_path), 40) if font_path.exists() else ImageFont.load_default(40)
    wrapped = textwrap.fill(text, width=36)
    draw.multiline_text((512, 512), wrapped, fill="#1b1f3b", font=font, anchor="mm", align="center", spacing=12)
    return img


def generate_image(prompt: str, fallback_text: str = "") -> dict:
    """
    Generate one panel image and save it to static/panels/.

    Returns:
        dict: {"url": browser URL, "file": path on disk,
               "source": name of the service that drew it (or None),
               "error": why a placeholder was used (or None)}
    """
    styled_prompt = f"{prompt}, comic book panel, bold ink outlines, vibrant colors, high detail"
    path = PANELS_DIR / sanitize_filename(prompt)
    image = source = None
    failures = []

    for name in IMAGE_PROVIDERS:
        if name not in PROVIDERS or _is_disabled(name):
            continue
        label, provider = PROVIDERS[name]
        try:
            image = Image.open(io.BytesIO(provider(styled_prompt))).convert("RGB")
            source = label
            break
        except ProviderUnavailable as e:
            log.warning("%s unavailable: %s", label, e)
            _disable(name, str(e), e.retry_after)
            failures.append(f"{label}: {e}")
        except Exception as e:
            log.warning("%s failed: %s", label, e)
            failures.append(f"{label}: {e}")

    error = None
    if image is None:
        error = "all image services failed (" + "; ".join(failures or ["none enabled"]) + ")"
        image = _placeholder(fallback_text or prompt)

    image.save(path, "PNG", optimize=True)
    return {"url": static_url(path), "file": str(path), "source": source, "error": error}


def generate_images(panels: list) -> list:
    """Generate all panel images in parallel, keeping the original panel order."""
    with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool:
        return list(pool.map(lambda p: generate_image(p["image_prompt"], p.get("scene_description", "")), panels))
