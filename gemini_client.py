"""
gemini_client.py
----------------
Shared Gemini helper used by both the outline (gemini_flash.py) and story
(gemini_pro.py) steps. It returns structured JSON, retries temporary errors,
falls back to the next model when one is unavailable, and turns API errors
into messages a user can understand.
"""

import json
import logging
import time

from google import genai
from google.genai import errors, types

from app.config import GEMINI_API_KEY

log = logging.getLogger("comiccraft")

_client = None

# Models that recently hit their quota: model name -> time when it may be retried
_exhausted = {}


class GeminiError(RuntimeError):
    """Raised with a user-friendly message when Gemini can't produce a result."""


def _get_client() -> genai.Client:
    global _client
    if not GEMINI_API_KEY:
        raise GeminiError(
            "GEMINI_API_KEY is missing. Get a free key at https://aistudio.google.com/apikey "
            "and paste it into the .env file, then restart the app."
        )
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _is_key_error(error: errors.APIError) -> bool:
    return error.code in (401, 403) or (error.code == 400 and "API key" in str(error))


def _friendly_message(error: Exception) -> str:
    if isinstance(error, errors.APIError):
        if _is_key_error(error):
            return "Gemini rejected the API key. Check GEMINI_API_KEY in the .env file."
        if error.code == 429:
            return (
                "Gemini's free-tier usage limit was reached. Wait a minute and try again. "
                "If it keeps happening, the daily limit is used up - try again tomorrow or use another API key."
            )
        if error.code == 404:
            return "The Gemini model was not found. Update GEMINI_OUTLINE_MODELS / GEMINI_STORY_MODELS in .env."
        if error.code >= 500:
            return "Gemini is overloaded right now. Please try again in a minute."
        return f"Gemini error {error.code}: {getattr(error, 'message', '') or error}"
    return "Gemini returned an unexpected response. Please try again."


def generate_json(prompt: str, schema, models: list, temperature: float = 0.9):
    """
    Ask Gemini for JSON matching `schema`, trying each model in `models` in order.

    Returns:
        The parsed JSON (dict or list).
    """
    client = _get_client()
    last_error = None

    for model in dict.fromkeys(models):  # keeps order, drops duplicates
        if _exhausted.get(model, 0) > time.time():
            continue  # this model recently ran out of quota - don't waste time on it
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=temperature,
                    ),
                )
                return json.loads(response.text)
            except errors.APIError as e:
                last_error = e
                log.warning("Gemini model %s failed (attempt %d): %s", model, attempt + 1, e)
                if e.code >= 500 and attempt == 0:
                    time.sleep(2)
                    continue  # overloaded: retry once, then move on to the next model
                if _is_key_error(e):
                    raise GeminiError(_friendly_message(e)) from e
                if e.code == 429:
                    # Daily limits don't come back soon; per-minute limits do
                    _exhausted[model] = time.time() + (600 if "PerDay" in str(e) else 60)
                break  # quota / overloaded / not found / bad request -> try the next model
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                # Empty or malformed response (e.g. blocked by safety filters) - just retry
                last_error = e
                log.warning("Gemini model %s returned invalid JSON (attempt %d)", model, attempt + 1)

    raise GeminiError(_friendly_message(last_error))
