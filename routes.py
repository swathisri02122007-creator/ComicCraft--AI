"""
routes.py
---------
All web pages and API endpoints for ComicCraft.

Routes are plain `def` functions (not `async def`) on purpose: the AI calls
are slow and blocking, so FastAPI runs them in a worker thread and the site
stays responsive for other visitors while a comic is being generated.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.config import EXPORTS_DIR, TEMPLATES_DIR, missing_keys, static_url
from app.exporters import save_pdf
from app.gemini_client import GeminiError
from app.gemini_flash import generate_outline
from app.gemini_pro import generate_story
from app.image_generator import generate_image, generate_images
from app.layout_builder import build_comic_layout

log = logging.getLogger("comiccraft")

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

SETTINGS = ["forest", "school", "space", "city", "ocean", "castle", "desert", "jungle"]
TONES = ["dramatic", "light-hearted", "funny", "mysterious", "adventurous", "poetic"]
STYLES = ["anime", "comic book", "cartoon", "watercolor", "pixel art", "realistic"]


class PromptRequest(BaseModel):
    """Schema for JSON-based comic generation requests."""
    prompt: str = Field(min_length=3, max_length=1000)
    character_name: str = Field(default="Hero", max_length=60)
    setting: str = Field(default="forest", max_length=60)
    tone: str = Field(default="dramatic", max_length=60)
    style: str = Field(default="anime", max_length=60)


def _build_full_comic(prompt: str, character_name: str, setting: str, tone: str, style: str) -> dict:
    """
    Shared pipeline used by both the form route and the JSON API route.
    Runs: outline -> story -> images -> layout -> PDF.
    """
    character_name = character_name.strip() or "Hero"
    full_prompt = (
        f"{prompt.strip()}\n"
        f"Main character: {character_name}. Setting: {setting}. Tone: {tone}. Art style: {style}."
    )

    # Step 1: panel outline
    outline = generate_outline(full_prompt)

    # Step 2: narration + dialogue for each panel
    story = generate_story(outline, tone)

    # Step 3: images (in parallel). Repeat the character's look + art style in
    # every prompt so the character stays consistent from panel to panel.
    for panel in outline["panels"]:
        panel["image_prompt"] = f"{panel['image_prompt']}. {outline['character_look']} {style} style"
    images = generate_images(outline["panels"])

    # Step 4: combine everything
    layout = build_comic_layout(images, story, outline)

    # Step 5: PDF
    pdf_url = save_pdf(layout, outline["comic_title"], character_name)

    return {
        "comic_title": outline["comic_title"],
        "character_name": character_name,
        "layout": layout,
        "pdf_path": pdf_url,
        "image_errors": sorted({img["error"] for img in images if img["error"]}),
        "image_sources": sorted({img["source"] for img in images if img["source"]}),
    }


def _error_page(request: Request, message: str, status_code: int):
    return templates.TemplateResponse(
        request, "error.html", {"message": message}, status_code=status_code
    )


@router.get("/")
def homepage(request: Request):
    """Homepage where users describe their story."""
    return templates.TemplateResponse(request, "index.html", {
        "settings": SETTINGS, "tones": TONES, "styles": STYLES, "missing_keys": missing_keys(),
    })


@router.post("/generate")
def generate_comic(
    request: Request,
    prompt: str = Form(..., min_length=3, max_length=1000),
    character_name: str = Form("Hero", max_length=60),
    setting: str = Form("forest", max_length=60),
    tone: str = Form("dramatic", max_length=60),
    style: str = Form("anime", max_length=60),
):
    """Handles the form, runs the AI pipeline, shows the comic preview page."""
    try:
        result = _build_full_comic(prompt, character_name, setting, tone, style)
    except GeminiError as e:
        return _error_page(request, str(e), 503)
    except Exception:
        log.exception("Comic generation failed")
        return _error_page(request, "Something went wrong while creating your comic. Please try again.", 500)

    return templates.TemplateResponse(request, "comic_preview.html", result)


@router.post("/generate-comic/json")
def generate_comic_json(payload: PromptRequest):
    """API route: accepts a JSON payload and returns the comic as JSON."""
    try:
        return _build_full_comic(
            payload.prompt, payload.character_name, payload.setting, payload.tone, payload.style
        )
    except GeminiError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        log.exception("Comic generation failed")
        raise HTTPException(status_code=500, detail="Comic generation failed. Please try again.")


@router.get("/gallery")
def gallery(request: Request):
    """Lists every comic PDF created so far, newest first."""
    comics = []
    for pdf in sorted(EXPORTS_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True):
        name = pdf.stem.rsplit("_", 2)[0].replace("_", " ")
        comics.append({
            "title": name or "Comic",
            "url": static_url(pdf),
            "created": datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%d %b %Y, %H:%M"),
        })
    return templates.TemplateResponse(request, "gallery.html", {"comics": comics})


@router.get("/about")
def about(request: Request):
    """Explains how ComicCraft works (handy for presentations)."""
    return templates.TemplateResponse(request, "about.html")


@router.get("/export-success")
def export_success(request: Request, pdf_path: str = ""):
    """Confirmation page shown after the comic PDF is downloaded."""
    if not (pdf_path.startswith("/static/exports/") and pdf_path.endswith(".pdf") and ".." not in pdf_path):
        pdf_path = ""
    return templates.TemplateResponse(request, "export_success.html", {"pdf_path": pdf_path})


@router.get("/health")
def health():
    """Simple status check - also reports whether the API keys are configured."""
    return {"status": "ok", "missing_keys": missing_keys()}


@router.post("/test-image")
def test_image(prompt: str = "a futuristic city at sunset, sci-fi, cinematic"):
    """Developer utility: generates a single image to test the Hugging Face setup."""
    result = generate_image(prompt)
    if result["error"]:
        raise HTTPException(status_code=502, detail=result["error"])
    return {"message": "Image generated successfully", "path": result["url"]}
