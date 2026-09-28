"""
main.py
-------
ComicCraft application entry point. Creates the FastAPI app, mounts static
files, and wires in the routes defined in routes.py.

Run with:
    uvicorn app.main:app --reload
"""

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import STATIC_DIR, missing_keys
from app.routes import router, templates

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("comiccraft")

app = FastAPI(
    title="ComicCraft",
    description="AI Comic Story Creator using Gemini + Stable Diffusion",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    """Show a friendly page (instead of raw JSON) when the form is filled in wrong."""
    if request.url.path.startswith("/generate-comic/json"):
        return JSONResponse({"detail": jsonable_encoder(exc.errors())}, status_code=422)
    return templates.TemplateResponse(
        request, "error.html",
        {"message": "Please check the form - the story prompt needs at least 3 characters (max 1000)."},
        status_code=422,
    )


if missing_keys():
    log.warning("Missing API keys in .env: %s - see README.md", ", ".join(missing_keys()))
