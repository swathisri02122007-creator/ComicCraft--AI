"""
gemini_flash.py
---------------
Step 1 of the pipeline: uses a fast Gemini model to turn the user's idea into
a comic title and a panel-by-panel outline (title, scene, image prompt).
"""

from pydantic import BaseModel

from app.config import NUM_PANELS, OUTLINE_MODELS
from app.gemini_client import GeminiError, generate_json


class OutlinePanel(BaseModel):
    title: str
    scene_description: str
    image_prompt: str


class ComicOutline(BaseModel):
    comic_title: str
    character_look: str
    panels: list[OutlinePanel]


def generate_outline(user_prompt: str, num_panels: int = NUM_PANELS) -> dict:
    """
    Create the comic outline.

    Returns:
        dict: {"comic_title": str, "character_look": str,
               "panels": [{"title", "scene_description", "image_prompt"}, ...]}
    """
    prompt = f"""
You are planning a short comic book.

Story idea:
{user_prompt}

Create:
- "comic_title": a catchy title for the comic (max 6 words).
- "character_look": one sentence describing exactly what the main character
  looks like (species/age, hair, clothing, colours) so artists draw them the same way every time.
- "panels": exactly {num_panels} panels that tell a complete story with a beginning,
  middle and satisfying ending. For each panel give:
    - "title": a short panel title (2-5 words)
    - "scene_description": one sentence describing what happens
    - "image_prompt": a detailed visual description for an illustrator. Describe the
      scene, camera angle, lighting and the main character's appearance. Do NOT put
      any text, speech bubbles or captions in the image.

Keep the content family-friendly.
"""

    data = generate_json(prompt, ComicOutline, OUTLINE_MODELS)
    outline = ComicOutline.model_validate(data)

    panels = [p.model_dump() for p in outline.panels if p.image_prompt.strip()][:num_panels]
    if not panels:
        raise GeminiError("Gemini returned an empty outline. Please try again.")

    return {
        "comic_title": outline.comic_title.strip() or "My Comic",
        "character_look": outline.character_look.strip(),
        "panels": panels,
    }
