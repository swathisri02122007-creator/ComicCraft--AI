"""
gemini_pro.py
-------------
Step 2 of the pipeline: expands the outline into narration and character
dialogue for every panel. The story comes back as structured JSON (one entry
per panel), so it always lines up with the panel images.
"""

from pydantic import BaseModel

from app.config import STORY_MODELS
from app.gemini_client import generate_json


class DialogueLine(BaseModel):
    speaker: str
    line: str


class PanelStory(BaseModel):
    narration: str
    dialogue: list[DialogueLine]


class ComicStory(BaseModel):
    panels: list[PanelStory]


def generate_story(outline: dict, tone: str) -> list:
    """
    Write narration + dialogue for each panel of the outline.

    Returns:
        list: one {"narration": str, "dialogue": [{"speaker", "line"}]} per panel,
              always the same length as outline["panels"].
    """
    panels = outline["panels"]
    formatted_outline = "\n".join(
        f"{i}. {p['title']}: {p['scene_description']}" for i, p in enumerate(panels, start=1)
    )

    prompt = f"""
You're a comic book writer. Write the text for the comic "{outline['comic_title']}".
Tone: {tone}.

Panel outline:
{formatted_outline}

Return exactly {len(panels)} panels, in the same order. For each panel give:
- "narration": 1-2 short sentences of caption text.
- "dialogue": 1-3 short speech-bubble lines, each with a "speaker" and a "line".

Keep it fun, punchy and family-friendly, like a real comic book.
"""

    data = generate_json(prompt, ComicStory, STORY_MODELS)
    story = [p.model_dump() for p in ComicStory.model_validate(data).panels]

    # Make sure every panel has text, even if Gemini returned too few entries
    for panel in panels[len(story):]:
        story.append({"narration": panel["scene_description"], "dialogue": []})
    return story[:len(panels)]
