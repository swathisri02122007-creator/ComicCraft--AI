"""
layout_builder.py
------------------
Step 4 of the pipeline: combines the outline, story text and images into one
structured per-panel layout used by the web preview and the PDF exporter.
"""


def build_comic_layout(images: list, story: list, outline: dict) -> list:
    """
    Args:
        images (list): results from image_generator.generate_images().
        story (list): per-panel narration/dialogue from gemini_pro.generate_story().
        outline (dict): the outline from gemini_flash.generate_outline().

    Returns:
        list: one dict per panel with "panel", "title", "scene_description",
              "narration", "dialogue", "image_url", "image_file" and "image_error".
    """
    layout = []
    for idx, (panel, text, image) in enumerate(zip(outline["panels"], story, images), start=1):
        layout.append({
            "panel": idx,
            "title": panel["title"].strip() or f"Panel {idx}",
            "scene_description": panel["scene_description"].strip(),
            "narration": text["narration"].strip(),
            "dialogue": [d for d in text["dialogue"] if d["line"].strip()],
            "image_url": image["url"],
            "image_file": image["file"],
            "image_error": image["error"],
        })
    return layout
