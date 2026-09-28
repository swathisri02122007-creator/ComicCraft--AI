"""
exporters.py
------------
Step 5 of the pipeline: compiles the comic (cover + one page per panel) into
a downloadable PDF using fpdf2.
"""

import os
import re
from datetime import datetime

from fpdf import FPDF
from PIL import Image

from app.config import EXPORTS_DIR, FONTS_DIR, static_url

FONT_REGULAR = FONTS_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONTS_DIR / "DejaVuSans-Bold.ttf"

# Used only if the bundled Unicode font is missing (core PDF fonts are Latin-1 only)
_ASCII_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                            "–": "-", "—": "-", "…": "...", "•": "*"})

INK = (27, 31, 59)
POP = (232, 72, 44)
MUTED = (122, 116, 102)


class ComicPDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.unicode = FONT_REGULAR.exists() and FONT_BOLD.exists()
        if self.unicode:
            self.add_font("DejaVu", "", str(FONT_REGULAR))
            self.add_font("DejaVu", "B", str(FONT_BOLD))
            self.family_name = "DejaVu"
        else:
            self.family_name = "Helvetica"

    def clean(self, text: str) -> str:
        if self.unicode:
            return text
        return text.translate(_ASCII_MAP).encode("latin-1", "replace").decode("latin-1")

    def font(self, size: int, bold: bool = False, color=INK):
        self.set_font(self.family_name, "B" if bold else "", size)
        self.set_text_color(*color)

    def footer(self):
        if self.page_no() > 1:
            self.set_y(-12)
            self.font(9, color=MUTED)
            self.cell(0, 6, f"Made with ComicCraft  -  {self.page_no() - 1}", align="C")


def _place_image(pdf: ComicPDF, image_file: str, max_h: float) -> None:
    """Draw an image centred on the page, scaled to fit without stretching."""
    with Image.open(image_file) as img:
        ratio = img.width / img.height
    w = min(pdf.epw, max_h * ratio)
    h = w / ratio
    x = pdf.l_margin + (pdf.epw - w) / 2
    y = pdf.get_y()
    pdf.set_draw_color(*INK)
    pdf.set_line_width(0.8)
    pdf.image(image_file, x=x, y=y, w=w, h=h)
    pdf.rect(x, y, w, h)
    pdf.set_y(y + h + 6)


def save_pdf(layout: list, comic_title: str = "My Comic", character_name: str = "") -> str:
    """
    Build the PDF and save it to static/exports/.

    Returns:
        str: the browser URL of the saved PDF.
    """
    pdf = ComicPDF()
    pdf.set_title(comic_title)
    pdf.set_creator("ComicCraft")

    # Cover page: title + first panel artwork
    pdf.add_page()
    pdf.ln(30)
    pdf.font(34, bold=True, color=POP)
    pdf.multi_cell(0, 14, pdf.clean(comic_title), align="C", new_x="LMARGIN", new_y="NEXT")
    if character_name:
        pdf.ln(2)
        pdf.font(14, color=MUTED)
        pdf.cell(0, 8, pdf.clean(f"Starring {character_name}"), align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    if layout and os.path.exists(layout[0]["image_file"]):
        _place_image(pdf, layout[0]["image_file"], max_h=150)

    for panel in layout:
        pdf.add_page()

        pdf.font(18, bold=True)
        pdf.multi_cell(0, 9, pdf.clean(f"Panel {panel['panel']}: {panel['title']}"), align="C",
                       new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        if os.path.exists(panel["image_file"]):
            _place_image(pdf, panel["image_file"], max_h=150)

        if panel["narration"]:
            pdf.font(12, color=MUTED)
            pdf.multi_cell(0, 7, pdf.clean(panel["narration"]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        for line in panel["dialogue"]:
            pdf.font(12, bold=True)
            pdf.write(7, pdf.clean(f"{line['speaker']}: "))
            pdf.font(12)
            pdf.write(7, pdf.clean(line["line"]))
            pdf.ln(8)

    safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", comic_title).strip("_")[:40] or "comic"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = EXPORTS_DIR / f"{safe_title}_{timestamp}.pdf"
    pdf.output(str(pdf_path))

    return static_url(pdf_path)
