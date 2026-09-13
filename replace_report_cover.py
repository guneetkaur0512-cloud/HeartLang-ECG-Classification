from pathlib import Path
import math
import tempfile

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.colors import HexColor


INPUT_PDF = Path(r"C:\Users\guneet\OneDrive\Desktop\HeartLang_College_Report.pdf")
OUTPUT_PDF = Path(
    r"C:\Users\guneet\OneDrive\Desktop\HeartLang_College_Report_With_New_Cover.pdf"
)


def draw_centered(c, text, x, y, font="Times-Roman", size=12, color=colors.black):
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawCentredString(x, y, text)


def draw_ecg_wave(c, x0, y0, width, color, stroke_width=2.1):
    c.setStrokeColor(color)
    c.setLineWidth(stroke_width)
    x = x0
    baseline = y0
    step = width / 18

    points = [
        (x, baseline),
        (x + step * 2, baseline),
        (x + step * 2.4, baseline + 8),
        (x + step * 2.8, baseline - 8),
        (x + step * 3.4, baseline),
        (x + step * 5.2, baseline),
        (x + step * 5.7, baseline + 42),
        (x + step * 6.1, baseline - 44),
        (x + step * 6.7, baseline),
        (x + step * 8.5, baseline),
        (x + step * 9.2, baseline + 14),
        (x + step * 10.0, baseline),
        (x + step * 18, baseline),
    ]

    path = c.beginPath()
    path.moveTo(points[0][0], points[0][1])
    for px, py in points[1:]:
        path.lineTo(px, py)
    c.drawPath(path)


def create_cover(path, page_width, page_height):
    c = canvas.Canvas(str(path), pagesize=(page_width, page_height))

    navy = HexColor("#0B1F3A")
    blue = HexColor("#1F4E79")
    light_blue = HexColor("#EAF3FA")
    pale = HexColor("#F7FBFF")
    red = HexColor("#C0392B")
    gold = HexColor("#C9A227")
    gray = HexColor("#3F4A56")

    margin = 42
    cx = page_width / 2

    # Background
    c.setFillColor(pale)
    c.rect(0, 0, page_width, page_height, fill=1, stroke=0)

    # Decorative top band
    c.setFillColor(navy)
    c.rect(0, page_height - 118, page_width, 118, fill=1, stroke=0)

    c.setFillColor(blue)
    c.rect(0, page_height - 126, page_width, 8, fill=1, stroke=0)

    # Subtle circles
    c.setFillColor(HexColor("#D9EAF7"))
    c.circle(page_width - 55, page_height - 165, 96, fill=1, stroke=0)
    c.setFillColor(HexColor("#EEF6FC"))
    c.circle(48, 92, 118, fill=1, stroke=0)

    # Main white panel
    panel_x = margin
    panel_y = 82
    panel_w = page_width - 2 * margin
    panel_h = page_height - 198
    c.setFillColor(colors.white)
    c.roundRect(panel_x, panel_y, panel_w, panel_h, 14, fill=1, stroke=0)
    c.setStrokeColor(HexColor("#D6E2EC"))
    c.setLineWidth(1.1)
    c.roundRect(panel_x, panel_y, panel_w, panel_h, 14, fill=0, stroke=1)

    # Header text inside dark band
    draw_centered(
        c,
        "PROJECT REPORT",
        cx,
        page_height - 54,
        font="Times-Bold",
        size=18,
        color=colors.white,
    )
    draw_centered(
        c,
        "Bachelor of Technology",
        cx,
        page_height - 82,
        font="Times-Roman",
        size=12,
        color=HexColor("#DCEAF5"),
    )

    # Title
    title_y = page_height - 185
    draw_centered(
        c,
        "HEARTLANG-BASED ECG CLASSIFICATION",
        cx,
        title_y,
        font="Times-Bold",
        size=23,
        color=navy,
    )
    draw_centered(
        c,
        "USING SIGNAL PREPROCESSING",
        cx,
        title_y - 30,
        font="Times-Bold",
        size=23,
        color=navy,
    )

    # ECG line
    draw_ecg_wave(c, panel_x + 56, title_y - 78, panel_w - 112, red, stroke_width=2.2)
    c.setStrokeColor(gold)
    c.setLineWidth(1.2)
    c.line(panel_x + 85, title_y - 105, panel_x + panel_w - 85, title_y - 105)

    # Submission block
    y = title_y - 152
    draw_centered(c, "A Project Report", cx, y, font="Times-Bold", size=16, color=blue)
    y -= 34
    draw_centered(
        c,
        "Submitted in partial fulfillment of the requirements for",
        cx,
        y,
        font="Times-Roman",
        size=12.5,
        color=gray,
    )
    y -= 28
    draw_centered(c, "Bachelor of Technology", cx, y, font="Times-Bold", size=15, color=navy)
    y -= 23
    draw_centered(c, "in", cx, y, font="Times-Roman", size=12, color=gray)
    y -= 24
    draw_centered(
        c,
        "Computer Science and Engineering",
        cx,
        y,
        font="Times-Bold",
        size=13.5,
        color=navy,
    )
    y -= 21
    draw_centered(
        c,
        "(Artificial Intelligence & Machine Learning)",
        cx,
        y,
        font="Times-Bold",
        size=13.5,
        color=navy,
    )

    # Submitted by block
    y -= 58
    c.setFillColor(light_blue)
    c.roundRect(cx - 155, y - 58, 310, 88, 10, fill=1, stroke=0)
    c.setStrokeColor(HexColor("#B9D4E8"))
    c.roundRect(cx - 155, y - 58, 310, 88, 10, fill=0, stroke=1)

    draw_centered(c, "Submitted by", cx, y + 4, font="Times-Roman", size=12, color=gray)
    draw_centered(c, "Guneet Kaur", cx, y - 24, font="Times-Bold", size=18, color=navy)

    # Footer institute block
    y = panel_y + 104
    draw_centered(
        c,
        "Department of Computer Science & Engineering",
        cx,
        y,
        font="Times-Bold",
        size=13.5,
        color=blue,
    )
    y -= 27
    draw_centered(c, "[College Name]", cx, y, font="Times-Bold", size=14, color=navy)
    y -= 33
    draw_centered(c, "2026", cx, y, font="Times-Bold", size=14, color=gray)

    # Bottom accent
    c.setFillColor(navy)
    c.rect(0, 0, page_width, 18, fill=1, stroke=0)
    c.setFillColor(blue)
    c.rect(0, 18, page_width, 5, fill=1, stroke=0)

    c.showPage()
    c.save()


def main():
    reader = PdfReader(str(INPUT_PDF))
    first_page = reader.pages[0]
    width = float(first_page.mediabox.width)
    height = float(first_page.mediabox.height)

    with tempfile.TemporaryDirectory() as tmp:
        cover_path = Path(tmp) / "new_cover.pdf"
        create_cover(cover_path, width, height)

        cover_reader = PdfReader(str(cover_path))
        writer = PdfWriter()
        writer.add_page(cover_reader.pages[0])

        for page in reader.pages[1:]:
            writer.add_page(page)

        with OUTPUT_PDF.open("wb") as f:
            writer.write(f)

    print(f"Saved: {OUTPUT_PDF}")
    print(f"Pages: {len(reader.pages)}")


if __name__ == "__main__":
    main()
