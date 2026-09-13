from pathlib import Path
import textwrap


SOURCE = Path(r"C:\Users\guneet\HeartLang\HeartLang_Daily_Diary.md")
DEST = Path(r"D:\HeartLang\HeartLang_Daily_Diary.pdf")


def clean_text(text):
    replacements = {
        "\u00b1": "+/-",
        "\u2192": "->",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00d7": "x",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def pdf_escape(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_lines(markdown):
    lines = []
    in_code = False

    for raw in markdown.splitlines():
        line = raw.rstrip()

        if line.startswith("```"):
            in_code = not in_code
            continue

        if not line:
            lines.append(("", "normal"))
            continue

        if in_code:
            wrapped = textwrap.wrap(line, width=86) or [""]
            lines.extend((part, "code") for part in wrapped)
            continue

        if line.startswith("# "):
            lines.append((line[2:].strip(), "title"))
        elif line.startswith("## "):
            lines.append(("", "normal"))
            lines.append((line[3:].strip(), "heading"))
        elif line.startswith("- "):
            wrapped = textwrap.wrap(line, width=88, subsequent_indent="  ")
            lines.extend((part, "normal") for part in wrapped)
        else:
            wrapped = textwrap.wrap(line, width=92) or [""]
            lines.extend((part, "normal") for part in wrapped)

    return lines


def content_stream(page_lines, page_no):
    commands = ["BT"]

    y = 760
    for text, style in page_lines:
        text = clean_text(text)

        if style == "title":
            font = "/F2"
            size = 18
            leading = 24
        elif style == "heading":
            font = "/F2"
            size = 13
            leading = 19
        elif style == "code":
            font = "/F3"
            size = 9
            leading = 13
        else:
            font = "/F1"
            size = 10
            leading = 15

        if text:
            commands.append(f"{font} {size} Tf")
            commands.append(f"1 0 0 1 54 {y} Tm ({pdf_escape(text)}) Tj")

        y -= leading if text else 10

    commands.append("/F1 9 Tf")
    commands.append(f"1 0 0 1 460 28 Tm (Page {page_no}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1", "replace")


def paginate(lines):
    pages = []
    current = []
    y = 760

    for item in lines:
        _, style = item
        if style == "title":
            leading = 24
        elif style == "heading":
            leading = 19
        elif style == "code":
            leading = 13
        else:
            leading = 15 if item[0] else 10

        if y - leading < 54 and current:
            pages.append(current)
            current = []
            y = 760

        current.append(item)
        y -= leading

    if current:
        pages.append(current)
    return pages


def write_pdf(pages):
    objects = []

    def add(obj):
        objects.append(obj)
        return len(objects)

    catalog_id = add(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = add(b"")
    font_helv = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    font_mono = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    page_ids = []
    content_ids = []

    for i, page_lines in enumerate(pages, start=1):
        stream = content_stream(page_lines, i)
        content = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream"
        )
        content_id = add(content)
        content_ids.append(content_id)

        page = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> "
            + f"/Contents {content_id} 0 R >>".encode("latin-1")
        )
        page_ids.append(add(page))

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode(
        "latin-1"
    )

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
    )

    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_bytes(pdf)


def main():
    markdown = SOURCE.read_text(encoding="utf-8")
    lines = build_lines(markdown)
    pages = paginate(lines)
    write_pdf(pages)
    print(f"Saved PDF: {DEST}")
    print(f"Pages: {len(pages)}")


if __name__ == "__main__":
    main()
