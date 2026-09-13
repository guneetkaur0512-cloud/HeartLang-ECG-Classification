from pathlib import Path
import html
import re


SOURCE = Path(r"C:\Users\guneet\HeartLang\HeartLang_Daily_Diary.md")
DEST = Path(r"D:\HeartLang\HeartLang_Daily_Diary.html")


def inline_code(text):
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
        else:
            out.append(html.escape(part))
    return "".join(out)


def markdown_to_html(markdown):
    lines = markdown.splitlines()
    body = []
    in_code = False
    code_lines = []
    in_list = False
    in_day = False

    def close_list():
        nonlocal in_list
        if in_list:
            body.append("</ul>")
            in_list = False

    def close_day():
        nonlocal in_day
        close_list()
        if in_day:
            body.append("</section>")
            in_day = False

    for raw in lines:
        line = raw.rstrip()

        if line.startswith("```"):
            if in_code:
                body.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
                code_lines = []
                in_code = False
            else:
                close_list()
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            close_list()
            continue

        if line.startswith("# "):
            close_day()
            body.append(f"<h1>{inline_code(line[2:].strip())}</h1>")
        elif line.startswith("## "):
            close_day()
            body.append("<section class=\"day-entry\">")
            in_day = True
            body.append(f"<h2>{inline_code(line[3:].strip())}</h2>")
        elif line.startswith("- "):
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{inline_code(line[2:].strip())}</li>")
        else:
            close_list()
            body.append(f"<p>{inline_code(line)}</p>")

    close_day()
    return "\n".join(body)


def main():
    markdown = SOURCE.read_text(encoding="utf-8")
    content = markdown_to_html(markdown)

    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HeartLang Daily Diary</title>
  <style>
    @page {{
      size: A4;
      margin: 18mm 16mm;
    }}
    body {{
      font-family: "Times New Roman", Times, serif;
      color: #111827;
      line-height: 1.36;
      font-size: 12pt;
      max-width: 840px;
      margin: 0 auto;
      background: #ffffff;
    }}
    h1 {{
      text-align: center;
      font-size: 23pt;
      margin: 0 0 20px;
      border-bottom: 2.2px solid #1f4e79;
      padding-bottom: 12px;
      color: #102a43;
      letter-spacing: 0.2px;
    }}
    .day-entry {{
      border-left: 3px solid #1f4e79;
      background: #fbfdff;
      padding: 8px 12px 9px 13px;
      margin: 10px 0 12px;
      page-break-inside: avoid;
      box-shadow: inset 0 0 0 1px #e7eef7;
    }}
    h2 {{
      font-size: 15pt;
      margin: 0 0 6px;
      color: #1f4e79;
      page-break-after: avoid;
    }}
    p {{
      margin: 5px 0 9px;
      text-align: justify;
    }}
    ul {{
      margin: 5px 0 10px 24px;
      padding: 0;
    }}
    li {{
      margin: 3px 0;
    }}
    code {{
      font-family: Consolas, "Courier New", monospace;
      background: #eef3f8;
      padding: 1px 4px;
      border-radius: 3px;
      font-size: 10.5pt;
    }}
    pre {{
      background: #f4f7fb;
      border: 1px solid #cfd9e6;
      border-left: 3px solid #6c8ebf;
      padding: 8px 10px;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
      font-size: 10pt;
      margin: 7px 0 10px;
      page-break-inside: avoid;
    }}
    pre code {{
      background: transparent;
      padding: 0;
    }}
    @media print {{
      body {{
        max-width: none;
      }}
      .day-entry {{
        box-shadow: none;
      }}
      h2 {{
        break-after: avoid;
      }}
    }}
  </style>
</head>
<body>
{content}
</body>
</html>
"""
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(doc, encoding="utf-8")
    print(f"Saved HTML: {DEST}")


if __name__ == "__main__":
    main()
