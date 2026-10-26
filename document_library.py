"""Export saved DocuMind drafts as a standalone, offline HTML library."""

import argparse
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from markdown_it import MarkdownIt


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>DocuMind Documents</title>
<style>
:root { color-scheme: light; --ink: #202622; --muted: #56615b; --line: #d8dfda; --accent: #176948; }
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: #fff; font: 16px Georgia, serif; }
header { padding: 22px 28px; background: #176948; color: #fff; border-bottom: 4px solid #66c2b1; display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }
header h1 { margin: 0; font-size: 26px; } header span { color: #d9f2e7; }
main { display: grid; grid-template-columns: minmax(240px, 310px) minmax(0, 1fr); min-height: calc(100vh - 80px); }
aside { padding: 20px; border-right: 1px solid var(--line); background: #f5f8f6; }
label { display: block; margin-bottom: 8px; font-weight: bold; }
input { width: 100%; padding: 11px; border: 1px solid #89988e; border-radius: 4px; font: inherit; }
input:focus-visible, button:focus-visible, a:focus-visible { outline: 3px solid #b36622; outline-offset: 3px; }
#count { color: var(--muted); font-size: 14px; }
nav { display: grid; gap: 0; max-height: 70vh; overflow: auto; }
button { width: 100%; text-align: left; padding: 14px 10px; border: 0; border-bottom: 1px solid var(--line); border-left: 3px solid transparent; background: transparent; color: var(--ink); font: inherit; cursor: pointer; overflow-wrap: anywhere; }
button:hover { background: #e7eee9; } button[aria-current="true"] { border-left-color: var(--accent); background: #e1ece5; }
button small { display: block; color: var(--muted); margin-top: 6px; }
.viewer { min-width: 0; padding: 28px 40px; }
.document-bar { display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px; border-bottom: 1px solid var(--line); padding-bottom: 20px; }
#title { font-size: 22px; margin: 0 0 10px; overflow-wrap: anywhere; color: #12523a; }
#dates { margin: 0; font-size: 14px; color: var(--muted); white-space: pre-line; line-height: 1.7; }
[hidden] { display: none !important; }
article { max-width: 1000px; line-height: 1.7; overflow-wrap: anywhere; }
article h1 { font-size: 28px; color: #12523a; } article h2 { font-size: 23px; color: #146e82; border-bottom: 1px solid #c8ded3; padding-bottom: 8px; } article h3 { font-size: 19px; color: #176948; }
article pre { overflow: auto; padding: 16px; background: #f2f4f3; border-radius: 4px; }
article code { font-size: 14px; } article table { display: block; overflow: auto; border-collapse: collapse; }
article th, article td { border: 1px solid var(--line); padding: 8px 12px; text-align: left; }
article th { background: #e7f1f5; color: #174c5b; }
article tr:nth-child(even) td { background: #f5f8f6; }
article blockquote { margin-left: 0; border-left: 3px solid #b66c25; padding: 8px 16px; color: #65461d; background: #fff7e8; }
article a { color: var(--accent); }
@media (max-width: 700px) {
  header { padding: 18px; } main { grid-template-columns: minmax(0, 1fr); }
  aside { border-right: 0; border-bottom: 1px solid var(--line); padding: 18px; }
  nav { max-height: 220px; } .viewer { padding: 22px 18px; }
}
</style>
</head>
<body>
<header><h1>DocuMind</h1><span>Documents</span></header>
<main>
<aside>
<label for="search">Search documents</label>
<input id="search" type="search" placeholder="Project or document">
<p id="count" role="status" aria-live="polite"></p>
<nav id="documents" aria-label="Documents"></nav>
</aside>
<section class="viewer" aria-label="Document preview">
<div class="document-bar"><div><h2 id="title">No documents yet</h2><p id="dates"></p></div></div>
<article id="preview"></article>
</section>
</main>
<script id="document-data" type="application/json">__DOCUMENTS__</script>
<script>
const documents = JSON.parse(document.getElementById('document-data').textContent);
const search = document.getElementById('search');
const navigation = document.getElementById('documents');
let selected = documents.length ? documents[0].path : null;
// Display the selected document's title, timestamps, and rendered content.
function showDocument(doc) {
  document.getElementById('title').textContent = doc ? doc.title : 'No documents found';
  document.getElementById('dates').textContent = doc ? `${doc.project}\nCreated (UTC): ${doc.created}\nUpdated (UTC): ${doc.updated}` : '';
  document.getElementById('preview').innerHTML = doc ? doc.html : '';
}
// Rebuild the filtered document list and display the current selection.
function render() {
  const query = search.value.trim().toLocaleLowerCase();
  const matches = documents.filter(doc => `${doc.project} ${doc.title} ${doc.filename}`.toLocaleLowerCase().includes(query));
  if (!matches.some(doc => doc.path === selected)) selected = matches.length ? matches[0].path : null;
  navigation.replaceChildren();
  document.getElementById('count').textContent = `${matches.length} of ${documents.length} documents`;
  for (const doc of matches) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = doc.title;
    button.setAttribute('aria-current', String(doc.path === selected));
    const project = document.createElement('small');
    project.textContent = doc.project;
    button.append(project);
    button.addEventListener('click', () => { selected = doc.path; render(); });
    navigation.append(button);
  }
  showDocument(matches.find(doc => doc.path === selected));
}
search.addEventListener('input', render);
render();
</script>
</body>
</html>
"""


def render_document_page(content, title):
    """Render one Markdown draft as a standalone styled HTML document."""
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table").disable("image")
    return (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    '<meta name="referrer" content="no-referrer">'
    f'<title>{escape(title)}</title>'
    '<style>body{margin:0;color:#202622;background:#fff;font:16px Georgia,serif;}'
    'main{max-width:1000px;margin:auto;padding:24px;line-height:1.7;overflow-wrap:anywhere;}'
    'pre{overflow:auto;padding:16px;background:#f2f4f3;}table{display:block;overflow:auto;'
    'border-collapse:collapse;}th,td{border:1px solid #d8dfda;padding:8px;text-align:left;}'
    'h1{color:#12523a;}h2{color:#146e82;border-bottom:1px solid #c8ded3;padding-bottom:8px;}'
    'h3{color:#176948;}th{background:#e7f1f5;color:#174c5b;}'
    'tr:nth-child(even) td{background:#f5f8f6;}'
    'blockquote{margin-left:0;border-left:3px solid #b66c25;padding:8px 16px;'
    'background:#fff7e8;color:#65461d;}a{color:#146e82;}'
    'header{padding:18px 24px;background:#176948;color:#fff;border-bottom:4px solid #66c2b1;}'
    'header a{color:#fff;text-underline-offset:4px;}img{max-width:100%;}'
    '</style></head><body><header><strong>DocuMind Documents</strong></header><main>'
    f'{renderer.render(content)}</main></body></html>'
        )


def render_document_library(draft_root):
    """Build a searchable HTML library from the saved project drafts."""
    root = Path(draft_root).resolve()
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table").disable("image")
    documents = []
    for path in root.glob("*/*.md"):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            continue
        content = path.read_text(encoding="utf-8")
        title = next(
            (line[2:].strip() for line in content.splitlines() if line.startswith("# ")),
            path.stem.replace("_", " "),
        )
        created = next(
            (line.removeprefix("> Created (UTC): ").strip()
             for line in content.splitlines() if line.startswith("> Created (UTC): ")),
            "Not recorded",
        )
        modified = path.stat().st_mtime
        documents.append({
            "path": path.relative_to(root).as_posix(),
            "filename": path.name,
            "project": path.parent.name,
            "title": title,
            "created": created,
            "updated": datetime.fromtimestamp(modified, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "modified": modified,
            "markdown": content,
            "html": renderer.render(content),
        })
    documents.sort(key=lambda document: (-document["modified"], document["path"]))
    payload = json.dumps(documents, ensure_ascii=True).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return PAGE.replace("__DOCUMENTS__", payload)


def create_document_routes(draft_root):
    """Create HTTP routes for the document library and individual saved drafts."""
    from starlette.exceptions import HTTPException
    from starlette.responses import HTMLResponse
    from starlette.routing import Route

    root = Path(draft_root).resolve()

    def library_page(request):
        """Serve a fresh library page without browser caching."""
        return HTMLResponse(render_document_library(root), headers={"Cache-Control": "no-store"})

    def document_page(request):
        """Validate the requested draft path and serve its rendered HTML page."""
        path = (root / request.path_params["document_path"]).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() != ".md" or not path.is_file():
            raise HTTPException(status_code=404, detail="Document not found")
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            raise HTTPException(status_code=404, detail="Document unavailable") from None
        return HTMLResponse(render_document_page(content, path.stem), headers={"Cache-Control": "no-store"})

    return [
        Route("/documents", library_page, methods=["GET"]),
        Route("/documents/{document_path:path}", document_page, methods=["GET"]),
    ]


def export_document_library(draft_root):
    """Write the offline library HTML using temporary-file replacement."""
    root = Path(draft_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = root / "index.html"
    temporary = root / "index.html.tmp"
    temporary.write_text(render_document_library(root), encoding="utf-8")
    temporary.replace(output)
    return str(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("draft_root", type=Path, help="Directory containing project Markdown draft folders")
    arguments = parser.parse_args()
    print(export_document_library(arguments.draft_root))
