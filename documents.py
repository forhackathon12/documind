"""Generate, save, and locate documentation drafts."""

import os
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from time import perf_counter
from urllib.parse import quote

import gradio as gr

from config import BASE_UPLOAD_FOLDER, DB_CONFIG
from retrieval_postgres import map_ollama_model


DEFAULT_CONFLUENCE_PROMPT = (
    "Create a Confluence-ready technical overview of this codebase. "
    "Explain its purpose, architecture, end-to-end workflow, key "
    "components and responsibilities, data flow, dependencies, "
    "configuration, error handling, and operational/security "
    "considerations. Use clear headings and cite source files and line "
    "ranges. Only make claims supported by the retrieved code."
)

EXCEPTIONS_PROMPT = (
    "Create a Confluence-ready exception and error-handling reference for "
    "this codebase. Identify custom exception classes, where exceptions "
    "are raised and caught, how errors propagate, and how failures are "
    "reported or recovered from. Cite source files and line ranges. Only "
    "describe behavior supported by the retrieved code; do not infer "
    "runtime behavior that is not shown."
)


def document_root():
    """Return the server directory containing generated project drafts."""
    return Path(BASE_UPLOAD_FOLDER).resolve() / "_confluence_drafts"


def export_html_library():
    """Export the saved drafts as HTML and warn if export fails."""
    try:
        from document_library import export_document_library
        return export_document_library(document_root())
    except (ImportError, OSError, UnicodeError) as export_exception:
        gr.Warning(
            "Markdown documents remain saved, but HTML export failed. "
            "Install requirements-html.txt in the app's Python environment. "
            f"Details: {export_exception}"
        )
        return None


def document_created_at(path):
    """Read a draft's recorded creation time without changing the file."""
    with path.open(encoding="utf-8") as draft:
        for line in draft:
            if line.startswith("> Created (UTC): "):
                return line.removeprefix("> Created (UTC): ").strip()
    return "Not recorded"


def list_documents(search="", selected=None):
    """Collect matching saved drafts and return library control updates."""
    root = document_root()
    documents = []
    for path in root.glob("*/*.md"):
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            relative_path = path.relative_to(root).as_posix()
            if search and search.casefold() not in relative_path.casefold():
                continue
            details = path.stat()
            documents.append((details.st_mtime, relative_path, [
                path.parent.name,
                path.name,
                document_created_at(path),
                datetime.fromtimestamp(details.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            ]))
        except (OSError, UnicodeError):
            continue
    documents.sort(reverse=True)
    choices = [entry[1] for entry in documents]
    selected = selected if selected in choices else (choices[0] if choices else None)
    preview, download = preview_document(selected)
    return (
        [entry[2] for entry in documents],
        gr.update(choices=choices, value=selected),
        preview,
        download,
        f"{len(documents)} document(s)",
        str(root / "index.html") if (root / "index.html").is_file() else None,
    )


def preview_document(relative_path):
    """Validate a saved draft path and return its webpage URL and file path."""
    if not relative_path:
        return "", None
    root = document_root()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != ".md":
        return "", None
    try:
        if not path.is_file():
            return "", None
        return "documents/" + quote(path.relative_to(root).as_posix(), safe="/"), str(path)
    except (ImportError, OSError, UnicodeError) as preview_exception:
        gr.Warning(f"Document unavailable. Refresh the library. Details: {preview_exception}")
        return "", None


def save_markdown_draft(
    project_folder,
    prompt,
    rag_result,
    model_name,
    document_title,
    filename
):
    """Save a sourced model response as a project draft and refresh HTML export."""
    if not rag_result.get("sources"):
        return None

    project_name = os.path.basename(
        os.path.normpath(project_folder)
    )
    export_folder = os.path.join(
        BASE_UPLOAD_FOLDER,
        "_confluence_drafts",
        project_name
    )
    os.makedirs(
        export_folder,
        exist_ok=True
    )

    draft_path = os.path.join(
        export_folder,
        filename
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    created_at = (
        document_created_at(Path(draft_path))
        if os.path.isfile(draft_path)
        else timestamp
    )
    draft_content = (
        f"# {project_name}: {document_title}\n\n"
        f"> Status: Draft for manual Confluence publishing\n"
        f"> Generated by DocuMind using `{model_name}`\n"
        f"> Created (UTC): {created_at}\n"
        f"> Updated (UTC): {timestamp}\n\n"
        "## Documentation request\n\n"
        f"{prompt}\n\n"
        f"## {document_title}\n\n"
        f"{rag_result['answer']}\n"
    )

    with open(
        draft_path,
        "w",
        encoding="utf-8"
    ) as draft_file:
        draft_file.write(draft_content)

    export_html_library()
    return draft_path


def generate_markdown_document(
    model_name,
    project_folder,
    prompt,
    document_title,
    filename
):
    """Retrieve project code, generate a draft, and return its link and timing."""
    if not project_folder:
        return "Upload a project and process it before generating a draft."
    if not model_name:
        return "Select an AI model before generating a draft."

    started_at = perf_counter()
    try:
        from retrieval_postgres import retrieve_from_postgres

        target_llm = map_ollama_model(model_name)
        rag_result = retrieve_from_postgres(
            query=prompt.strip(),
            db_config=DB_CONFIG,
            table_name="documind_nprd.chunk_details",
            top_k=7,
            llm_model=target_llm,
            prompt_type="standard",
            project_filter=os.path.basename(
                os.path.normpath(project_folder)
            ),
        )

        # Model failures can include retrieved sources but must not become saved drafts.
        if "LLM Generation Error:" in rag_result.get("answer", ""):
            return escape(rag_result["answer"])

        if not rag_result.get("sources"):
            return escape(rag_result.get("answer", "No indexed code chunks were found."))

        draft_path = save_markdown_draft(
            project_folder,
            prompt.strip(),
            rag_result,
            target_llm,
            document_title,
            filename
        )
        relative_path = Path(draft_path).resolve().relative_to(document_root()).as_posix()
        document_url = "./documents/" + quote(relative_path, safe="/")
        link_label = "Open Exceptions" if filename == "exceptions_draft.md" else "Open Document"
        return (
            '<p><strong>Draft generated.</strong> '
            f'<a href="{document_url}" target="_blank" rel="noopener noreferrer">{link_label}</a><br>'
            f'Model: {escape(target_llm)} | Processing time: {perf_counter() - started_at:.1f} seconds</p>'
        )
    except Exception as draft_exception:
        return f"Draft generation failed: {escape(str(draft_exception))}"


def generate_project_document(model_name, project_folder):
    """Generate the architecture draft using the standard documentation request."""
    return generate_markdown_document(
        model_name,
        project_folder,
        DEFAULT_CONFLUENCE_PROMPT,
        "Codebase Architecture",
        "confluence_draft.md"
    )


def generate_exceptions_document(model_name, project_folder):
    """Generate the exception-handling draft using the exceptions request."""
    return generate_markdown_document(
        model_name,
        project_folder,
        EXCEPTIONS_PROMPT,
        "Exceptions and Error Handling",
        "exceptions_draft.md"
    )
