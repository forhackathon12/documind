import gradio as gr
import os
import shutil
import subprocess

import numpy as np
import ollama
import psycopg2
import psycopg2.extensions

from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from tree_sitter import Language, Parser
import tree_sitter_python as tspython


def _adapt_numpy_array(arr):
    return psycopg2.extensions.AsIs(f"ARRAY{arr.tolist()}::real[]")


psycopg2.extensions.register_adapter(np.ndarray, _adapt_numpy_array)



# --------------------------------------------------
# Tree-sitter Python language
# --------------------------------------------------
PY_LANGUAGE = Language(tspython.language())


# --------------------------------------------------
# Base target upload folder
# --------------------------------------------------
BASE_UPLOAD_FOLDER = os.environ.get("BASE_UPLOAD_FOLDER", "/home/azureuser/proj-documind/test/repo")

try:
    os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)
except Exception:
    BASE_UPLOAD_FOLDER = os.path.expanduser("~/proj-documind/test/repo")
    os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)


# --------------------------------------------------
# PostgreSQL configuration
# --------------------------------------------------
DB_CONFIG = {
    "host": os.environ.get("PGHOST", "localhost"),
    "database": os.environ.get("PGDATABASE", "documind_dev"),
    "user": os.environ.get("PGUSER", "vadmin"),
    "password": os.environ.get("PGPASSWORD", "documind@123"),
}


# --------------------------------------------------
# Get all files from project folder
# --------------------------------------------------
def get_upload_folder_files(project_folder):
    if not project_folder:
        return []

    if not os.path.isdir(project_folder):
        return []

    file_paths = []

    for root, directories, filenames in os.walk(project_folder):
        for filename in sorted(filenames):
            path = os.path.join(
                root,
                filename
            )

            if os.path.isfile(path):
                file_paths.append(path)

    return file_paths


# --------------------------------------------------
# Upload Function
# --------------------------------------------------
def process_upload(files, project_name):
    """
    Validate the project name, create the project folder,
    copy the selected files, update upload status,
    store the project folder, and enable model selection.
    """

    # ----------------------------------------------
    # Validate project name
    # ----------------------------------------------
    if not project_name or not project_name.strip():
        return (
            "⚠️ Please enter a project name.",
            None,
            gr.update(interactive=False)
        )

    project_name = project_name.strip()

    if (
        project_name in {".", ".."}
        or "/" in project_name
        or "\\" in project_name
        or os.path.isabs(project_name)
    ):
        return (
            (
                "⚠️ Invalid project name.\n\n"
                "Enter only the project name.\n"
                "Do not include '/', '\\', '.', or '..'."
            ),
            None,
            gr.update(interactive=False)
        )

    # ----------------------------------------------
    # Validate selected files
    # ----------------------------------------------
    if not files:
        return (
            "⚠️ Please select at least one file to upload.",
            None,
            gr.update(interactive=False)
        )

    # ----------------------------------------------
    # Create project folder
    # ----------------------------------------------
    project_folder = os.path.join(
        BASE_UPLOAD_FOLDER,
        project_name
    )

    try:
        os.makedirs(
            project_folder,
            exist_ok=True
        )

    except Exception as e:
        return (
            (
                "❌ Unable to create the project folder.\n\n"
                f"Project Name: {project_name}\n"
                f"Target Folder: {project_folder}\n\n"
                f"Error: {str(e)}"
            ),
            None,
            gr.update(interactive=False)
        )

    # ----------------------------------------------
    # Convert selected files into a list
    # ----------------------------------------------
    file_list = (
        files
        if isinstance(files, list)
        else [files]
    )

    saved_files = []
    errors = []

    # ----------------------------------------------
    # Copy files into the project folder
    # ----------------------------------------------
    for file_obj in file_list:
        try:
            source_path = (
                file_obj.name
                if hasattr(file_obj, "name")
                else str(file_obj)
            )

            filename = os.path.basename(
                source_path
            )

            destination_path = os.path.join(
                project_folder,
                filename
            )

            # Avoid copying the same file onto itself
            if (
                os.path.abspath(source_path)
                != os.path.abspath(destination_path)
            ):
                shutil.copy2(
                    source_path,
                    destination_path
                )

            saved_files.append(filename)

        except Exception as e:
            errors.append(
                f"{file_obj}: {str(e)}"
            )

    # ----------------------------------------------
    # No files uploaded successfully
    # ----------------------------------------------
    if not saved_files:
        status = (
            "❌ No files were uploaded successfully.\n\n"
            f"Project Name: {project_name}\n"
            f"Upload Path: {project_folder}"
        )

        if errors:
            status += "\n\nUpload Errors:\n"

            status += "\n".join(
                f"• {error}"
                for error in errors
            )

        return (
            status,
            None,
            gr.update(interactive=False)
        )

    # ----------------------------------------------
    # Build upload status
    # ----------------------------------------------
    status = (
        f"✅ Successfully uploaded "
        f"{len(saved_files)} file(s).\n\n"
        f"Project Name: {project_name}\n"
        f"Uploaded To: {project_folder}\n\n"
        "Uploaded Files:\n"
    )

    status += "\n".join(
        f"• {file_name}"
        for file_name in saved_files
    )

    if errors:
        status += "\n\nUpload Errors:\n"

        status += "\n".join(
            f"• {error}"
            for error in errors
        )

    # Enable model selector after successful upload
    return (
        status,
        project_folder,
        gr.update(
            interactive=True,
            value=None
        )
    )


# --------------------------------------------------
# Enable or disable Generate Output button
# --------------------------------------------------
def update_generate_button(
    model_name,
    project_folder,
    project_indexed
):
    if model_name and project_folder:
        return (
            gr.update(interactive=True),
            gr.update(interactive=bool(project_indexed)),
            gr.update(interactive=bool(project_indexed)),
            gr.update(interactive=bool(project_indexed))
        )

    return (
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False)
    )


# --------------------------------------------------
# Reset upload when project name changes
# --------------------------------------------------
def reset_project_workflow():
    return (
        None,
        "",
        gr.update(
            value=None,
            interactive=False
        ),
        gr.update(
            interactive=False
        ),
        "",
        False,
        gr.update(interactive=False),
        "",
        None,
        gr.update(interactive=False),
        "",
        None,
        gr.update(interactive=False),
        ""
    )


# --------------------------------------------------
# Reset processing when uploaded file selection changes
# --------------------------------------------------
def reset_file_workflow():
    return (
        None,
        "",
        gr.update(
            value=None,
            interactive=False
        ),
        gr.update(
            interactive=False
        ),
        "",
        False,
        gr.update(interactive=False),
        "",
        None,
        gr.update(interactive=False),
        "",
        None,
        gr.update(interactive=False),
        ""
    )


# --------------------------------------------------
# Tree-sitter Chunking Function
# --------------------------------------------------
def tree_sitter_chunk(code):
    chunks = []

    try:
        source_bytes = code.encode(
            "utf-8"
        )

        parser = Parser(
            PY_LANGUAGE
        )

        tree = parser.parse(
            source_bytes
        )

        root_node = tree.root_node

        print(
            "Tree-sitter root node:",
            root_node.type
        )

        print(
            "Tree-sitter syntax error:",
            root_node.has_error
        )

        def traverse_node(node):
            # Capture functions and classes
            # at any nesting level
            if node.type in (
                "function_definition",
                "class_definition"
            ):
                chunk_content = source_bytes[
                    node.start_byte:node.end_byte
                ].decode(
                    "utf-8",
                    errors="replace"
                )

                name_node = node.child_by_field_name(
                    "name"
                )

                chunk_name = (
                    source_bytes[
                        name_node.start_byte:
                        name_node.end_byte
                    ].decode(
                        "utf-8",
                        errors="replace"
                    )
                    if name_node
                    else "Unknown"
                )

                chunks.append(
                    {
                        "type": node.type,
                        "name": chunk_name,
                        "start_line": (
                            node.start_point.row + 1
                        ),
                        "end_line": (
                            node.end_point.row + 1
                        ),
                        "content": chunk_content
                    }
                )

            # Recursively check named child nodes
            for child in node.named_children:
                traverse_node(child)

        traverse_node(
            root_node
        )

        # Fallback for Python files with no
        # functions or classes
        if not chunks and code.strip():
            chunks.append(
                {
                    "type": "module",
                    "name": "module_level_code",
                    "start_line": 1,
                    "end_line": len(
                        code.splitlines()
                    ),
                    "content": code
                }
            )

        return chunks

    except Exception as e:
        return [
            {
                "type": "ERROR",
                "name": "TreeSitterParseError",
                "start_line": 0,
                "end_line": 0,
                "content": str(e)
            }
        ]


# --------------------------------------------------
# Generate embedding for one chunk
# --------------------------------------------------
def generate_chunk_embedding(chunk_content):
    response = ollama.embed(
        model="mxbai-embed-large",
        input=chunk_content
    )

    if hasattr(
        response,
        "embeddings"
    ):
        embedding = response.embeddings[0]

    else:
        embedding = response["embeddings"][0]

    # Convert to NumPy array for pgvector
    return np.array(
        embedding,
        dtype=np.float32
    )


# --------------------------------------------------
# Embedding output for display
# --------------------------------------------------
def embedding_code(chunks):
    embeddings_output = []

    for i, chunk in enumerate(
        chunks,
        start=1
    ):
        chunk_content = chunk.get(
            "content",
            ""
        )

        if chunk.get("type") == "ERROR":
            embeddings_output.append(
                f"""
CHUNK {i}
Embedding skipped because Tree-sitter parsing failed.

Error:
{chunk_content}
"""
            )

            continue

        if not chunk_content.strip():
            embeddings_output.append(
                f"""
CHUNK {i}
Embedding skipped because the chunk is empty.
"""
            )

            continue

        try:
            embedding = generate_chunk_embedding(
                chunk_content
            )

            embeddings_output.append(
                f"""
CHUNK {i}
Embedding Size: {len(embedding)}
First 10 Values:
{embedding[:10].tolist()}
"""
            )

        except Exception as e:
            embeddings_output.append(
                f"""
CHUNK {i}
Embedding failed.

Error:
{str(e)}
"""
            )

    return embeddings_output


# --------------------------------------------------
# Bulk insert chunk records into PostgreSQL
# --------------------------------------------------
def insert_chunks_into_postgres(
    cursor,
    chunk_records
):
    if not chunk_records:
        return 0

    insert_query = """
        INSERT INTO documind_nprd.chunk_details
        (
            file_name,
            chunk_number,
            chunk_type,
            chunk_name,
            chunk_start_line,
            chunk_end_line,
            chunk_content,
            chunk_embeddings
        )
        VALUES %s
    """

    execute_values(
        cursor,
        insert_query,
        chunk_records,
        template=(
            "(%s, %s, %s, %s, %s, %s, %s, %s)"
        ),
        page_size=100
    )

    return len(chunk_records)

# --------------------------------------------------
# Map UI Model to Local Ollama LLM
# --------------------------------------------------
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


def map_ollama_model(selected_name: str) -> str:
    """Map UI model selection to an available local Ollama LLM."""
    try:
        res = ollama.list()
        pulled = []
        for m in getattr(res, "models", []):
            name = getattr(m, "model", "")
            if "embed" not in name.lower():
                pulled.append(name)
        for p in pulled:
            prefix = selected_name.lower().split("-")[0]
            if prefix in p.lower():
                return p
        if pulled:
            return pulled[0]
    except Exception:
        pass
    return "qwen2.5:0.5b"


def save_markdown_draft(
    project_folder,
    prompt,
    rag_result,
    model_name,
    document_title,
    filename
):
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
    draft_content = (
        f"# {project_name}: {document_title}\n\n"
        f"> Status: Draft for manual Confluence publishing\n"
        f"> Generated by DocuMind using `{model_name}`\n\n"
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

    return draft_path


def generate_markdown_document(
    model_name,
    project_folder,
    prompt,
    document_title,
    filename
):
    if not project_folder:
        return "Upload a project and process it before generating a draft.", None
    if not model_name:
        return "Select an AI model before generating a draft.", None

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

        if not rag_result.get("sources"):
            return rag_result.get("answer", "No indexed code chunks were found."), None

        draft_path = save_markdown_draft(
            project_folder,
            prompt.strip(),
            rag_result,
            target_llm,
            document_title,
            filename
        )
        return (
            "Draft generated. Download the Markdown file below and publish it "
            "to Confluence manually when ready.",
            draft_path
        )
    except Exception as draft_exception:
        return f"Draft generation failed: {str(draft_exception)}", None


def generate_project_document(model_name, project_folder):
    return generate_markdown_document(
        model_name,
        project_folder,
        DEFAULT_CONFLUENCE_PROMPT,
        "Codebase Architecture",
        "confluence_draft.md"
    )


def generate_exceptions_document(model_name, project_folder):
    return generate_markdown_document(
        model_name,
        project_folder,
        EXCEPTIONS_PROMPT,
        "Exceptions and Error Handling",
        "exceptions_draft.md"
    )


# --------------------------------------------------
# Generate Output Function
# --------------------------------------------------
def generate_output(
    model_name,
    project_folder
):
    conn = None
    cursor = None

    results = []

    # Stores chunks from all processed files
    chunk_records = []

    database_status = ""
    indexing_succeeded = False

    # ----------------------------------------------
    # Validate project folder
    # ----------------------------------------------
    if not project_folder:
        return (
            "⚠️ Please enter a project name, select files, "
            "and click 'Upload Files' first.",
            False
        )

    # ----------------------------------------------
    # Validate model selection
    # ----------------------------------------------
    if not model_name:
        return (
            "⚠️ Please select an AI model before processing the project.",
            False
        )

    # ----------------------------------------------
    # Read files from project folder
    # ----------------------------------------------
    file_paths = get_upload_folder_files(
        project_folder
    )

    if not file_paths:
        return (
            "⚠️ No files are present in the project folder:\n"
            f"{project_folder}",
            False
        )

    project_name = os.path.basename(
        os.path.normpath(project_folder)
    )

    try:
        # ----------------------------------------------
        # Connect to PostgreSQL
        # ----------------------------------------------
        conn = psycopg2.connect(
            **DB_CONFIG
        )

        register_vector(
            conn
        )

        cursor = conn.cursor()

        # ----------------------------------------------
        # Process all files
        # ----------------------------------------------
        for path in file_paths:
            filename = os.path.relpath(
                path,
                project_folder
            )
            stored_filename = f"{project_name}/{filename.replace(os.sep, '/')}"

            print(
                f"Processing file: {path}"
            )

            try:
                # --------------------------------------
                # Read file content
                # --------------------------------------
                with open(
                    path,
                    "r",
                    encoding="utf-8"
                ) as file:
                    code = file.read()

                print(
                    f"\nFILE: {filename}"
                )

                print("=" * 80)
                print(code)
                print("=" * 80)

                # --------------------------------------
                # Generate Tree-sitter chunks
                # --------------------------------------
                chunks = tree_sitter_chunk(
                    code
                )

                chunk_text = []
                embedded_text = []

                if chunks:
                    for idx, chunk in enumerate(
                        chunks,
                        start=1
                    ):
                        chunk_type = chunk.get(
                            "type",
                            "Unknown"
                        )

                        chunk_name = chunk.get(
                            "name",
                            "Unknown"
                        )

                        chunk_start_line = chunk.get(
                            "start_line",
                            0
                        )

                        chunk_end_line = chunk.get(
                            "end_line",
                            0
                        )

                        chunk_content = chunk.get(
                            "content",
                            ""
                        )

                        chunk_text.append(
                            f"""
CHUNK Number {idx}
Chunk Type: {chunk_type}
Chunk Name: {chunk_name}
Chunk Start Line: {chunk_start_line}
Chunk End Line: {chunk_end_line}
--------------------------------------------------
Chunk Content:
{chunk_content}
"""
                        )

                        # Do not insert Tree-sitter errors
                        if chunk_type == "ERROR":
                            embedded_text.append(
                                f"""
CHUNK {idx}
Embedding skipped because Tree-sitter parsing failed.

Error:
{chunk_content}
"""
                            )

                            continue

                        # Do not embed empty chunks
                        if not chunk_content.strip():
                            embedded_text.append(
                                f"""
CHUNK {idx}
Embedding skipped because the chunk is empty.
"""
                            )

                            continue

                        try:
                            # ----------------------------------
                            # Generate embedding
                            # ----------------------------------
                            embedding = (
                                generate_chunk_embedding(
                                    chunk_content
                                )
                            )

                            chunk_number = idx

                            # ----------------------------------
                            # Add chunk to bulk insert list
                            # ----------------------------------
                            chunk_records.append(
                                (
                                    stored_filename,
                                    chunk_number,
                                    chunk_type,
                                    chunk_name,
                                    chunk_start_line,
                                    chunk_end_line,
                                    chunk_content,
                                    embedding
                                )
                            )

                            embedded_text.append(
                                f"""
CHUNK {idx}
Chunk ID: {chunk_number}
Embedding Size: {len(embedding)}
First 10 Values:
{embedding[:10].tolist()}

Database Status:
Added to the PostgreSQL bulk-insert list.
"""
                            )

                        except Exception as embedding_exception:
                            embedded_text.append(
                                f"""
CHUNK {idx}
Embedding failed.

Error:
{str(embedding_exception)}

Database Status:
Chunk was not added to the PostgreSQL insert list.
"""
                            )

                else:
                    chunk_text.append(
                        "No chunks found."
                    )

                    embedded_text.append(
                        "No embeddings generated."
                    )

                # --------------------------------------
                # Execute only Python files
                # --------------------------------------
                execution_output = ""
                execution_error = ""

                if path.lower().endswith(".py"):
                    try:
                        execution = subprocess.run(
                            [
                                "python3.12",
                                path
                            ],
                            capture_output=True,
                            text=True,
                            timeout=60,
                            cwd=os.path.dirname(path)
                        )

                        execution_output = (
                            execution.stdout.strip()
                        )

                        execution_error = (
                            execution.stderr.strip()
                        )

                        if not execution_output:
                            execution_output = (
                                "(No output produced)"
                            )

                    except subprocess.TimeoutExpired:
                        execution_output = (
                            "Execution stopped because the file "
                            "exceeded the 60-second timeout."
                        )

                    except Exception as execution_exception:
                        execution_output = (
                            "Python file execution failed.\n\n"
                            f"Error: "
                            f"{str(execution_exception)}"
                        )

                else:
                    execution_output = (
                        "Execution skipped because this is "
                        "not a Python file."
                    )

                # --------------------------------------
                # Build output for current file
                # --------------------------------------
                result_text = f"""
================================================================================
FILE: {filename}
FULL PATH: {path}
================================================================================

TREE-SITTER CHUNKS
================================================================================

{''.join(chunk_text)}

EMBEDDED CHUNKS
================================================================================

{''.join(embedded_text)}

EXECUTION OUTPUT
================================================================================

{execution_output}
"""

                if execution_error:
                    result_text += f"""

EXECUTION ERRORS
================================================================================

{execution_error}
"""

                results.append(
                    result_text
                )

            except UnicodeDecodeError:
                results.append(
                    f"""
================================================================================
FILE: {filename}
================================================================================

SKIPPED

The file is not a UTF-8 text file, so its contents could not be read.
"""
                )

            except Exception as file_exception:
                results.append(
                    f"""
================================================================================
FILE: {filename}
================================================================================

FAILED

Error: {str(file_exception)}
"""
                )

        # ----------------------------------------------
        # Insert all chunks after all files finish
        # ----------------------------------------------
        inserted_count = insert_chunks_into_postgres(
            cursor,
            chunk_records
        )

        # Commit once for all chunks
        conn.commit()
        indexing_succeeded = inserted_count > 0

        database_status = (
            "\n\nPOSTGRESQL BULK INSERT\n"
            "============================================"
            "====================================\n\n"
            f"✅ Successfully inserted "
            f"{inserted_count} chunk(s) into:\n"
            "documind_nprd.chunk_details"
        )

    except Exception as database_exception:
        if conn is not None:
            conn.rollback()

        database_status = (
            "\n\nPOSTGRESQL BULK INSERT\n"
            "============================================"
            "====================================\n\n"
            "❌ PostgreSQL bulk insert failed.\n\n"
            f"Error: {str(database_exception)}\n\n"
            "The transaction was rolled back."
        )

    finally:
        if cursor is not None:
            cursor.close()

        if conn is not None:
            conn.close()

    # --------------------------------------------------
    # AI RAG Retrieval & Documentation Generation
    # --------------------------------------------------
    ai_generation_output = ""
    try:
        from retrieval_postgres import retrieve_from_postgres
        target_llm = map_ollama_model(model_name)
        print(f"Generating RAG output using model: {target_llm} from documind_nprd.chunk_details...")

        rag_result = retrieve_from_postgres(
            query="Summarize the codebase architecture, key classes, methods, and responsibilities.",
            db_config=DB_CONFIG,
            table_name="documind_nprd.chunk_details",
            top_k=6,
            llm_model=target_llm,
            prompt_type="standard",
            project_filter=project_name,
        )

        ai_generation_output = f"""
================================================================================
🤖 AI CODE DOCUMENTATION & INSIGHTS (Generated via Local LLM: {target_llm})
================================================================================

{rag_result['answer']}
"""
    except Exception as rag_err:
        ai_generation_output = f"""
================================================================================
⚠️ AI Documentation Generation Notice:
{str(rag_err)}
================================================================================
"""

    return (
        f"📂 Project folder: {project_folder}\n\n"
        f"📄 Processing {len(file_paths)} file(s)\n\n"
        f"🤖 Selected Model: {model_name}\n\n"
        + ai_generation_output + "\n\n"
        + database_status + "\n\n"
        + "\n".join(results)
        + "\n\n✅ Output generation completed.",
        indexing_succeeded
    )



# --------------------------------------------------
# UI
# --------------------------------------------------
def build_ui():
    with gr.Blocks(
        title="DocuMind",
        css="""
        #model_radio .wrap {
            flex-direction: column !important;
            align-items: flex-start !important;
        }

        #model_radio .wrap label {
            width: 100% !important;
        }
        """
    ) as demo:

        # Stores the project folder after
        # a successful upload
        project_folder_state = gr.State(
            value=None
        )
        project_indexed_state = gr.State(
            value=False
        )

        gr.Markdown(
            """
<h1 style="text-align: center;">
🧠 DocuMind Dashboard
</h1>

<p style="text-align: center;">
Enter the project name, upload your documents, select an AI model,
and generate insights from your files.
</p>
"""
        )

        # --------------------------------------------------
        # Project Name
        # --------------------------------------------------
        with gr.Row():
            project_name = gr.Textbox(
                label="Project Name",
                placeholder=(
                    "Enter project name, for example: projectA"
                ),
                scale=1
            )

        # --------------------------------------------------
        # File Upload Section
        # --------------------------------------------------
        with gr.Row():

            with gr.Column(scale=1):
                files = gr.File(
                    label="Select Files to Upload",
                    file_count="directory",
                    type="filepath"
                )

            with gr.Column(scale=1):
                upload_btn = gr.Button(
                    "📤 Upload Files",
                    variant="secondary",
                    min_width=120,
                    size="sm",
                    interactive=True
                )

                upload_status = gr.Textbox(
                    label="Upload Status",
                    lines=8,
                    max_lines=12,
                    interactive=False
                )

        # --------------------------------------------------
        # Model Selection Section
        # --------------------------------------------------
        with gr.Row():

            with gr.Column(scale=1):
                pass

            with gr.Column(scale=2):
                model_selector = gr.Radio(
                    choices=[
                        "Qwen2.5-Coder-7B-Instruct",
                        "Phi-3.5-min-instruct",
                        "Deepseek-Coder-6.7BInstruct"
                    ],
                    label="Select Model",
                    value=None,
                    interactive=False,
                    elem_id="model_radio"
                )

            with gr.Column(scale=1):
                pass

        # --------------------------------------------------
        # Project Processing and Documentation Section
        # --------------------------------------------------
        with gr.Row():

            with gr.Column(scale=2):
                process_btn = gr.Button(
                    "Process & Index Project",
                    variant="primary",
                    min_width=120,
                    size="sm",
                    interactive=False
                )

                output_box = gr.Textbox(
                    label="Processing Output",
                    lines=25,
                    max_lines=40,
                    interactive=False
                )

                gr.Markdown(
                    "### Generate Documents\n"
                    "Confluence is not connected. Generated Markdown files "
                    "can be downloaded and published manually."
                )
                with gr.Row():
                    document_btn = gr.Button(
                        "Generate Document",
                        variant="secondary",
                        interactive=False
                    )
                    exceptions_btn = gr.Button(
                        "Generate Exceptions",
                        variant="secondary",
                        interactive=False
                    )
                with gr.Row():
                    with gr.Column():
                        document_status = gr.Textbox(
                            label="Document status",
                            interactive=False
                        )
                        document_file = gr.File(
                            label="Download document",
                            interactive=False
                        )
                    with gr.Column():
                        exceptions_status = gr.Textbox(
                            label="Exceptions status",
                            interactive=False
                        )
                        exceptions_file = gr.File(
                            label="Download exceptions document",
                            interactive=False
                        )
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("---")
                gr.Markdown("### Ask DocuMind")
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Question or Prompt",
                        placeholder="e.g. Explain how database connection works, or list all classes and exceptions",
                        lines=2,
                        scale=4,
                    )
                    ask_btn = gr.Button(
                        "Ask DocuMind",
                        variant="secondary",
                        scale=1,
                        interactive=False
                    )

                query_answer_box = gr.Textbox(
                    label="AI Answer",
                    lines=10,
                    interactive=False,
                )

        def on_ask_code(user_q, chosen_model, active_project_folder):
            if not user_q or not user_q.strip():
                return "⚠️ Please enter a question or query."
            if not active_project_folder:
                return "⚠️ Upload and process a project before asking questions."
            try:
                from retrieval_postgres import retrieve_from_postgres
                target_llm = map_ollama_model(chosen_model)
                q_clean = user_q.strip()
                is_doc = any(w in q_clean.lower() for w in ["confluence", "architect", "overview", "document", "high level", "summary", "summarize"])
                res = retrieve_from_postgres(
                    query=q_clean,
                    db_config=DB_CONFIG,
                    table_name="documind_nprd.chunk_details",
                    top_k=7 if is_doc else 5,
                    llm_model=target_llm,
                    prompt_type="standard" if is_doc else "qa",
                    project_filter=os.path.basename(
                        os.path.normpath(active_project_folder)
                    ),
                )
                return res["answer"]
            except Exception as e:
                return f"❌ Query error: {str(e)}"

        ask_btn.click(
            fn=on_ask_code,
            inputs=[query_input, model_selector, project_folder_state],
            outputs=[query_answer_box],
            api_name=False,
        )

        # --------------------------------------------------
        # Reset workflow if project name changes
        # --------------------------------------------------
        project_name.change(
            fn=reset_project_workflow,
            inputs=[],
            outputs=[
                project_folder_state,
                upload_status,
                model_selector,
                process_btn,
                output_box,
                project_indexed_state,
                document_btn,
                document_status,
                document_file,
                exceptions_btn,
                exceptions_status,
                exceptions_file,
                ask_btn,
                query_answer_box
            ],
            api_name=False
        )

        # --------------------------------------------------
        # Reset workflow if selected files change
        # --------------------------------------------------
        files.change(
            fn=reset_file_workflow,
            inputs=[],
            outputs=[
                project_folder_state,
                upload_status,
                model_selector,
                process_btn,
                output_box,
                project_indexed_state,
                document_btn,
                document_status,
                document_file,
                exceptions_btn,
                exceptions_status,
                exceptions_file,
                ask_btn,
                query_answer_box
            ],
            api_name=False
        )

        # --------------------------------------------------
        # Upload Button Action
        # --------------------------------------------------
        upload_btn.click(
            fn=process_upload,
            inputs=[
                files,
                project_name
            ],
            outputs=[
                upload_status,
                project_folder_state,
                model_selector
            ],
            api_name=False
        )

        # --------------------------------------------------
        # Enable Generate button after model selection
        # --------------------------------------------------
        model_selector.change(
            fn=update_generate_button,
            inputs=[
                model_selector,
                project_folder_state,
                project_indexed_state
            ],
            outputs=[
                process_btn,
                document_btn,
                exceptions_btn,
                ask_btn
            ],
            api_name=False
        )

        project_indexed_state.change(
            fn=update_generate_button,
            inputs=[
                model_selector,
                project_folder_state,
                project_indexed_state
            ],
            outputs=[
                process_btn,
                document_btn,
                exceptions_btn,
                ask_btn
            ],
            api_name=False
        )

        # --------------------------------------------------
        # Generate Button Action
        # --------------------------------------------------
        process_btn.click(
            fn=generate_output,
            inputs=[
                model_selector,
                project_folder_state
            ],
            outputs=[
                output_box,
                project_indexed_state
            ],
            api_name=False
        )

        document_btn.click(
            fn=generate_project_document,
            inputs=[
                model_selector,
                project_folder_state
            ],
            outputs=[
                document_status,
                document_file
            ],
            api_name=False
        )

        exceptions_btn.click(
            fn=generate_exceptions_document,
            inputs=[
                model_selector,
                project_folder_state
            ],
            outputs=[
                exceptions_status,
                exceptions_file
            ],
            api_name=False
        )

    return demo


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    demo = build_ui()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True
    )
