import gradio as gr
import os
import shutil
import subprocess

import numpy as np
import ollama
import psycopg2

from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from tree_sitter import Language, Parser
import tree_sitter_python as tspython


# --------------------------------------------------
# Tree-sitter Python language
# --------------------------------------------------
PY_LANGUAGE = Language(tspython.language())


# --------------------------------------------------
# Base target upload folder
# --------------------------------------------------
BASE_UPLOAD_FOLDER = "/home/azureuser/proj-documind/test/repo"

os.makedirs(
    BASE_UPLOAD_FOLDER,
    exist_ok=True
)


# --------------------------------------------------
# PostgreSQL configuration
# --------------------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "database": "documind_dev",
    "user": "vadmin",
    "password": "documind@123"
}


# --------------------------------------------------
# Validate and set project folder
# --------------------------------------------------
def set_project_name(project_name):
    if not project_name or not project_name.strip():
        return (
            "⚠️ Please enter a project name.",
            None,
            gr.update(interactive=False),
            gr.update(interactive=False)
        )

    project_name = project_name.strip()

    # Prevent absolute paths and directory traversal
    if (
        project_name in (".", "..")
        or "/" in project_name
        or "\\" in project_name
        or os.path.isabs(project_name)
    ):
        return (
            (
                "⚠️ Invalid project name.\n\n"
                "Enter only the project name. Do not include `/`, `\\`, "
                "`.` or `..` as folder paths."
            ),
            None,
            gr.update(interactive=False),
            gr.update(interactive=False)
        )

    project_folder = os.path.join(
        BASE_UPLOAD_FOLDER,
        project_name
    )

    try:
        os.makedirs(
            project_folder,
            exist_ok=True
        )

        return (
            (
                f"✅ Project selected successfully.\n\n"
                f"Project Name: {project_name}\n"
                f"Target Folder: {project_folder}"
            ),
            project_folder,
            gr.update(interactive=True),
            gr.update(interactive=True)
        )

    except Exception as e:
        return (
            (
                "❌ Unable to create the project folder.\n\n"
                f"Error: {str(e)}"
            ),
            None,
            gr.update(interactive=False),
            gr.update(interactive=False)
        )


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
def process_upload(files, project_folder):
    if not project_folder:
        return (
            "⚠️ Please enter a project name and click "
            "'Proceed with Project' first."
        )

    if not files:
        return "⚠️ No files uploaded."

    os.makedirs(
        project_folder,
        exist_ok=True
    )

    saved_paths = []

    file_list = (
        files
        if isinstance(files, list)
        else [files]
    )

    for file_obj in file_list:
        try:
            temp_path = (
                file_obj.name
                if hasattr(file_obj, "name")
                else str(file_obj)
            )

            filename = os.path.basename(
                temp_path
            )

            destination = os.path.join(
                project_folder,
                filename
            )

            shutil.copyfile(
                temp_path,
                destination
            )

            saved_paths.append(
                destination
            )

        except Exception as e:
            saved_paths.append(
                f"ERROR: Unable to save {file_obj}: {str(e)}"
            )

    successfully_saved = [
        path
        for path in saved_paths
        if not path.startswith("ERROR:")
    ]

    errors = [
        path
        for path in saved_paths
        if path.startswith("ERROR:")
    ]

    status = (
        f"✅ Saved {len(successfully_saved)} file(s) to:\n"
        f"{project_folder}\n\n"
    )

    if successfully_saved:
        status += "\n".join(
            f"• {os.path.basename(path)}"
            for path in successfully_saved
        )

    if errors:
        status += (
            "\n\nUpload errors:\n"
            + "\n".join(errors)
        )

    return status


# --------------------------------------------------
# Tree-sitter Chunking Function
# --------------------------------------------------
def tree_sitter_chunk(code):
    chunks = []

    try:
        source_bytes = code.encode("utf-8")

        parser = Parser(PY_LANGUAGE)

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
                        name_node.start_byte:name_node.end_byte
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
                        "start_line": node.start_point.row + 1,
                        "end_line": node.end_point.row + 1,
                        "content": chunk_content
                    }
                )

            # Recursively check all named child nodes
            for child in node.named_children:
                traverse_node(child)

        traverse_node(root_node)

        # Fallback for Python files with no
        # functions or classes
        if not chunks and code.strip():
            chunks.append(
                {
                    "type": "module",
                    "name": "module_level_code",
                    "start_line": 1,
                    "end_line": len(code.splitlines()),
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

    if hasattr(response, "embeddings"):
        embedding = response.embeddings[0]
    else:
        embedding = response["embeddings"][0]

    # Convert to numpy array for pgvector
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
# Bulk insert all chunk records into PostgreSQL
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
        template="(%s, %s, %s, %s, %s, %s, %s, %s)",
        page_size=100
    )

    return len(chunk_records)


# --------------------------------------------------
# Read all file contents from project folder
# --------------------------------------------------
def show_file_contents(project_folder):
    if not project_folder:
        return (
            "⚠️ Please enter a project name and click "
            "'Proceed with Project' first."
        )

    file_paths = get_upload_folder_files(
        project_folder
    )

    if not file_paths:
        return (
            "⚠️ No files are present in the project folder:\n"
            f"{project_folder}"
        )

    output = []

    for path in file_paths:
        filename = os.path.relpath(
            path,
            project_folder
        )

        try:
            with open(
                path,
                "r",
                encoding="utf-8"
            ) as file:
                code = file.read()

            output.append(
                f"""
================================================================================
FILE: {filename}
PATH: {path}
================================================================================

{code}
"""
            )

        except UnicodeDecodeError:
            output.append(
                f"""
================================================================================
FILE: {filename}
================================================================================

⚠️ File skipped because it is not a UTF-8 text file.
"""
            )

        except Exception as e:
            output.append(
                f"""
================================================================================
FILE: {filename}
================================================================================

❌ Unable to read file.

Error: {str(e)}
"""
            )

    return "\n".join(output)


# --------------------------------------------------
# Generate Output Function
# --------------------------------------------------
def generate_output(model_name, project_folder):
    conn = None
    cursor = None

    results = []

    # This list stores chunks from all processed files
    chunk_records = []

    if not project_folder:
        return (
            "⚠️ Please enter a project name and click "
            "'Proceed with Project' first."
        )

    file_paths = get_upload_folder_files(
        project_folder
    )

    if not file_paths:
        return (
            "⚠️ No files are present in the project folder:\n"
            f"{project_folder}"
        )

    try:
        # --------------------------------------------------
        # Connect to PostgreSQL
        # --------------------------------------------------
        conn = psycopg2.connect(
            **DB_CONFIG
        )

        register_vector(conn)

        cursor = conn.cursor()

        # --------------------------------------------------
        # Process all files
        # --------------------------------------------------
        for path in file_paths:
            filename = os.path.relpath(
                path,
                project_folder
            )

            print(
                f"Processing file: {path}"
            )

            try:
                # --------------------------------------------------
                # Read file from project folder
                # --------------------------------------------------
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

                # --------------------------------------------------
                # Generate Tree-sitter chunks
                # --------------------------------------------------
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

                        # Do not insert Tree-sitter error chunks
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

                        # Do not embed or insert empty chunks
                        if not chunk_content.strip():
                            embedded_text.append(
                                f"""
CHUNK {idx}
Embedding skipped because the chunk is empty.
"""
                            )

                            continue

                        try:
                            # --------------------------------------------------
                            # Generate the chunk embedding
                            # --------------------------------------------------
                            embedding = generate_chunk_embedding(
                                chunk_content
                            )

                            # --------------------------------------------------
                            # Set chunk number
                            # --------------------------------------------------
                            chunk_number = idx

                            # --------------------------------------------------
                            # Append all chunk details to the list
                            # --------------------------------------------------
                            chunk_records.append(
                                (
                                    filename,
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

                # --------------------------------------------------
                # Execute only Python files
                # --------------------------------------------------
                execution_output = ""
                execution_error = ""

                if path.lower().endswith(".py"):
                    try:
                        execution = subprocess.run(
                            ["python3.12", path],
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
                            f"Error: {str(execution_exception)}"
                        )

                else:
                    execution_output = (
                        "Execution skipped because this is "
                        "not a Python file."
                    )

                # --------------------------------------------------
                # Build result for current file
                # --------------------------------------------------
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

            except Exception as e:
                results.append(
                    f"""
================================================================================
FILE: {filename}
================================================================================

FAILED

Error: {str(e)}
"""
                )

        # --------------------------------------------------
        # Insert all chunks after all files finish
        # --------------------------------------------------
        inserted_count = insert_chunks_into_postgres(
            cursor,
            chunk_records
        )

        # Commit once for all chunks
        conn.commit()

        database_status = (
            "\n\nPOSTGRESQL BULK INSERT\n"
            "================================================================================\n\n"
            f"✅ Successfully inserted {inserted_count} chunk(s) into:\n"
            "documind_nprd.chunk_details"
        )

    except Exception as database_exception:
        if conn is not None:
            conn.rollback()

        database_status = (
            "\n\nPOSTGRESQL BULK INSERT\n"
            "================================================================================\n\n"
            "❌ PostgreSQL bulk insert failed.\n\n"
            f"Error: {str(database_exception)}\n\n"
            "The transaction was rolled back."
        )

    finally:
        if cursor is not None:
            cursor.close()

        if conn is not None:
            conn.close()

    return (
        f"📂 Project folder: {project_folder}\n\n"
        f"📄 Processing {len(file_paths)} file(s)\n\n"
        f"🤖 Selected Model: {model_name}\n\n"
        + "\n".join(results)
        + database_status
        + "\n\n✅ Output generation completed."
    )


# --------------------------------------------------
# Combined Upload Action
# --------------------------------------------------
def upload_and_show(
    files,
    project_folder
):
    if not project_folder:
        message = (
            "⚠️ Please enter a project name and click "
            "'Proceed with Project' first."
        )

        return message, message

    upload_status = process_upload(
        files,
        project_folder
    )

    # Read files from the project folder
    # after they are copied
    file_content = show_file_contents(
        project_folder
    )

    return (
        upload_status,
        file_content
    )


# --------------------------------------------------
# UI
# --------------------------------------------------
# --------------------------------------------------
# UI
# --------------------------------------------------
def build_ui():
    with gr.Blocks(
        title="DocuMind"
    ) as demo:

        # Stores the selected project folder
        # for the current Gradio session
        project_folder_state = gr.State(
            value=None
        )

        gr.Markdown(
            """
# 🧠 DocuMind Dashboard

Enter the project name, upload your documents, select an AI model,
and generate insights from your files.
"""
        )

        # --------------------------------------------------
        # PROJECT SECTION
        # --------------------------------------------------
        with gr.Row():

            project_name = gr.Textbox(
                label="Project Name",
                placeholder="Enter project name, for example: projectA",
                scale=1
            )

            project_btn = gr.Button(
                "✅ Proceed with Project Name",
                variant="primary",
                min_width=80,
                scale=1
            )

            project_status = gr.Textbox(
                label="Project Status",
                lines=3,
                max_lines=5,
                interactive=False
            )

        with gr.Row():

            # --------------------------------------------------
            # LEFT PANEL
            # --------------------------------------------------
            with gr.Column(scale=1):
                files = gr.File(
                    label="Upload Files",
                    file_count="directory"
                )

            # --------------------------------------------------
            # MIDDLE PANEL
            # --------------------------------------------------
            with gr.Column(scale=1):

                upload_btn = gr.Button(
                    "📤 Upload Files",
                    variant="secondary",
                    min_width=120,
                    interactive=False
                )

                upload_status = gr.Textbox(
                    label="Upload Status",
                    lines=8,
                    max_lines=10,
                    interactive=False
                )

            with gr.Column(scale=1):

                code_box = gr.Textbox(
                    label="Uploaded File Content",
                    lines=20,
                    max_lines=30,
                    interactive=False
                )

        with gr.Row():

            model_selector = gr.Radio(
                choices=[
                    "Qwen2.5-Coder-7B-Instruct",
                    "Phi-3.5-min-instruct",
                    "Deepseek-Coder-6.7BInstruct"
                ],
                label="Select Model",
                value="Qwen2.5-Coder-7B-Instruct"
            )

        with gr.Row():

            # --------------------------------------------------
            # RIGHT PANEL
            # --------------------------------------------------
            with gr.Column(scale=2):

                generate_btn = gr.Button(
                    "🚀 Generate Output",
                    variant="primary",
                    min_width=120,
                    interactive=False
                )

                output_box = gr.Textbox(
                    label="Generated Output",
                    lines=25,
                    max_lines=40,
                    interactive=False
                )

        # --------------------------------------------------
        # Project Button Action
        # --------------------------------------------------
        project_btn.click(
            fn=set_project_name,
            inputs=[
                project_name
            ],
            outputs=[
                project_status,
                project_folder_state,
                upload_btn,
                generate_btn
            ],
            api_name=False
        )

        # Allow Enter key to proceed with project
        project_name.submit(
            fn=set_project_name,
            inputs=[
                project_name
            ],
            outputs=[
                project_status,
                project_folder_state,
                upload_btn,
                generate_btn
            ],
            api_name=False
        )

        # --------------------------------------------------
        # Upload Button Action
        # --------------------------------------------------
        upload_btn.click(
            fn=upload_and_show,
            inputs=[
                files,
                project_folder_state
            ],
            outputs=[
                upload_status,
                code_box
            ],
            api_name=False
        )

        # --------------------------------------------------
        # Generate Button Action
        # --------------------------------------------------
        generate_btn.click(
            fn=generate_output,
            inputs=[
                model_selector,
                project_folder_state
            ],
            outputs=[
                output_box
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