"""Project uploads, Python chunking, embeddings, and indexing."""

import os
import shutil
import subprocess
from time import perf_counter

import gradio as gr
import numpy as np
import ollama
import psycopg2
import psycopg2.extensions
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

from config import BASE_UPLOAD_FOLDER, DB_CONFIG
from retrieval_postgres import map_ollama_model


def _adapt_numpy_array(arr):
    """Convert a NumPy vector into a PostgreSQL real-array expression."""
    return psycopg2.extensions.AsIs(f"ARRAY{arr.tolist()}::real[]")


psycopg2.extensions.register_adapter(np.ndarray, _adapt_numpy_array)

PY_LANGUAGE = Language(tspython.language())


def get_upload_folder_files(project_folder):
    """Find the files stored beneath the selected project directory."""
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


def tree_sitter_chunk(code):
    """Extract Python function and class chunks with names and source locations."""
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
            root_node.type,
            flush=True
        )

        print(
            "Tree-sitter syntax error:",
            root_node.has_error,
            flush=True
        )

        def traverse_node(node):
            """Visit syntax nodes recursively and collect function or class definitions."""
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

        # Flush stage markers so a native crash can be localized in server logs.
        print("Tree-sitter traversal starting", flush=True)
        traverse_node(
            root_node
        )
        print(f"Tree-sitter traversal completed: {len(chunks)} definition chunks", flush=True)

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


def generate_chunk_embedding(chunk_content):
    """Ask Ollama to represent one code chunk as a float32 search vector."""
    print(f"Embedding request starting: {len(chunk_content)} characters", flush=True)
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

    print(f"Embedding response received: {len(embedding)} dimensions", flush=True)
    # Convert to NumPy array for pgvector
    return np.array(
        embedding,
        dtype=np.float32
    )


def embedding_code(chunks):
    """Generate embedding previews or failure messages for the supplied chunks."""
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


def insert_chunks_into_postgres(
    cursor,
    chunk_records
):
    """Insert prepared chunk records in bulk and return the inserted count."""
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


def generate_output(
    model_name,
    project_folder
):
    """Index project files, execute Python inputs, and return processing status."""
    started_at = perf_counter()
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
            f"Indexing: {inserted_count} chunk(s) saved."
            if indexing_succeeded else "Indexing: No chunks were saved."
        )

    except Exception as database_exception:
        if conn is not None:
            conn.rollback()

        database_status = (
            f"Indexing failed: {str(database_exception)}. Transaction rolled back."
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
    target_llm = "Not resolved"
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

        if "LLM Generation Error:" in rag_result.get("answer", ""):
            ai_generation_output = rag_result["answer"]
        else:
            ai_generation_output = (
                "Overview: Generated." if rag_result.get("sources")
                else "Overview: No matching code chunks found."
            )
    except Exception as rag_err:
        ai_generation_output = f"Overview failed: {str(rag_err)}"

    print("\n".join(results))
    return (
        f"Processing finished in {perf_counter() - started_at:.1f} seconds.\n"
        f"Files found: {len(file_paths)}\n"
        "Embedding model: mxbai-embed-large\n"
        f"Generation model: {target_llm}\n"
        f"{database_status}\n{ai_generation_output}",
        indexing_succeeded
    )
