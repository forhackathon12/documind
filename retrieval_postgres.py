"""
retrieval_postgres.py – Retrieval Pipeline for PostgreSQL (DocuMind)
=====================================================================
Responsibility: Connect directly to the database populated by `ingestionpipeline.py`
and retrieve matching code chunks / generate answers without any re-ingestion.

Database Schema targeted from `ingestionpipeline.py`:
  • Table:  documind_nprd.chunk_details
  • Fields: file_name, chunk_number, chunk_type, chunk_name,
            chunk_start_line, chunk_end_line, chunk_content, chunk_embeddings
  • Model:  mxbai-embed-large (1024 dimensions via Ollama)

Features:
  • Generates query embeddings using `mxbai-embed-large` via Ollama
  • Executes cosine similarity search (<=>) directly on pgvector
  • Formats retrieved chunks with file name, type, and line ranges
  • Anti-hallucination prompt formatting with strict context boundaries
  • Generates answers via local Ollama LLM (Qwen, Phi, DeepSeek, etc.)
  • Standalone CLI & callable Python API

Usage (Python):
    from retrieval_postgres import retrieve_from_postgres, run_postgres

    result = retrieve_from_postgres(
        query="Explain how tree-sitter chunking is performed",
        top_k=5,
        llm_model="qwen2.5:0.5b"
    )
    print(result["answer"])
    for s in result["sources"]:
        print(f"[{s['file_name']}:{s['chunk_start_line']}] {s['chunk_name']}")

Usage (CLI):
    python3 retrieval_postgres.py "What does the main function do?" --k 5
"""

import os
import argparse
from typing import Optional, List, Dict, Any, Union

import numpy as np
import ollama
import psycopg2
from pgvector.psycopg2 import register_vector

# Import database configuration directly from friend's ingestion pipeline
try:
    from ingestionpipeline import DB_CONFIG as INGESTION_DB_CONFIG
except ImportError:
    INGESTION_DB_CONFIG = {
        "host": "localhost",
        "database": "documind_dev",
        "user": "vadmin",
        "password": "documind@123"
    }

# Default target table from ingestionpipeline.py
DEFAULT_TABLE = "documind_nprd.chunk_details"
DEFAULT_EMBED_MODEL = "mxbai-embed-large"
DEFAULT_LLM_MODEL = "qwen2.5:0.5b"


# ── Database Connection Helper ────────────────────────────────────────────────

def get_postgres_connection(db_config: Optional[Dict[str, Any]] = None):
    """
    Establish a connection to the PostgreSQL vector database and register pgvector.

    Args:
        db_config: Dict with host, database, user, password, port (optional).
                   Defaults to INGESTION_DB_CONFIG or environment variables.

    Returns:
        psycopg2 connection object with pgvector registered.
    """
    cfg = dict(db_config or INGESTION_DB_CONFIG)

    # Allow overriding from environment variables if set
    if os.environ.get("PGHOST"):
        cfg["host"] = os.environ.get("PGHOST")
    if os.environ.get("PGDATABASE"):
        cfg["database"] = os.environ.get("PGDATABASE")
    if os.environ.get("PGUSER"):
        cfg["user"] = os.environ.get("PGUSER")
    if os.environ.get("PGPASSWORD"):
        cfg["password"] = os.environ.get("PGPASSWORD")
    if os.environ.get("PGPORT"):
        cfg["port"] = int(os.environ.get("PGPORT"))

    try:
        conn = psycopg2.connect(**cfg)
        try:
            register_vector(conn)
        except Exception:
            pass
        return conn
    except Exception as primary_err:
        # If default credentials fail, check if PGVECTOR_CONNECTION_STRING is provided
        raw_conn_str = os.environ.get("PGVECTOR_CONNECTION_STRING")
        if raw_conn_str:
            try:
                # Convert sqlalchemy url if necessary
                clean_str = raw_conn_str.replace("postgresql+psycopg://", "postgresql://").replace("postgresql+psycopg2://", "postgresql://")
                conn = psycopg2.connect(clean_str)
                try:
                    register_vector(conn)
                except Exception:
                    pass
                return conn
            except Exception:
                pass

        raise ConnectionError(
            f"Could not connect to PostgreSQL database '{cfg.get('database')}' at '{cfg.get('host')}': {primary_err}\n"
            f"Make sure PostgreSQL is running with credentials: user={cfg.get('user')}, host={cfg.get('host')}"
        )


# ── Embedding Generation ──────────────────────────────────────────────────────

def generate_query_embedding(
    query_text: str,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> np.ndarray:
    """
    Generate vector embedding for the query using Ollama.
    Matches the `generate_chunk_embedding` logic in `ingestionpipeline.py`.

    Args:
        query_text: The user query string.
        model_name: Ollama embedding model (default: "mxbai-embed-large").

    Returns:
        numpy float32 vector array.
    """
    try:
        response = ollama.embed(
            model=model_name,
            input=query_text,
        )
        if hasattr(response, "embeddings"):
            embedding = response.embeddings[0]
        else:
            embedding = response["embeddings"][0]

        return np.array(embedding, dtype=np.float32)

    except Exception as e:
        raise RuntimeError(
            f"Failed to generate query embedding with model '{model_name}': {e}\n"
            f"Run `ollama pull {model_name}` to download the model."
        )


# ── Vector Search Query ───────────────────────────────────────────────────────

def _build_chunk_scope_filter(
    file_filter: Optional[str],
    project_filter: Optional[str],
) -> tuple:
    clauses = []
    params = []

    if file_filter:
        clauses.append("file_name = %s")
        params.append(file_filter)

    if project_filter:
        project_prefix = f"{project_filter.rstrip('/')}/"
        clauses.append("left(file_name, %s) = %s")
        params.extend([len(project_prefix), project_prefix])

    if not clauses:
        return "", params

    return " WHERE " + " AND ".join(clauses), params


def search_similar_chunks(
    conn,
    query_embedding: np.ndarray,
    table_name: str = DEFAULT_TABLE,
    top_k: int = 5,
    file_filter: Optional[str] = None,
    project_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Execute cosine similarity search on the table `documind_nprd.chunk_details`.
    Supports both native pgvector (<=>) and array embeddings via numpy.
    """
    cursor = conn.cursor()

    # 1. Try native pgvector <=> operator
    try:
        base_query = f"""
            SELECT
                file_name,
                chunk_number,
                chunk_type,
                chunk_name,
                chunk_start_line,
                chunk_end_line,
                chunk_content,
                (chunk_embeddings <=> %s) AS distance
            FROM {table_name}
        """
        filter_sql, filter_params = _build_chunk_scope_filter(
            file_filter,
            project_filter
        )
        base_query += filter_sql
        params = [query_embedding, *filter_params]
        base_query += " ORDER BY chunk_embeddings <=> %s ASC LIMIT %s "
        params.extend([query_embedding, top_k])

        cursor.execute(base_query, params)
        rows = cursor.fetchall()
        results = []
        for row in rows:
            dist = float(row[7]) if row[7] is not None else 1.0
            results.append({
                "file_name": row[0],
                "chunk_number": row[1],
                "chunk_type": row[2],
                "chunk_name": row[3],
                "chunk_start_line": row[4],
                "chunk_end_line": row[5],
                "chunk_content": row[6],
                "distance": dist,
                "similarity": round(max(0.0, 1.0 - dist), 4),
                "preview": (row[6][:220] + "…") if len(row[6]) > 220 else row[6],
            })
        cursor.close()
        return results

    except Exception:
        # 2. Resilient fallback: fetch chunks and compute cosine similarity via numpy
        conn.rollback()
        cursor = conn.cursor()

        select_query = f"""
            SELECT
                file_name,
                chunk_number,
                chunk_type,
                chunk_name,
                chunk_start_line,
                chunk_end_line,
                chunk_content,
                chunk_embeddings
            FROM {table_name}
        """
        filter_sql, params = _build_chunk_scope_filter(
            file_filter,
            project_filter
        )
        select_query += filter_sql

        try:
            cursor.execute(select_query, params)
            rows = cursor.fetchall()
        except Exception as query_err:
            cursor.close()
            raise RuntimeError(
                f"Database query error against '{table_name}': {query_err}\n"
                f"Verify that table '{table_name}' exists."
            )

        cursor.close()
        if not rows:
            return []

        scored = []
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        for r in rows:
            if r[7] is None:
                continue
            emb = np.array(r[7], dtype=np.float32)
            c_norm = emb / (np.linalg.norm(emb) + 1e-10)
            sim = float(np.dot(q_norm, c_norm))
            dist = max(0.0, 1.0 - sim)

            scored.append({
                "file_name": r[0],
                "chunk_number": r[1],
                "chunk_type": r[2],
                "chunk_name": r[3],
                "chunk_start_line": r[4],
                "chunk_end_line": r[5],
                "chunk_content": r[6],
                "distance": dist,
                "similarity": round(sim, 4),
                "preview": (r[6][:220] + "…") if len(r[6]) > 220 else r[6],
            })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]



# ── Anti-Hallucination Prompt & Generation ───────────────────────────────────

def build_anti_hallucination_prompt(
    query: str,
    chunks: List[Dict[str, Any]],
    prompt_type: str = "standard",
    custom_instructions: str = "",
) -> tuple:
    """
    Construct grounded system and user prompt messages.
    Separates persona into system role and retrieved code + task into user role.
    """
    context_blocks = []
    for idx, c in enumerate(chunks, start=1):
        content = c.get("chunk_content", "")
        # Cap single chunk content to 1800 chars so prompt fits comfortably in model context window
        if len(content) > 1800:
            content = content[:1800] + "\n# ... [truncated for context length] ..."
        context_blocks.append(
            f"### Source File: {c.get('file_name', 'unknown')} | Type: {c.get('chunk_type', '')} | Name: {c.get('chunk_name', '')} (Lines {c.get('chunk_start_line', '')}–{c.get('chunk_end_line', '')})\n"
            f"```python\n{content}\n```"
        )

    formatted_context = "\n\n".join(context_blocks)

    q_lower = query.lower()
    is_doc_request = any(w in q_lower for w in ["confluence", "architect", "document", "high level", "overview", "summary", "summarize"])

    if prompt_type == "standard" or is_doc_request:
        system_msg = (
            "You are a Principal Solutions Architect writing an enterprise Atlassian Confluence Technical Specification.\n\n"
            "CRITICAL CONFLUENCE GUIDELINES:\n"
            "1. NEVER copy-paste or dump raw Python code blocks into the document. Explain architecture, component logic, data flows, inputs, and outputs in clear technical prose and markdown tables.\n"
            "2. Structure the document following industry-standard Confluence Architecture templates:\n"
            "   - **Document Control & Metadata**: Status (APPROVED), Architect, Target Stack\n"
            "   - **1. Executive Summary & Purpose**: Business context and technical goals\n"
            "   - **2. End-to-End Architecture & Workflow**: Numbered sequential data flow\n"
            "   - **3. Component Interface Table**: Markdown table with Component, Role, Inputs, Outputs\n"
            "   - **4. Architectural Decisions (ADRs)**: Key trade-offs (e.g. pgvector, local SLM inference)\n"
            "   - **5. Resilience & Operational View**: Error handling, fallbacks, and recovery\n"
            "3. Ensure the document is complete, highly professional, and concise."
        )
        task_instruction = (
            f"TASK:\n{query}\n\n"
            "Generate a complete, publication-ready Confluence page following the Confluence guidelines above. "
            "Do NOT output raw Python source code blocks."
        )
    elif prompt_type == "exceptions" or "exception" in q_lower or "error" in q_lower:
        system_msg = (
            "You are an SRE / Operations Assistant. Build a complete report of all errors, try/except blocks, "
            "and exceptions handled or raised in the code based on the provided context."
        )
        task_instruction = (
            f"TASK:\n{query}\n\n"
            "Please list all exceptions, errors, and handlers found in the code above with file names and line numbers."
        )
    elif prompt_type == "custom" and custom_instructions.strip():
        system_msg = f"{custom_instructions.strip()}"
        task_instruction = f"TASK:\n{query}"
    else:
        system_msg = (
            "You are a helpful software engineer and technical assistant. "
            "Analyze the provided code snippets and answer the user question accurately and thoroughly based on the code."
        )
        task_instruction = f"QUESTION:\n{query}"

    user_msg = f"""Here is the source code extracted from the repository:

{formatted_context}

{task_instruction}"""

    return system_msg, user_msg


def generate_llm_answer(
    system_msg: str,
    user_msg: str,
    model_name: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.1,
) -> str:
    """
    Invoke local Ollama LLM to synthesize the response using chat roles.
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            options={"temperature": temperature, "num_predict": 1500},
        )
        return response["message"]["content"]
    except Exception as e:
        return (
            f"❌ LLM Generation Error: {e}\n\n"
            f"Make sure Ollama is running and `{model_name}` is downloaded (`ollama pull {model_name}`)."
        )



# ── Public Pipeline Functions ─────────────────────────────────────────────────

def retrieve_from_postgres(
    query: str,
    db_config: Optional[Dict[str, Any]] = None,
    table_name: str = DEFAULT_TABLE,
    top_k: int = 5,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    llm_model: str = DEFAULT_LLM_MODEL,
    prompt_type: str = "standard",
    custom_instructions: str = "",
    file_filter: Optional[str] = None,
    project_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main retrieval pipeline connecting directly to `ingestionpipeline.py`'s Postgres table.

    Args:
        query:               User prompt, search topic, or question.
        db_config:           PostgreSQL config dict (defaults to INGESTION_DB_CONFIG).
        table_name:          Table to search (default: "documind_nprd.chunk_details").
        top_k:               Number of chunks to retrieve (default: 5).
        embedding_model:     Embedding model name (default: "mxbai-embed-large").
        llm_model:           Ollama LLM model name (default: "qwen2.5:0.5b").
        prompt_type:         "standard" | "exceptions" | "custom" | "qa".
        custom_instructions: Instructions when prompt_type="custom".
        file_filter:         Optional single file filter.
        project_filter:      Optional project folder prefix stored in file_name.

    Returns:
        Dict with keys:
          - 'answer': Generated markdown answer
          - 'sources': List of retrieved chunk dictionaries
          - 'query': User query
          - 'num_sources': Number of chunks retrieved
          - 'table': Table queried
    """
    # 1. Connect to PostgreSQL
    conn = get_postgres_connection(db_config)

    try:
        # 2. Embed user query using mxbai-embed-large
        query_vector = generate_query_embedding(query, model_name=embedding_model)

        # 3. Retrieve top-k similar chunks
        chunks = search_similar_chunks(
            conn=conn,
            query_embedding=query_vector,
            table_name=table_name,
            top_k=top_k,
            file_filter=file_filter,
            project_filter=project_filter,
        )

        if not chunks:
            return {
                "answer": f"⚠️ No matching chunks found in table `{table_name}` for the given query.",
                "sources": [],
                "query": query,
                "num_sources": 0,
                "table": table_name,
            }

        # 4. Build prompt and run local LLM
        system_msg, user_msg = build_anti_hallucination_prompt(
            query=query,
            chunks=chunks,
            prompt_type=prompt_type,
            custom_instructions=custom_instructions,
        )

        answer = generate_llm_answer(
            system_msg=system_msg,
            user_msg=user_msg,
            model_name=llm_model,
        )


        return {
            "answer": answer,
            "sources": chunks,
            "query": query,
            "num_sources": len(chunks),
            "table": table_name,
        }

    finally:
        conn.close()


def run_postgres(
    query: str,
    top_k: int = 5,
    llm_model: str = DEFAULT_LLM_MODEL,
    prompt_type: str = "standard",
    custom_instructions: str = "",
) -> str:
    """
    Simplified helper: connects to ingestion pipeline's Postgres table and returns answer string.
    """
    res = retrieve_from_postgres(
        query=query,
        top_k=top_k,
        llm_model=llm_model,
        prompt_type=prompt_type,
        custom_instructions=custom_instructions,
    )
    return res["answer"]


# ── Gradio Web UI ─────────────────────────────────────────────────────────────

def build_retrieval_ui():
    """Build and return the Gradio UI for the retrieval pipeline."""
    import gradio as gr

    def on_retrieve(
        query,
        prompt_type,
        custom_instructions,
        top_k,
        llm_model,
        db_host,
        db_name,
        db_user,
        db_pass,
        table_name,
    ):
        if not query.strip():
            return "⚠️ Please enter a query or question to retrieve.", ""

        custom_cfg = {
            "host": db_host.strip() or INGESTION_DB_CONFIG["host"],
            "database": db_name.strip() or INGESTION_DB_CONFIG["database"],
            "user": db_user.strip() or INGESTION_DB_CONFIG["user"],
            "password": db_pass.strip() or INGESTION_DB_CONFIG["password"],
        }

        try:
            result = retrieve_from_postgres(
                query=query.strip(),
                db_config=custom_cfg,
                table_name=table_name.strip() or DEFAULT_TABLE,
                top_k=int(top_k),
                llm_model=llm_model,
                prompt_type=prompt_type,
                custom_instructions=custom_instructions or "",
            )
        except Exception as e:
            return f"❌ Retrieval Error: {e}", ""

        # Format sources into clean markdown
        sources_md = [f"### 📚 Retrieved Sources ({result['num_sources']} chunks from `{result['table']}`)\n"]
        for s in result["sources"]:
            sources_md.append(
                f"#### Chunk #{s['chunk_number']} — `{s['file_name']}`\n"
                f"* **Type**: `{s['chunk_type']}` | **Name**: `{s['chunk_name']}` | **Lines**: {s['chunk_start_line']}–{s['chunk_end_line']} | **Similarity**: `{s['similarity']}`\n"
                f"```python\n{s['chunk_content']}\n```\n"
            )

        return result["answer"], "\n".join(sources_md)

    with gr.Blocks(
        title="DocuMind – PostgreSQL Retrieval",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    ) as demo:
        gr.Markdown(
            """
# 🔍 DocuMind &nbsp;·&nbsp; PostgreSQL Retrieval Pipeline
Direct retrieval from `documind_nprd.chunk_details` populated by `ingestionpipeline.py`.
**Zero Ingestion &bull; 100% Local via Ollama**
"""
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=350):
                gr.Markdown("### 📝 1. Query & Prompt")
                query_input = gr.Textbox(
                    label="Search Query / Question",
                    placeholder="e.g. Explain how Tree-sitter chunking is implemented",
                    lines=3,
                )
                prompt_type = gr.Radio(
                    choices=["standard", "exceptions", "qa", "custom"],
                    value="standard",
                    label="Output Mode",
                    info="standard → Confluence doc · exceptions → error report · qa → direct answer · custom → your prompt",
                )
                custom_p = gr.Textbox(
                    label="Custom Prompt Instructions",
                    placeholder="e.g. Provide a bulleted list of all functions with return types…",
                    lines=3,
                    visible=False,
                )

                gr.Markdown("---")
                gr.Markdown("### ⚙️ 2. Retrieval & Model")
                top_k = gr.Slider(
                    minimum=1,
                    maximum=15,
                    value=5,
                    step=1,
                    label="Top-K Chunks to Retrieve",
                )
                llm_model = gr.Dropdown(
                    choices=[
                        "qwen2.5:0.5b",
                        "qwen2.5-coder:3b",
                        "qwen2.5-coder:7b",
                        "phi3:mini",
                        "mistral:7b",
                    ],
                    value="qwen2.5:0.5b",
                    label="Local LLM (Ollama)",
                )

                with gr.Accordion("🛠️ Database Settings (from ingestionpipeline.py)", open=False):
                    db_host = gr.Textbox(label="Host", value=INGESTION_DB_CONFIG.get("host", "localhost"))
                    db_name = gr.Textbox(label="Database", value=INGESTION_DB_CONFIG.get("database", "documind_dev"))
                    db_user = gr.Textbox(label="User", value=INGESTION_DB_CONFIG.get("user", "vadmin"))
                    db_pass = gr.Textbox(label="Password", value=INGESTION_DB_CONFIG.get("password", "documind@123"), type="password")
                    table_name = gr.Textbox(label="Table", value=DEFAULT_TABLE)

                run_btn = gr.Button("🚀 Retrieve & Generate", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 📄 LLM Answer")
                answer_out = gr.Markdown(
                    value="_Output will appear here after clicking **Retrieve & Generate**._"
                )
                with gr.Accordion("📚 Retrieved Code Chunks & Sources", open=True):
                    sources_out = gr.Markdown(
                        value="_Retrieved chunks with file locations and similarity scores will appear here._"
                    )

        prompt_type.change(
            fn=lambda p: gr.update(visible=(p == "custom")),
            inputs=prompt_type,
            outputs=custom_p,
        )

        run_btn.click(
            fn=on_retrieve,
            inputs=[
                query_input,
                prompt_type,
                custom_p,
                top_k,
                llm_model,
                db_host,
                db_name,
                db_user,
                db_pass,
                table_name,
            ],
            outputs=[answer_out, sources_out],
            api_name=False,
        )

    return demo


# ── Entry Point (UI or CLI) ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # If launched with a query argument, run CLI mode
    if len(sys.argv) > 1 and sys.argv[1] not in ("--ui", "--server"):
        parser = argparse.ArgumentParser(
            description="Retrieve from ingestionpipeline's PostgreSQL pgvector table without re-ingesting."
        )
        parser.add_argument(
            "query",
            type=str,
            help="Query or question to ask the vector database",
        )
        parser.add_argument("--k", type=int, default=5, help="Number of chunks")
        parser.add_argument("--table", type=str, default=DEFAULT_TABLE, help="Target table")
        parser.add_argument("--embed", type=str, default=DEFAULT_EMBED_MODEL, help="Embedding model")
        parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL, help="Ollama LLM model")
        parser.add_argument("--prompt-type", type=str, default="standard", choices=["standard", "exceptions", "custom", "qa"])

        args = parser.parse_args()

        print(f"🔍 Searching table '{args.table}' using query: '{args.query}'…")
        try:
            result = retrieve_from_postgres(
                query=args.query,
                table_name=args.table,
                top_k=args.k,
                embedding_model=args.embed,
                llm_model=args.model,
                prompt_type=args.prompt_type,
            )
            print("\n" + "=" * 60)
            print("📄 ANSWER:")
            print("=" * 60)
            print(result["answer"])

            print("\n" + "=" * 60)
            print(f"📚 RETRIEVED SOURCES ({result['num_sources']} chunks):")
            print("=" * 60)
            for s in result["sources"]:
                print(f"\n• [{s['file_name']}] {s['chunk_type']} '{s['chunk_name']}' (lines {s['chunk_start_line']}-{s['chunk_end_line']}) | Similarity: {s['similarity']}")
                print(f"  {s['preview']}")
        except Exception as err:
            print(f"\n❌ Error: {err}")

    else:
        # Launch Gradio Web UI on port 7862
        print("🚀 Starting Retrieval Web Dashboard on port 7862…")
        ui = build_retrieval_ui()
        ui.launch(
            server_name="0.0.0.0",
            server_port=7862,
            share=True,
        )

