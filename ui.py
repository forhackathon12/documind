"""Gradio dashboard layout and event wiring."""

import gradio as gr

from ingestion import generate_output, process_upload
from documents import generate_project_document, generate_exceptions_document
from retrieval_postgres import (
    DEFAULT_TABLE, INGESTION_DB_CONFIG, on_ask_code, retrieve_from_postgres,
)


DASHBOARD_CSS = """
        #model_radio .wrap {
            flex-direction: column !important;
            align-items: flex-start !important;
        }

        #model_radio .wrap label {
            width: 100% !important;
        }
        .gradio-container {
            --dm-accent: #176948;
            --dm-secondary: #146e82;
        }
        #documind-header {
            background: #eaf5ef;
            border-top: 4px solid var(--dm-accent);
            border-bottom: 1px solid #bfd8cb;
            padding: 20px 16px;
            color: #254638;
        }
        #documind-header h1 { color: #12523a; }
        .gradio-container a {
            color: var(--dm-secondary);
            text-decoration: underline;
            text-underline-offset: 3px;
            font-weight: 600;
        }
        .gradio-container a:hover { color: #12523a; }
        .gradio-container h3 {
            color: var(--dm-accent);
            border-bottom: 2px solid #c8ded3;
            padding-bottom: 8px;
        }
        .gradio-container button.primary {
            background: #176948;
            border-color: #176948;
            color: #fff;
        }
        .gradio-container button.primary:hover:not(:disabled) { background: #12523a; }
        .gradio-container .dm-action {
            background: var(--action-color) !important;
            border: 1px solid var(--action-color) !important;
            color: #fff !important;
            border-radius: 6px;
            font-weight: 600;
        }
        .gradio-container .dm-action:hover:not(:disabled) {
            filter: brightness(0.9);
        }
        .gradio-container .dm-action:disabled {
            background: var(--action-muted) !important;
            border-color: var(--action-muted) !important;
            color: var(--action-ink) !important;
            opacity: 0.65;
            cursor: not-allowed;
        }
        .dm-green { --action-color: #176948; --action-muted: #d4e9de; --action-ink: #28513c; }
        .dm-teal { --action-color: #146e82; --action-muted: #d8edf2; --action-ink: #25515c; }
        .dm-amber { --action-color: #925600; --action-muted: #f9e6bf; --action-ink: #684814; }
        .gradio-container input:focus, .gradio-container textarea:focus {
            outline: 2px solid #2b8a79;
            outline-offset: 2px;
        }
        #model_radio { border-top: 3px solid #146e82; }
        #documind-header p, #documind-header a { color: #254638; }
        .gradio-container .dm-field {
            --field-accent: #176948;
            --field-tint: #edf7f1;
            --block-label-text-color: var(--field-accent);
            --block-background-fill: var(--field-tint);
            --block-border-color: #b8d7cb;
            --input-background-fill: #fff;
            --input-text-color: #243c32;
            background: var(--field-tint) !important;
            border-color: #b8d7cb !important;
            border-left: 3px solid var(--field-accent) !important;
            border-radius: 6px;
        }
        .gradio-container .dm-field-blue {
            --field-accent: #146e82;
            --field-tint: #edf5fa;
        }
        .gradio-container .dm-field [data-testid="block-info"],
        .gradio-container .dm-field [data-testid="block-label"] {
            color: var(--field-accent) !important;
            font-weight: 600;
        }
        .gradio-container .dm-field textarea,
        .gradio-container .dm-field input:not([type="radio"]) {
            background: #fff !important;
            color: #243c32 !important;
            border-color: #b8d7cb !important;
        }
        .gradio-container .dm-field textarea:disabled {
            background: var(--field-tint) !important;
            opacity: 1;
            -webkit-text-fill-color: #243c32;
        }
        .gradio-container .dm-upload-area button {
            background: #edf5fa !important;
            color: #146e82 !important;
            border: 1px dashed #488c9c;
        }
        .gradio-container .dm-upload-area button:hover { background: #deedf5 !important; }
        #model_radio label {
            background: #fff;
            color: #245362;
            border-color: #c1d9e0;
        }
        #model_radio label:has(input:checked) {
            background: #dcefe7;
            border-color: #176948;
            color: #12523a;
        }
        #model_radio input { accent-color: #176948; }
        .gradio-container .dm-generation-status {
            min-height: 112px;
            box-sizing: border-box;
            padding: 16px;
            background: #edf7f1;
            border: 1px solid #b8d7cb;
            border-left: 4px solid #176948;
            border-radius: 6px;
            color: #254638;
            overflow-wrap: anywhere;
        }
        .gradio-container .dm-exceptions-status {
            background: #fff7e8;
            border-color: #e6cd9f;
            border-left-color: #925600;
            color: #684814;
        }
        .gradio-container .dm-generation-status .prose,
        .gradio-container .dm-generation-status p,
        .gradio-container .dm-generation-status strong { color: inherit; }
        .gradio-container .dm-generation-status p { margin: 8px 0; }
        """


def update_upload_mode(upload_mode):
    """Switch the upload control between file selection and folder selection."""
    return gr.update(
        label="Select Folder to Upload" if upload_mode == "Folder" else "Select Files to Upload",
        file_count="directory" if upload_mode == "Folder" else "multiple",
        value=None,
    )


def update_generate_button(
    model_name,
    project_folder,
    project_indexed
):
    """Enable actions only when their model, project, and indexing prerequisites exist."""
    if model_name and project_folder:
        return (
            gr.update(interactive=True),
            gr.update(interactive=bool(project_indexed)),
            gr.update(interactive=bool(project_indexed))
        )

    return (
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False)
    )


def reset_project_workflow():
    """Clear session results and disable later actions when the project changes."""
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
        gr.update(interactive=False),
        "",
        ""
    )


def reset_file_workflow():
    """Clear session results and disable later actions when input files change."""
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
        gr.update(interactive=False),
        "",
        ""
    )


def build_ui():
    """Build the main dashboard and connect its controls to pipeline callbacks."""
    with gr.Blocks(
        title="DocuMind"
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
<p style="text-align: center;">
Want to go to documents? <a href="./documents" target="_blank" rel="noopener noreferrer">Click here</a>.
</p>
""",
        elem_id="documind-header"
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
                scale=1,
                elem_classes=["dm-field"]
            )

        # --------------------------------------------------
        # File Upload Section
        # --------------------------------------------------
        with gr.Row():

            with gr.Column(scale=1):
                upload_mode = gr.Dropdown(
                    choices=["Files", "Folder"],
                    value="Folder",
                    label="Upload Type",
                    interactive=True,
                    elem_classes=["dm-field", "dm-field-blue"],
                )
                files = gr.File(
                    label="Select Folder to Upload",
                    file_count="directory",
                    type="filepath",
                    elem_classes=["dm-field", "dm-field-blue", "dm-upload-area"]
                )
                upload_mode.change(
                    fn=update_upload_mode,
                    inputs=[upload_mode],
                    outputs=[files],
                    api_name=False,
                )

            with gr.Column(scale=1):
                upload_btn = gr.Button(
                    "📤 Upload Files",
                    variant="secondary",
                    min_width=120,
                    size="sm",
                    interactive=True,
                    elem_classes=["dm-action", "dm-teal"]
                )

                upload_status = gr.Textbox(
                    label="Upload Status",
                    lines=8,
                    max_lines=12,
                    interactive=False,
                    elem_classes=["dm-field"]
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
                    elem_id="model_radio",
                    elem_classes=["dm-field", "dm-field-blue"]
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
                    interactive=False,
                    elem_classes=["dm-action", "dm-green"]
                )

                output_box = gr.Textbox(
                    label="Processing Summary",
                    lines=6,
                    max_lines=8,
                    interactive=False,
                    elem_classes=["dm-field", "dm-field-blue"]
                )

                gr.Markdown(
                    "### Generate Documents"
                )
                with gr.Row():
                    with gr.Column():
                        document_btn = gr.Button(
                            "Generate Document",
                            variant="secondary",
                            interactive=False,
                            elem_classes=["dm-action", "dm-green"]
                        )
                        document_status = gr.Markdown(
                            value="Your code documentation will appear here.",
                            elem_classes=["dm-generation-status"]
                        )
                    with gr.Column():
                        exceptions_btn = gr.Button(
                            "Generate Exceptions",
                            variant="secondary",
                            interactive=False,
                            elem_classes=["dm-action", "dm-amber"]
                        )
                        exceptions_status = gr.Markdown(
                            value="Your exception-handling report will appear here.",
                            elem_classes=["dm-generation-status", "dm-exceptions-status"]
                        )
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("---")
                gr.Markdown("### Ask DocuMind")
                # Query-model selection is independent of upload and indexing controls.
                ask_model_selector = gr.Dropdown(
                    choices=[
                        "Qwen2.5-Coder-7B-Instruct",
                        "Phi-3.5-min-instruct",
                        "Deepseek-Coder-6.7BInstruct"
                    ],
                    value="Qwen2.5-Coder-7B-Instruct",
                    label="Ask DocuMind Model",
                    interactive=True,
                    elem_classes=["dm-field", "dm-field-blue"],
                )
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Question or Prompt",
                        placeholder="Project: sample\nExplain how exceptions are handled.",
                        lines=3,
                        scale=4,
                        elem_classes=["dm-field", "dm-field-blue"],
                    )
                    ask_btn = gr.Button(
                        "Ask DocuMind",
                        variant="secondary",
                        scale=1,
                        interactive=True,
                        elem_classes=["dm-action", "dm-teal"]
                    )

                query_answer_box = gr.Textbox(
                    label="AI Answer",
                    lines=10,
                    interactive=False,
                    elem_classes=["dm-field"],
                )


        ask_btn.click(
            fn=on_ask_code,
            inputs=[query_input, ask_model_selector, project_folder_state],
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
                exceptions_btn,
                exceptions_status,
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
                exceptions_btn,
                exceptions_status,
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
                exceptions_btn
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
                exceptions_btn
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
            fn=lambda: "**Processing document...** Please wait.",
            inputs=[],
            outputs=[document_status],
            queue=False,
            show_progress="hidden",
            api_name=False
        ).then(
            fn=generate_project_document,
            inputs=[
                model_selector,
                project_folder_state
            ],
            outputs=[
                document_status
            ],
            show_progress="minimal",
            show_progress_on=[document_status],
            api_name=False
        )

        exceptions_btn.click(
            fn=lambda: "**Processing exception report...** Please wait.",
            inputs=[],
            outputs=[exceptions_status],
            queue=False,
            show_progress="hidden",
            api_name=False
        ).then(
            fn=generate_exceptions_document,
            inputs=[
                model_selector,
                project_folder_state
            ],
            outputs=[
                exceptions_status
            ],
            show_progress="minimal",
            show_progress_on=[exceptions_status],
            api_name=False
        )

    return demo


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
        """Run a retrieval-dashboard request and format its answer and source list."""
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
