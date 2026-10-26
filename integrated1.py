"""Start DocuMind with: python integrated1.py."""

import os

from config import BASE_UPLOAD_FOLDER
from documents import document_root
from ui import DASHBOARD_CSS, build_ui


if __name__ == "__main__":
    from document_library import create_document_routes

    demo = build_ui()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        css=DASHBOARD_CSS,
        show_error=True,
        app_kwargs={"routes": create_document_routes(document_root())},
        allowed_paths=[
            os.path.join(
                BASE_UPLOAD_FOLDER,
                "_confluence_drafts"
            )
        ]
    )
