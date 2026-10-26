"""Shared DocuMind runtime configuration."""

import os

BASE_UPLOAD_FOLDER = os.environ.get("BASE_UPLOAD_FOLDER", "/home/azureuser/proj-documind/test/repo")

try:
    os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)
except Exception:
    BASE_UPLOAD_FOLDER = os.path.expanduser("~/proj-documind/test/repo")
    os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)

DB_CONFIG = {
    "host": os.environ.get("PGHOST", "localhost"),
    "database": os.environ.get("PGDATABASE", "documind_dev"),
    "user": os.environ.get("PGUSER", "vadmin"),
    "password": os.environ.get("PGPASSWORD", "documind@123"),
}
