"""Hugging Face Space entry point (Gradio SDK, free CPU tier).

Hugging Face runs `python app.py`; we serve the FastAPI backend on port 7860 and mount a
small Gradio status page at "/". All API routes (/api/..., /docs) are matched first.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("WEIGHTS_DIR", os.path.join(ROOT, "models", "weights"))
os.environ.setdefault("DB_PATH", "/tmp/app.db")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, os.path.join(ROOT, "backend"))  # so `app` resolves to backend/app, not this file

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402
from app.main import app as api  # noqa: E402

with gr.Blocks(title="Cyclone Decision Support API") as status_page:
    gr.Markdown(
        "# Cyclone Decision Support API\n"
        "Backend for the SIH 2026 PS 26070 prototype (research prototype, not an official IMD/NDMA product).\n\n"
        "- Health check: [`/api/health`](/api/health)\n"
        "- Interactive API docs: [`/docs`](/docs)\n\n"
        "The dashboard itself is hosted separately on Vercel."
    )

server = gr.mount_gradio_app(api, status_page, path="/")

if __name__ == "__main__":
    uvicorn.run(server, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
