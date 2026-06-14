"""
Hugging Face Spaces entry point — Gradio UI + /api/search REST endpoint.

HF Spaces serves the `app` variable (ASGI) directly:
    uvicorn app:app --host 0.0.0.0 --port 7860
"""
import json
import numpy as np
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from search import LSHIndex, search

# ── Startup: load data and build index ────────────────────────────────────────
with open("settings.json") as f:
    _cfg = json.load(f)

with open(_cfg["data"]["json_file"]) as f:
    _movies = json.load(f)

_vectors = np.load(_cfg["data"]["vectors_file"])
_encoder = SentenceTransformer(_cfg["model"]["name"])
_index   = LSHIndex(
    _vectors,
    num_tables=_cfg["lsh"]["num_tables"],
    num_bits=_cfg["lsh"]["num_bits"],
)
print(f"Ready — {len(_movies)} movies indexed.")

# ── Core search ───────────────────────────────────────────────────────────────
def _run_search(query: str, k: int) -> list:
    return search(query.strip(), _index, _encoder, _movies, max(1, min(int(k), 20)))

# ── Gradio UI ─────────────────────────────────────────────────────────────────
def gradio_search(query: str, k: int):
    if not query.strip():
        return []
    results = _run_search(query, k)
    return [
        [i + 1, m["title"], m["year"], m["genres"], f"{m['score']:.3f}"]
        for i, m in enumerate(results)
    ]

demo = gr.Interface(
    fn=gradio_search,
    inputs=[
        gr.Textbox(
            label="Query",
            placeholder='e.g. "sci-fi adventure set in outer space"',
            lines=1,
        ),
        gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Results (k)"),
    ],
    outputs=gr.Dataframe(
        headers=["Rank", "Title", "Year", "Genres", "Score"],
        datatype=["number", "str", "str", "str", "str"],
        label="Top matches",
        wrap=True,
    ),
    title="CineMatch — Semantic Movie Search",
    description=(
        "Neural embeddings + Locality-Sensitive Hashing over 4,803 TMDB films. "
        "Describe a movie in plain language — find the closest matches by meaning, not keywords."
    ),
    allow_flagging="never",
    theme=gr.themes.Soft(),
)

# ── FastAPI app ────────────────────────────────────────────────────────────────
fastapi_app = FastAPI(title="CineMatch API")

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://leonshumil.github.io",
        "http://localhost:8080",
        "http://localhost:5000",
        "http://127.0.0.1:8080",
    ],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

@fastapi_app.get("/api/search")
def api_search(query: str = "", k: int = 5):
    if not query.strip():
        return JSONResponse({"error": "query is required"}, status_code=400)
    results = _run_search(query, k)
    return {"results": results}

@fastapi_app.get("/api/health")
def api_health():
    return {"status": "ok", "movies": len(_movies)}

# Mount Gradio at / — HF Spaces serves the `app` variable as an ASGI app
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
