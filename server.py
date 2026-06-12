"""
Local Flask API for semantic movie search.

Install server extras (beyond requirements.txt):
    pip install flask flask-cors

Usage:
    python server.py          # runs on http://localhost:5000

Endpoints:
    POST /search  body: {"query": str, "k": int}  → {"query", "k", "results": [...]}
    GET  /search  params: ?query=...&k=5           → same
    GET  /health                                   → {"status": "ok", "movies": N}
"""
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from search import LSHIndex, search  # search.py loads settings.json at import

app = Flask(__name__)
CORS(app)

# ── Startup ──────────────────────────────────────────────────────────────────
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

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/search", methods=["GET", "POST"])
def search_endpoint():
    if request.method == "POST":
        body  = request.get_json(silent=True) or {}
        query = body.get("query", "").strip()
        k     = int(body.get("k", _cfg["search"]["top_k"]))
    else:
        query = request.args.get("query", "").strip()
        k     = int(request.args.get("k", _cfg["search"]["top_k"]))

    if not query:
        return jsonify({"error": "query parameter is required"}), 400

    k       = max(1, min(k, 20))
    results = search(query, _index, _encoder, _movies, k)
    return jsonify({"query": query, "k": k, "results": results})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "movies": len(_movies)})


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=False)
