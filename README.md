# CineMatch — Semantic Movie Search

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-6366f1?style=flat-square)](https://leonshumil.github.io/locality-sensitive-hashing/)
[![HF Space](https://img.shields.io/badge/HF%20Space-CineMatch-f59e0b?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/LeonS98/CineMatch)
[![BGU](https://img.shields.io/badge/BGU-Data%20Structures%202026-22c55e?style=flat-square)](https://in.bgu.ac.il/)

Search movies by meaning, not keywords. Type *"heartwarming story about a father and son"* — get *The Lion King*, even without those words in the database.

---

## Overview

CineMatch is a fully deployed semantic movie search engine over **4,803 TMDB films**. A custom LSH index built from scratch with NumPy narrows candidates in sub-linear time; cosine similarity re-ranks them exactly. The Python backend runs on Hugging Face Spaces; the frontend is a static GitHub Pages site.

Built as a Data Structures final project at **Ben-Gurion University (BGU), 2026**. The constraint: implement LSH from scratch — no sklearn, scipy, faiss, or other ANN libraries.

---

## Demo

| | URL |
|---|---|
| Frontend | https://leonshumil.github.io/locality-sensitive-hashing/ |
| Gradio UI | https://huggingface.co/spaces/LeonS98/CineMatch |
| REST endpoint | `https://leons98-cinematch.hf.space/api/search?query=...&k=5` |

The frontend falls back to precomputed demo results (`demo_results.json`) when the Space is unreachable. First request on a cold start may take ~30 s.

---

## How It Works

1. **Embed** — The query is encoded into a **384-dim vector** by `all-MiniLM-L6-v2` (sentence-transformers).
2. **Hash** — The vector is projected through **20 independent hash tables** (8 random hyperplanes each), producing binary bucket keys.
3. **Retrieve** — Movies sharing at least one bucket with the query become candidates — typically **~1 % of the index**.
4. **Rank** — Candidates are re-scored by exact cosine similarity; the top-*k* are returned with title, year, genres, overview, poster, and score.

---

## Architecture

```
Browser
  │
  ▼
GitHub Pages  (index.html · style.css · app.js)
  │
  │  GET /api/search?query=...&k=5
  ▼
Hugging Face Space  (app.py — FastAPI + Gradio)
  │
  ├─► all-MiniLM-L6-v2 ──► 384-dim query vector
  │
  ├─► LSHIndex (20 tables × 8 bits, numpy only)
  │       └─► bucket lookup → candidate set (~1% of 4,803 films)
  │
  └─► cosine_similarity reranking → top-k results
  │
  ▼
Browser renders movie cards
  │
  └─► TMDB API → movie posters
```

---

## Files

| File | Description |
|---|---|
| `search.py` | Core library: `cosine_similarity`, `LSHIndex`, `search()` |
| `app.py` | HF Spaces entry point — Gradio UI + FastAPI `/api/search` endpoint |
| `index.html` | GitHub Pages frontend |
| `style.css` | Dark-theme stylesheet |
| `app.js` | Frontend logic — calls HF Space API, falls back to demo results |
| `demo_results.json` | Precomputed fallback results (6 queries × 20 results each) |
| `settings.json` | LSH parameters and file paths |
| `tmdb_data.json` | Movie metadata (4,803 films) |
| `tmdb_vectors.npy` | Precomputed 384-dim embeddings |
| `config.example.js` | TMDB API key template — copy to `config.js` and fill in your key |
| `requirements.txt` | Python dependencies for local development |
| `requirements_hf.txt` | Dependencies for HF Spaces (rename to `requirements.txt` when deploying) |

---

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Validate the LSH implementation
python3 test_local.py

# Enable movie posters (requires a free TMDB API key)
cp config.example.js config.js   # paste your key inside

# Open index.html in a browser — runs in demo mode (no server needed)
```

Get a free TMDB key at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api).

### LSH parameters (`settings.json`)

```json
{
  "lsh": { "num_tables": 20, "num_bits": 8, "hyperplane_seed": 42 },
  "search": { "top_k": 5 }
}
```

Increasing `num_tables` improves recall; increasing `num_bits` reduces bucket collisions (sharper candidates).
