# Deploying CineMatch to Hugging Face Spaces

- **Space URL:** https://huggingface.co/spaces/LeonS98/CineMatch
- **Live API endpoint:** https://leons98-cinematch.hf.space/api/search
- **Gradio UI:** https://leons98-cinematch.hf.space/

---

## One-time setup

### 1. Clone the HF Space repo

```bash
git clone https://huggingface.co/spaces/LeonS98/CineMatch hf-cinematch
cd hf-cinematch
```

### 2. Create the Space README (only needed once)

Create `README.md` inside the cloned Space repo with this header
(keep it separate from the GitHub project README):

```markdown
---
title: CineMatch
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

Semantic movie search using LSH + sentence embeddings.
```

### 3. Copy required files from the GitHub project

Run this from the root of the GitHub project:

```bash
PROJECT=/path/to/locality-sensitive-hashing   # adjust
SPACE=/path/to/hf-cinematch                   # adjust

cp $PROJECT/app.py              $SPACE/
cp $PROJECT/search.py           $SPACE/
cp $PROJECT/settings.json       $SPACE/
cp $PROJECT/tmdb_data.json      $SPACE/
cp $PROJECT/tmdb_vectors.npy    $SPACE/
cp $PROJECT/requirements_hf.txt $SPACE/requirements.txt   # rename here
```

> **Note:** `requirements_hf.txt` must be renamed to `requirements.txt`
> in the Space repo. The GitHub project's `requirements.txt` is not used.

---

## Deploy / update

```bash
cd hf-cinematch
git add app.py search.py settings.json tmdb_data.json tmdb_vectors.npy requirements.txt README.md
git commit -m "deploy: update CineMatch backend"
git push
```

Hugging Face will rebuild the Space automatically. First cold start
(model download + index build) takes about 2–3 minutes.

---

## Verify

```bash
# Health check
curl https://leons98-cinematch.hf.space/api/health

# Live search
curl "https://leons98-cinematch.hf.space/api/search?query=sci-fi+space&k=3"
```

Expected response shape:
```json
{
  "results": [
    { "movie_id": 49047, "title": "Gravity", "year": "2013",
      "genres": "Science Fiction Thriller Drama", "score": 0.887, ... },
    ...
  ]
}
```

---

## Files NOT pushed to HF Spaces

| File | Reason |
|---|---|
| `index.html`, `style.css`, `app.js` | Served by GitHub Pages, not needed on HF |
| `config.js` | Contains TMDB key — never commit |
| `demo_results.json` | Static fallback for GitHub Pages only |
| `server.py` | Local Flask server, replaced by `app.py` on HF |
| `test_local.py`, `search_template.py` | Dev tooling only |
