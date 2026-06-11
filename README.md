**Project Overview**
- **Goal:** Build a simple semantic movie search using embeddings and Locality-Sensitive Hashing (LSH).
- **Approach:** Encode text queries to vectors, use LSH to narrow candidates, then re-rank with a precise similarity measure.

**High-Level Design**
- **Diagram:** Replace the placeholder image `LSH-architecture.png` with your diagram showing components and data flow.

  ![LSH Architecture](LSH-arcitecture.png)

- **Components (summary):**
  - **Encoder:** converts free-text queries into fixed-length vectors (embeddings).
  - **Vector Store:** the precomputed movie vectors (one row per movie).
  - **LSHIndex:** a set of independent hash tables built with random hyperplanes to bucket similar vectors.
  - **Search Flow:** encode query -> compute hash keys -> gather candidate indices from LSH tables -> re-rank candidates by cosine similarity -> return top-k results.

**Repository Layout**
- **`search.py`** : Main submission file (implement the required functions and classes here).
- **`search_template.py`** : Starter template / reference.
- **`settings.json`** : Configuration (LSH params, file paths, top-k).
- **`tmdb_data.json`** : Movies metadata used for final output.
- **`tmdb_vectors.npy`** : Numpy array of normalized embeddings (one vector per movie).
- **`test_local.py`** : Local checker used to validate the implementation.
- **`requirements.txt`** : Python dependencies for local testing.
- **`LSH-architecture.png`** : (placeholder) diagram showing the architecture — add your image here.

**Quick Start (edit as needed)**
- **Install deps:** `pip install -r requirements.txt`
- **Run local checks:** `python test_local.py`
- **Implement:** Fill `search.py` with the required stubs: `cosine_similarity`, `LSHIndex`, and `search`.

**Implementation Notes (keep brief)**
- Keep `search.py` import-safe (no side-effects at import time).
- Read LSH parameters from `settings.json` when needed.
- The search flow should be: encode -> LSH candidate lookup -> precise re-rank.

**Next Steps / To-Do**
- Replace the diagram `LSH-architecture.png` with your finalized image.
- Flesh out the prose in each section with exact wording you want for submission.
- Implement the functions in `search.py` and run `test_local.py` to validate behavior.

If you want, I can:
- implement the functions now, or
- add a placeholder `LSH-architecture.png` file, or
- expand any README section with more detail.
# locality-sensitive-hashing
