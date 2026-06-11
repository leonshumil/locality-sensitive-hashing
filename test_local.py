"""
test_local.py  --  Local sanity-check before VPL submission
============================================================
Run this on your own machine BEFORE uploading to VPL.
It mirrors the three graded sections so you can catch crashes early.

Usage:
    python test_local.py          # tests search.py
    python test_local.py myfile   # tests myfile.py (without .py extension)

What this checks
----------------
  Section 1  cosine_similarity        (no data files needed)
  Section 2  LSHIndex structure       (no data files needed)
  Section 3  Full search recall       (requires data files -- see below)

Data files for Section 3
------------------------
  Section 3 needs the movie data files to run. Their paths are read from
  settings.json (the "data" section). Make sure the files exist at those
  paths relative to this script, or update settings.json to match your
  local folder layout.

  Default paths (from settings.json):
      tmdb_data.json        -- movie metadata
      tmdb_vectors.npy      -- pre-computed movie embeddings

  If either file is missing, Section 3 will print an error and exit.

This is NOT the real grader. Passing here does not guarantee a perfect
VPL score, but it catches the most common crashes before submission.
"""

import sys
import os
import importlib
import traceback
import numpy as np

# ---------------------------------------------------------------------------
# 0.  Import student module
# ---------------------------------------------------------------------------

module_name = sys.argv[1] if len(sys.argv) > 1 else "search"

print("=" * 62)
print("  Local VPL pre-check")
print("  Testing module: {}.py".format(module_name))
print("=" * 62)
print()

try:
    student = importlib.import_module(module_name)
except Exception as e:
    print("FATAL: could not import {}.py".format(module_name))
    print("  Error: {}".format(e))
    print()
    print("Fix the import error above before submitting to VPL.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Check for forbidden imports (best-effort scan of source file)
# ---------------------------------------------------------------------------
FORBIDDEN = ["sklearn", "scipy", "torch", "tensorflow", "keras", "faiss", "pandas"]
src_path = "{}.py".format(module_name)
if os.path.exists(src_path):
    src = open(src_path).read()
    found = [lib for lib in FORBIDDEN if ("import " + lib) in src or ("from " + lib) in src]
    if found:
        print("WARNING: forbidden library imports detected: {}".format(", ".join(found)))
        print("  These are NOT installed on VPL -- remove them before submitting.")
        print()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

passed = 0
failed = 0

def check(label, condition, hint=""):
    global passed, failed
    if condition:
        passed += 1
        print("  PASS  {}".format(label))
    else:
        failed += 1
        msg = "        hint: {}".format(hint) if hint else ""
        print("  FAIL  {}{}".format(label, "\n" + msg if msg else ""))

def section(title):
    print()
    print("-" * 62)
    print("  {}".format(title))
    print("-" * 62)

# ---------------------------------------------------------------------------
# Section 1 -- cosine_similarity
# ---------------------------------------------------------------------------
section("Section 1: cosine_similarity")

try:
    fn = student.cosine_similarity

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    check("identical unit vectors -> 1.0",
          abs(fn(a, b) - 1.0) < 1e-4,
          "cosine([1,0,0], [1,0,0]) should return 1.0")

    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    check("perpendicular vectors -> 0.0",
          abs(fn(a, b) - 0.0) < 1e-4,
          "cosine([1,0], [0,1]) should return 0.0")

    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    check("opposite vectors -> -1.0",
          abs(fn(a, b) - (-1.0)) < 1e-4,
          "cosine([1,0], [-1,0]) should return -1.0")

    z = np.array([0.0, 0.0])
    check("zero vector -> 0.0 (no crash)",
          fn(z, a) == 0.0 and fn(a, z) == 0.0,
          "cosine with a zero vector should return 0.0, not raise an error")

    # Non-unit vectors -- should still give correct result
    a = np.array([3.0, 0.0])
    b = np.array([0.0, 5.0])
    check("non-unit perpendicular -> 0.0",
          abs(fn(a, b) - 0.0) < 1e-4,
          "cosine([3,0], [0,5]) should return 0.0")

except AttributeError:
    print("  ERROR: cosine_similarity not found in {}.py".format(module_name))
    failed += 3
except Exception as e:
    print("  ERROR in cosine_similarity: {}".format(e))
    traceback.print_exc()
    failed += 3

# ---------------------------------------------------------------------------
# Section 2 -- LSHIndex structure (injected hyperplanes, no movie data)
# ---------------------------------------------------------------------------
section("Section 2: LSHIndex structure (no data files needed)")
print("  Setup: 4 synthetic 2-D vectors, 1 table, 2 bits")
print("  Inject axis-aligned planes: n0=[1,0], n1=[0,1]")
print("  Expected keys: Q1[+,+]->3  Q2[-,+]->2  Q3[-,-]->0  Q4[+,-]->1")
print()

try:
    sv = np.array([
        [ 1.0,  1.0],
        [-1.0,  1.0],
        [-1.0, -1.0],
        [ 1.0, -1.0],
    ], dtype=np.float32) / np.sqrt(2.0)

    ix = student.LSHIndex(sv, num_tables=1, num_bits=2)

    # Check required attributes exist
    check("self.planes attribute exists",
          hasattr(ix, "planes"),
          "LSHIndex must set self.planes in __init__")

    check("self.planes shape is (1, 2, 2)",
          hasattr(ix, "planes") and ix.planes.shape == (1, 2, 2),
          "self.planes should have shape (num_tables, num_bits, d) = (1, 2, 2)")

    check("self.tables attribute exists",
          hasattr(ix, "tables"),
          "LSHIndex must set self.tables in __init__")

    check("self.tables has 1 entry",
          hasattr(ix, "tables") and len(ix.tables) == 1,
          "self.tables should be a list of length num_tables=1")

    # Inject known planes and rebuild
    ix.planes = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    ix.tables = ix._build_tables()
    t = ix.tables[0]

    check("bucket Q1: v0=[+,+] -> key 3",
          0 in t.get(3, []),
          "v0 ([+,+]) should hash to key 3. Got table: {}".format(dict(sorted(t.items()))))

    check("bucket Q2: v1=[-,+] -> key 2",
          1 in t.get(2, []),
          "v1 ([-,+]) should hash to key 2. Got table: {}".format(dict(sorted(t.items()))))

    check("bucket Q3: v2=[-,-] -> key 0",
          2 in t.get(0, []),
          "v2 ([-,-]) should hash to key 0. Got table: {}".format(dict(sorted(t.items()))))

    check("bucket Q4: v3=[+,-] -> key 1",
          3 in t.get(1, []),
          "v3 ([+,-]) should hash to key 1. Got table: {}".format(dict(sorted(t.items()))))

    # query() tests
    q = sv[0].copy()
    res = ix.query(q, k=1)

    check("query(v0, k=1) returns at least 1 result",
          len(res) >= 1,
          "query() returned an empty list")

    check("query(v0, k=1): top result is v0 (idx=0)",
          len(res) >= 1 and res[0][1] == 0,
          "Expected idx=0, got: {}".format(res))

    check("query(v0, k=1): score is approx 1.0",
          len(res) >= 1 and abs(res[0][0] - 1.0) < 1e-4,
          "Expected score~1.0, got: {}".format(res[0][0] if res else "no results"))

    # Results should be sorted descending
    res_all = ix.query(q, k=4)
    scores = [s for s, _ in res_all]
    check("query results are sorted descending by score",
          scores == sorted(scores, reverse=True),
          "Scores should decrease: {}".format(scores))

except AttributeError as e:
    print("  ERROR: missing attribute or method -- {}".format(e))
    failed += 5
except Exception as e:
    print("  ERROR in LSHIndex: {}".format(e))
    traceback.print_exc()
    failed += 5

# ---------------------------------------------------------------------------
# Section 3 -- Full recall test (skipped if data files are missing)
# ---------------------------------------------------------------------------
section("Section 3: Full search recall (requires data files and model)")

import json

settings_ok  = os.path.exists("settings.json")
data_ok      = False
vectors_ok   = False

if settings_ok:
    with open("settings.json") as f:
        cfg = json.load(f)
    data_ok    = os.path.exists(cfg["data"]["json_file"])
    vectors_ok = os.path.exists(cfg["data"]["vectors_file"])

if not (settings_ok and data_ok and vectors_ok):
    missing = []
    if not settings_ok:    missing.append("settings.json")
    if not data_ok:        missing.append(cfg["data"]["json_file"] if settings_ok else "tmdb_data.json")
    if not vectors_ok:     missing.append(cfg["data"]["vectors_file"] if settings_ok else "tmdb_vectors.npy")
    print("  ERROR -- missing files: {}".format(", ".join(missing)))
    print("  Section 3 requires these files to run.")
    print("  Check that the paths in settings.json match your local folder layout:")
    if settings_ok:
        print("    data.json_file    = {}".format(cfg["data"]["json_file"]))
        print("    data.vectors_file = {}".format(cfg["data"]["vectors_file"]))
    print("  Sections 1 and 2 results above are still valid.")
else:
    try:
        print("  Loading vectors and model (this may take ~30 seconds)...")
        from sentence_transformers import SentenceTransformer

        vectors = np.load(cfg["data"]["vectors_file"]).astype(np.float32)
        with open(cfg["data"]["json_file"], encoding="utf-8") as f:
            movies = json.load(f)
        encoder = SentenceTransformer(cfg["model"]["name"])
        index   = student.LSHIndex(vectors,
                                   num_tables=cfg["lsh"]["num_tables"],
                                   num_bits=cfg["lsh"]["num_bits"])
        top_k   = cfg["search"]["top_k"]
        queries = cfg["comparison"]["default_queries"]

        def brute_top_k(q_vec, k):
            sims = vectors @ q_vec
            return set(np.argsort(sims)[-k:][::-1].tolist())

        print()
        if not hasattr(student, "search"):
            print("  ERROR: search() function not found -- Section 3 requires it.")
            for query in queries:
                check("search() '{}'".format(query[:38]), False,
                      "search() not implemented in {}.py".format(module_name))
        else:
            for query in queries:
                try:
                    # Ground truth
                    q_vec      = encoder.encode(
                        [query], normalize_embeddings=True, convert_to_numpy=True,
                    )[0].astype(np.float32)
                    true_top    = brute_top_k(q_vec, top_k)
                    true_titles = {movies[i]["title"] for i in true_top}

                    # Student full pipeline
                    results        = student.search(query, index, encoder, movies, k=top_k)
                    result_titles  = {r["title"] for r in results if "title" in r}
                    hits           = len(true_titles & result_titles)
                    recall         = hits / top_k
                    ok             = hits >= 2

                    label = "Recall@{} '{}'  ({}/{} exact matches, {:.0%})".format(
                        top_k, query[:38], hits, top_k, recall)
                    check(label, ok,
                          "search() returned fewer than 2 exact matches -- check encoding and LSH index")
                except Exception as e:
                    check("search() '{}'".format(query[:38]), False,
                          "search() raised: {}".format(e))

        # Format check on search() output
        if hasattr(student, "search"):
            print()
            print("  Format check: testing search() return structure...")
            try:
                results = student.search(queries[0], index, encoder, movies, k=top_k)
                check("search() returns a list",
                      isinstance(results, list),
                      "search() should return a list of dicts")
                check("search() returns k results",
                      len(results) == top_k,
                      "Expected {} results, got {}".format(top_k, len(results)))
                check("search() results have 'title' key",
                      len(results) > 0 and "title" in results[0],
                      "Each result dict must have a 'title' key")
                check("search() results have 'score' key",
                      len(results) > 0 and "score" in results[0],
                      "Each result dict must have a 'score' key")

                print()
                print("  Sample output for: \"{}\"".format(queries[0]))
                for r in results[:3]:
                    print("    [{:.4f}]  {}".format(r.get("score", 0), r.get("title", "?")))
            except Exception as e:
                print("  search() raised an error: {}".format(e))
                traceback.print_exc()

    except Exception as e:
        print("  ERROR during Section 3: {}".format(e))
        traceback.print_exc()
        failed += 5

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = passed + failed
print()
print("=" * 62)
print("  Results: {}/{} checks passed".format(passed, total))
if failed == 0:
    print("  All checks passed -- looks good for VPL submission!")
else:
    print("  {} check(s) failed -- fix before submitting.".format(failed))
print("=" * 62)
print()
print("Reminder: this is a pre-check, not the real grader.")
print("The real grader may have additional edge cases.")
