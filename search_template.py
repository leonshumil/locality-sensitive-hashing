"""
search_template.py  --  Starter template for the LSH Movie Search project
==========================================================================
Instructions:
  1. Copy this file and rename the copy to search.py
  2. Fill in every section marked TODO
  3. Do NOT submit this template -- only submit search.py

VPL-safe imports only (no sklearn, scipy, torch, tensorflow, faiss, pandas).
"""

import os
import json
import heapq
import warnings
import numpy as np

# Suppress console noise (tokenizer / HuggingFace warnings)
os.environ["TOKENIZERS_PARALLELISM"]          = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# Load shared config
with open("settings.json") as f:
    _cfg = json.load(f)


# =============================================================================
# 1.  cosine_similarity(a, b)
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Return the cosine similarity between vectors a and b.
    - Use numpy dot product and norms only (no sklearn / scipy).
    - Return 0.0 if either vector has zero norm.
    """
    # TODO: implement
    raise NotImplementedError


# =============================================================================
# 2.  LSHIndex
# =============================================================================

class LSHIndex:
    """
    Random-hyperplane LSH index for approximate nearest-neighbour search.

    After __init__ returns, these attributes MUST exist:
    self.planes  -- np.ndarray, shape (num_tables, num_bits, d)
                    where d is the vector dimension (384 in this project,
                    as the all-MiniLM-L6-v2 model always outputs 384-dimensional vectors)
    self.tables  -- list of num_tables dicts {key -> [idx, ...]}
                    Each dictionary represents one hash table, where:
                    - `key` is the integer bucket ID computed from the hyperplanes.
                    - `[idx, ...]` is a list of integers representing the movies inside 
                        that bucket (e.g., [0, 5, 12] means the 1st, 6th, and 13th 
                        movies from tmdb_data.json)."""

    def __init__(self, vectors: np.ndarray, num_tables: int, num_bits: int, **kwargs):
        """
        vectors     -- np.ndarray, shape (n, d), float32, unit-normalised
        num_tables  -- number of independent hash tables (L)
        num_bits    -- number of hyperplanes per table (K)
        **kwargs    -- accept and ignore any extra arguments without error
        """
        self._vectors    = vectors
        self._num_tables = num_tables
        self._num_bits   = num_bits
        n, d = vectors.shape

        # TODO: Initialize self.planes with random normal values.
#       Use numpy's default random number generator, seeded with the
#       'hyperplane_seed' value from settings file["hyperplane_seed"].
#       self.planes must have shape (num_tables, num_bits, d) --
#       num_bits hyperplanes per table, each of dimension d --
#       and must be cast to float32.
        raise NotImplementedError

        self.tables = self._build_tables()

    def _build_tables(self) -> list:
        
        """
        Build hash tables from self.planes and self._vectors.
        
        Returns:
            A list of dictionaries. Each dictionary maps an integer hash key 
            to a list of movie indices (as detailed in __init__).

        Hash key formula:
            bit_i = 1  if  dot(vector, plane_i) >= 0  else  0
            key   = sum(bit_i * 2**i  for i in range(num_bits))

        Important: This is called by __init__ AND by the automated grader after 
        injecting test hyperplanes. It must execute without any extra arguments.
        """
        # TODO: implement
        raise NotImplementedError

    def query(self, q_vec: np.ndarray, k: int) -> list:
        """
        Return the top-k approximate nearest neighbours for q_vec.

        Steps:
          1. For each table, hash q_vec to get a bucket key.
          2. Collect all candidate indices from those buckets.
          3. Score each unique candidate using cosine_similarity (or dot product
             if vectors are already unit-normalised).
          4. Return the top-k results by score as a list of (score, idx) tuples, sorted by score descending.

        """
        # TODO: implement
        raise NotImplementedError


# =============================================================================
# 3.  search  --  the main public function
# =============================================================================

def search(query: str, index: LSHIndex, encoder, movies: list,
           k: int = 5) -> list:
    """
    Encode `query` and return the top-k most relevant movies.
    [Must encode the query to a vector internally using the encoder]
    Parameters:
        query   -- the search string
        index   -- a built LSHIndex over the movie vectors
        encoder -- a loaded SentenceTransformer model
        movies  -- list of movie dicts from tmdb_data.json
        k       -- number of results to return (default 5)

    Returns:
        list of k dicts, sorted best-first.
        Each dict must contain at least:
            "title" : str
            "score" : float  (cosine similarity)
        You may include other fields from tmdb_data.json.
    """
    # TODO:
    # Step 1: encode the query string into a vector using the encoder.
    #         Make sure the vector is normalized and in float32 format.
    #      q_vec = encoder.encode(
    #     [query],
    #     normalize_embeddings=True,
    #     convert_to_numpy=True,)[0].astype(np.float32)
    # Step 2: search the index for the top-k nearest neighbours.
    # Step 3: return a list of k movie dicts, each with a 'score' field added,
    #         sorted best-first.
    raise NotImplementedError


