import os
import warnings

# Minimal import-time suppression (do not import heavy external libs here)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings('ignore')

"""search.py

This file contains stub definitions (no implementations) required by the
Final-Project-DS-26 assignment. Implementations intentionally omitted.
"""

def cosine_similarity(a, b):
    """Compute cosine similarity between two 1-D numpy arrays.

    Not implemented: placeholder for the real function.
    """
    raise NotImplementedError


class LSHIndex:
    """Locality-Sensitive Hashing index stub.

    Required attributes and methods (to be implemented):
    - self.planes (np.ndarray)
    - self.tables (list of dicts)
    - _build_tables(self)
    - query(self, q_vec, k)
    """

    def __init__(self, vectors, num_tables, num_bits, **kwargs):
        """Initialize the LSHIndex. Placeholder only."""
        raise NotImplementedError

    def _build_tables(self):
        """Build hash tables from self._vectors and self.planes."""
        raise NotImplementedError

    def query(self, q_vec, k):
        """Return top-k candidates given an encoded query vector."""
        raise NotImplementedError


def search(query, index, encoder, movies, k=5):
    """Search function stub.

    - `query`: natural-language string
    - `index`: LSHIndex instance
    - `encoder`: sentence-transformers model
    - `movies`: list of movie dicts
    - `k`: number of results

    Must return a list of `k` movie dicts with at least `title` and `score`.
    """
    raise NotImplementedError
