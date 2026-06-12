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

    #normailize vectors
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    #having two norms of zero would cause a division by zero error, so we return 0.0 in that case.
    if a_norm == 0 or b_norm == 0:
        return 0.0
    
    #return float in [-1.0, 1.0]
    return float(np.dot(a, b) / (a_norm * b_norm))
   

# =============================================================================
# 2.  LSHIndex
# =============================================================================

class LSHIndex:

    def __init__(self, vectors: np.ndarray, num_tables: int, num_bits: int, **kwargs):

        # given parameters
        self._vectors    = vectors
        self._num_tables = num_tables
        self._num_bits   = num_bits
        n, d = vectors.shape

        # set seed from lsh.hyperplane_seed in settings.json, use defulat rng
        seed = _cfg["lsh"]["hyperplane_seed"]
        rng  = np.random.default_rng(seed)

        # set random hyperplanes as float 32
        self.planes = rng.standard_normal((num_tables, num_bits, d)).astype(np.float32)

        # build hash tables
        self.tables = self._build_tables()

    def _build_tables(self) -> list:
        
        #build hash tables and projecting vectors onto hyperplanes
        tables = []
        projections = self.planes @ self._vectors.T  
 
        # convert projections to bits, bit_i = 1  if  dot(vector, plane_i) >= 0  else  0
        bits = (projections >= 0).astype(np.uint32) 
 
        #calc power of 2 for each bit position
        powers = (2 ** np.arange(self._num_bits, dtype=np.uint32)) 
 
        #do the actual hashing and build tables
        for t in range(self._num_tables):
            # key   = sum(bit_i * 2**i  for i in range(num_bits))
            keys = bits[t].T @ powers 
 
            table: dict = {}
            for idx, key in enumerate(keys.tolist()):
                if key not in table:
                    table[key] = []
                table[key].append(idx)
 
            tables.append(table)
 
        return tables

    def query(self, q_vec: np.ndarray, k: int) -> list:

        # 1. For each table, hash q_vec to get a bucket key.
        q_proj = self.planes @ q_vec            
        q_bits = (q_proj >= 0).astype(np.uint32)  
        powers = (2 ** np.arange(self._num_bits, dtype=np.uint32))  
        q_keys = q_bits @ powers                 
 
        # 2. Collect all candidate indices from those buckets.
        candidates: set = set()
        for t, key in enumerate(q_keys.tolist()):
            bucket = self.tables[t].get(key, [])
            candidates.update(bucket)
 
        if not candidates:
            return []
 
        #   3. Score each unique candidate using cosine_similarity (or dot product if vectors are already unit-normalised).
        cand_list = list(candidates)
        cand_vecs = self._vectors[cand_list]    
        scores    = cand_vecs @ q_vec      

        # 4. Return the top-k results by score as a list of (score, idx) tuples, sorted by score descending.
        if len(cand_list) <= k:
            order  = np.argsort(scores)[::-1]
            return [(float(scores[i]), cand_list[i]) for i in order]
 
        # use argpartition for efficient top-k retrieval
        top_k_pos = np.argpartition(scores, -k)[-k:]
        top_k_pos = top_k_pos[np.argsort(scores[top_k_pos])[::-1]]
        return [(float(scores[i]), cand_list[i]) for i in top_k_pos]
 

# =============================================================================
# 3.  search  --  the main public function
# =============================================================================

def search(query: str, index: LSHIndex, encoder, movies: list,
           k: int = 5) -> list:
    
    # Step 1: encode the query string into a vector using the encoder.
    #         Make sure the vector is normalized and in float32 format.
    #      q_vec = encoder.encode(
    #     [query],
    #     normalize_embeddings=True,
    #     convert_to_numpy=True,)[0].astype(np.float32)
    q_vec = encoder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)
 
    # Step 2: search the index for the top-k nearest neighbours.
    candidates = index.query(q_vec, k) 
 
    # Step 3: return a list of k movie dicts, each with a 'score' field added,
    #         sorted best-first.
    results = []
    for score, idx in candidates:
        movie = dict(movies[idx])   
        movie["score"] = score
        results.append(movie)
 
    # return best matches sorted by score descending
    results.sort(key=lambda m: m["score"], reverse=True)
 
    return results[:k]
 


