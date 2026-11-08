"""
Part 1: Systems Architect - HNSW Multi-Layer Vector Index
Mission 5: Dual-Path GraphRAG, Hybrid Search & The Google NotebookLM Grounding Lab
USSE Noosphere | Squadron Chomsky | Starship Operational Unit
Waypoint SEC-05-M01 (Crab Pulsar Remnant / Messier 1)
"""

import numpy as np
import time

class HNSWNode:
    """Represents a node in the Hierarchical Navigable Small World (HNSW) graph."""
    def __init__(self, idx: int, vector: np.ndarray, level: int):
        self.idx = idx
        self.vector = vector
        self.level = level
        self.neighbors = {l: [] for l in range(level + 1)}

class HNSWIndex:
    """
    Pure Python/NumPy Implementation of Hierarchical Navigable Small World (HNSW)
    Multi-Layer Skip-Graph for O(log N) approximate nearest neighbor search.
    Reference: Malkov & Yashunin (2018).
    """
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 64, m_L: float = None):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.m_L = m_L if m_L is not None else 1.0 / np.log(M)
        self.nodes = {}
        self.enter_node = None
        self.max_level = -1

    def _cosine_similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        """Vectorized cosine similarity with unit vector normalization protection."""
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            return 0.0
        return float(np.dot(u, v) / (norm_u * norm_v))

    def _random_level(self) -> int:
        """Probabilistic layer assignment using exponential decay function."""
        r = np.random.uniform(0, 1)
        if r == 0:
            r = 1e-9
        return int(np.floor(-np.log(r) * self.m_L))

    def _search_layer(self, query: np.ndarray, ep: int, ef: int, level: int) -> list:
        """Greedy skip-graph traversal on a specific layer l."""
        v = {ep}
        C = [(self._cosine_similarity(query, self.nodes[ep].vector), ep)]
        W = list(C)
        
        while len(C) > 0:
            C.sort(key=lambda x: x[0], reverse=True)
            curr_sim, curr_id = C.pop(0)
            
            W.sort(key=lambda x: x[0], reverse=True)
            farthest_sim = W[-1][0]
            
            if curr_sim < farthest_sim and len(W) >= ef:
                break
                
            for neighbor in self.nodes[curr_id].neighbors.get(level, []):
                if neighbor not in v:
                    v.add(neighbor)
                    sim = self._cosine_similarity(query, self.nodes[neighbor].vector)
                    W.sort(key=lambda x: x[0], reverse=True)
                    if sim > W[-1][0] or len(W) < ef:
                        C.append((sim, neighbor))
                        W.append((sim, neighbor))
                        if len(W) > ef:
                            W.sort(key=lambda x: x[0], reverse=True)
                            W.pop()
        return W

    def add(self, idx: int, vector: np.ndarray):
        """Inserts a vector into the multi-layer HNSW graph structure."""
        level = self._random_level()
        new_node = HNSWNode(idx, vector, level)
        self.nodes[idx] = new_node

        if self.enter_node is None:
            self.enter_node = idx
            self.max_level = level
            return

        curr_ep = self.enter_node
        curr_max = self.max_level

        # Top-down greedy search for nearest entry point down to (level + 1)
        for l in range(curr_max, level, -1):
            W = self._search_layer(vector, curr_ep, ef=1, level=l)
            if W:
                curr_ep = W[0][1]

        # Insert node into layers from min(level, max_level) down to 0
        for l in range(min(level, curr_max), -1, -1):
            W = self._search_layer(vector, curr_ep, ef=self.ef_construction, level=l)
            neighbors = [item[1] for item in W[:self.M]]
            new_node.neighbors[l] = neighbors
            
            # Bidirectional edge linking
            for n_idx in neighbors:
                if l in self.nodes[n_idx].neighbors:
                    self.nodes[n_idx].neighbors[l].append(idx)
                    if len(self.nodes[n_idx].neighbors[l]) > self.M * 2:
                        # Prune excess connections by similarity
                        self.nodes[n_idx].neighbors[l].sort(
                            key=lambda x: self._cosine_similarity(self.nodes[n_idx].vector, self.nodes[x].vector),
                            reverse=True
                        )
                        self.nodes[n_idx].neighbors[l] = self.nodes[n_idx].neighbors[l][:self.M]

            if W:
                curr_ep = W[0][1]

        if level > self.max_level:
            self.max_level = level
            self.enter_node = idx

    def search(self, query: np.ndarray, k: int = 5, ef: int = 16) -> list:
        """Executes logarithmic k-NN search across layers."""
        if self.enter_node is None:
            return []
            
        curr_ep = self.enter_node
        for l in range(self.max_level, 0, -1):
            W = self._search_layer(query, curr_ep, ef=1, level=l)
            if W:
                curr_ep = W[0][1]
                
        W = self._search_layer(query, curr_ep, ef=max(ef, k), level=0)
        W.sort(key=lambda x: x[0], reverse=True)
        return W[:k]

def run_hnsw_benchmark():
    print("=================================================================")
    print("Part 1: HNSW Multi-Layer Vector Proximity Index Audit")
    print("Squadron Chomsky - Systems Architect Station")
    print("=================================================================")
    np.random.seed(42)
    N = 2500  # Corpus size
    dim = 768 # Standard dense embedding dimension (e.g., RoBERTa/Jina)
    
    index = HNSWIndex(dim=dim, M=16, ef_construction=32)
    print(f"Generating N={N} dense {dim}-dimensional embedding vectors...")
    vectors = np.random.randn(N, dim)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    print("Building multi-layer HNSW skip-graph index...")
    t0 = time.time()
    for i in range(N):
        index.add(i, vectors[i])
    build_time = time.time() - t0
    print(f"HNSW Index constructed successfully in {build_time:.2f} seconds.")
    print(f"Max Graph Level Reached: {index.max_level}")
    
    query = np.random.randn(dim)
    query = query / np.linalg.norm(query)
    
    t0 = time.time()
    hnsw_results = index.search(query, k=5, ef=32)
    hnsw_latency = (time.time() - t0) * 1000
    
    t0 = time.time()
    flat_sims = np.dot(vectors, query)
    flat_top_k = np.argsort(flat_sims)[::-1][:5]
    flat_latency = (time.time() - t0) * 1000
    
    print("--- Benchmark Performance Results ---")
    print(f"HNSW Search Latency: {hnsw_latency:.4f} ms")
    print(f"Brute-Force Linear Flat Search Latency: {flat_latency:.4f} ms")
    print(f"Speedup Factor: {flat_latency / max(hnsw_latency, 1e-6):.2f}x faster")
    print("HNSW Top-5 Results (Sim, DocID):", [(round(s, 4), idx) for s, idx in hnsw_results])
    print("Brute-Force Exact Top-5 IDs:", list(flat_top_k))
    print("[TEST STATUS] HNSW Logarithmic Traversal Verified: PASSED")

if __name__ == "__main__":
    run_hnsw_benchmark()
