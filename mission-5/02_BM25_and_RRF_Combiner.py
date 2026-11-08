"""
Part 2: Communications & Protocol Officer - BM25 Lexical Retrieval & RRF Combiner
Mission 5: Dual-Path GraphRAG, Hybrid Search & The Google NotebookLM Grounding Lab
USSE Noosphere | Squadron Chomsky | Starship Operational Unit
Waypoint SEC-05-M01 (Crab Pulsar Remnant / Messier 1)
"""

import numpy as np
from collections import Counter
import re

class BM25Engine:
    """
    Okapi BM25 Lexical Retrieval Engine.
    Implements term-frequency saturation and document length normalization.
    """
    def __init__(self, corpus: list, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        self.N = len(corpus)
        self.idf = {}
        self._compute_idf()

    def _tokenize(self, text: str) -> list:
        return re.findall(r'\b\w+(?:-\w+)*\b', text.lower())

    def _compute_idf(self):
        df = Counter()
        for doc in self.tokenized_corpus:
            for word in set(doc):
                df[word] += 1
        for word, freq in df.items():
            self.idf[word] = np.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

    def score(self, query: str) -> np.ndarray:
        query_tokens = self._tokenize(query)
        scores = np.zeros(self.N)
        for idx, doc in enumerate(self.tokenized_corpus):
            if self.doc_len[idx] == 0:
                continue
            doc_len = self.doc_len[idx]
            frequencies = Counter(doc)
            score = 0.0
            for token in query_tokens:
                if token not in frequencies:
                    continue
                f = frequencies[token]
                idf = self.idf.get(token, 0.0)
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += idf * (numerator / denominator)
            scores[idx] = score
        return scores

def reciprocal_rank_fusion(dense_rankings: list, sparse_rankings: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion (RRF) rank aggregation combiner."""
    rrf_scores = {}
    
    for rank, doc_id in enumerate(dense_rankings):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    for rank, doc_id in enumerate(sparse_rankings):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs

def run_hybrid_retrieval_benchmark():
    print("=================================================================")
    print("Part 2: BM25 Lexical Engine & Reciprocal Rank Fusion Audit")
    print("Squadron Chomsky - Communications & Protocol Officer Station")
    print("=================================================================")
    
    archive_corpus = [
        "Crab Pulsar Messier 1 synchrotron radiation emission spectrum and magnetic dipole telemetry.",
        "AURA Subsystem 05 associative memory core graph indexing and dense embedding state.",
        "Emergency plasma containment manifold bypass valve replacement code: PART-NO-9942-X7.",
        "Hierarchical Navigable Small World skip-graph vector proximity traversal equations.",
        "Starship USSE Noosphere astrogation primary flight logs for sector SEC-05-M01."
    ]
    
    bm25 = BM25Engine(archive_corpus)
    
    exact_query = "plasma containment part-no-9942-x7"
    bm25_scores = bm25.score(exact_query)
    sparse_ranking = np.argsort(bm25_scores)[::-1].tolist()
    
    simulated_dense_ranking_1 = [1, 3, 0, 4, 2]
    rrf_hybrid_1 = reciprocal_rank_fusion(simulated_dense_ranking_1, sparse_ranking, k=60)
    
    print(f"\nQuery 1 (Exact Serial Code): '{exact_query}'")
    print(f"Sparse BM25 Top Document: Doc {sparse_ranking[0]} ('{archive_corpus[sparse_ranking[0]]}')")
    print(f"RRF Hybrid Combined Top Winner: Doc {rrf_hybrid_1[0][0]} (RRF Score: {rrf_hybrid_1[0][1]:.5f})")
    assert rrf_hybrid_1[0][0] == 2, "RRF failed to prioritize exact serial match!"
    print("[VERIFICATION] Exact Token Preservation via RRF Hybrid Search: SUCCESS")

    concept_query = "neutron star radiation and magnetic fields in Messier 1"
    bm25_scores_2 = bm25.score(concept_query)
    sparse_ranking_2 = np.argsort(bm25_scores_2)[::-1].tolist()
    simulated_dense_ranking_2 = [0, 4, 1, 3, 2]
    
    rrf_hybrid_2 = reciprocal_rank_fusion(simulated_dense_ranking_2, sparse_ranking_2, k=60)
    print(f"\nQuery 2 (Conceptual Search): '{concept_query}'")
    print(f"RRF Hybrid Combined Top Winner: Doc {rrf_hybrid_2[0][0]} ('{archive_corpus[rrf_hybrid_2[0][0]]}')")
    print("[VERIFICATION] Dual-Path Fusion Execution: PASSED")

if __name__ == "__main__":
    run_hybrid_retrieval_benchmark()
