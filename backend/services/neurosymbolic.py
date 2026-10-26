from backend.models.gnn import gnn_predictor
from backend.services.symbolic import symbolic_reasoner
from backend.services.analysis import find_candidates, score_candidates, get_name
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

CANDIDATE_LIMIT = 200
SYMBOLIC_BOOST = 0.2

class NeuroSymbolicService:
    def __init__(self):
        self._result_cache: Dict[str, List[Dict]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_key(self, disease_id: str, top_k: int) -> str:
        """Create cache key for full prediction results"""
        return f"{disease_id.lower()}:{top_k}"

    def predict_repurposing(self, disease_id: str, top_k: int = 10, use_cache: bool = True) -> List[Dict]:
        """
        Rank drug candidates for a disease.

        1. Candidates: drugs linked to the disease through genes (drug -> gene -> disease),
           excluding known treatments, ordered by how many shared genes support them.
        2. Score: the trained GNN if a checkpoint is loaded, otherwise a transparent
           graph-evidence score n / (n + 5). Placeholder scores are never used here.
        3. Explain: pair-specific symbolic rules for the top candidates only.
        4. With a trained GNN, +0.2 (capped at 1.0) when an explanation path exists.
        """
        cache_key = self._get_cache_key(disease_id, top_k)
        if use_cache and cache_key in self._result_cache:
            self._cache_hits += 1
            return self._result_cache[cache_key]
        self._cache_misses += 1

        disease_name = get_name("Disease", disease_id)
        if disease_name is None:
            return []
        entity = {"id": disease_id, "name": disease_name, "kind": "disease"}

        candidates = find_candidates(entity, limit=CANDIDATE_LIMIT)
        if not candidates:
            return []
        source = score_candidates(entity, candidates)
        top = candidates[:top_k]

        # explanations are independent queries, so run them concurrently
        with ThreadPoolExecutor(max_workers=5) as pool:
            explanations = list(pool.map(
                lambda c: symbolic_reasoner.explain(c["id"], disease_id, c["name"], disease_name), top))

        results = []
        for cand, expl in zip(top, explanations):
            score = cand["score"]
            if source == "gnn" and expl:
                score = min(score + SYMBOLIC_BOOST, 1.0)
            results.append({
                "drug_id": cand["id"],
                "drug_name": cand["name"],
                "score": score,
                "explanations": expl,
                "score_source": source,
                "shared_genes": cand["evidence"],
            })
        results.sort(key=lambda r: r["score"], reverse=True)

        self._result_cache[cache_key] = results
        return results

    def clear_all_caches(self):
        """Clear all caches in the system"""
        self._result_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        gnn_predictor.clear_cache()
        symbolic_reasoner.clear_cache()

    def get_cache_stats(self) -> Dict:
        """Get comprehensive cache statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "service_cache": {
                "cache_size": len(self._result_cache),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate": f"{hit_rate:.2%}"
            },
            "gnn_cache": gnn_predictor.get_cache_stats(),
            "symbolic_cache": symbolic_reasoner.get_cache_stats()
        }

service = NeuroSymbolicService()
