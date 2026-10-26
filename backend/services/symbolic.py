"""
Symbolic reasoner: turns graph evidence for a (drug, disease) pair into
explanations the UI can draw.

Every rule is specific to the pair. A rule only fires when a real path
connects THIS drug to THIS disease, so an explanation always means something.
The graph queries live in analysis.pair_evidence (one place, shared with /analyze).

Explanation format (consumed by the frontend GraphView):
    {"rule": str, "confidence": 1.0, "path": [node, REL, node, REL, node, ...]}
where each node is {"id", "name"} and each REL is a relationship name.
"""
from typing import Dict, List, Optional, Tuple

from backend.services.analysis import pair_evidence


class SymbolicReasoner:
    def __init__(self):
        # Documentation of the patterns (the queries themselves are in analysis.pair_evidence)
        self.rules = {
            "Mechanism of Action": "(Drug)-[:BINDS]->(Gene)-[:ASSOCIATES]->(Disease)",
            "Gene Regulation": "(Drug)-[:UPREGULATES|DOWNREGULATES]->(Gene)-[:ASSOCIATES]->(Disease)",
            "Similar Compound Strategy": "(Drug)-[:RESEMBLES]-(Drug2)-[:TREATS]->(Disease)",
            "Pharmacologic Class": "(Drug)<-[:INCLUDES]-(Class)-[:INCLUDES]->(Drug2)-[:TREATS]->(Disease)",
            "Similar Disease Strategy": "(Drug)-[:TREATS]->(Disease2)-[:RESEMBLES]-(Disease)",
        }
        self._explanation_cache: Dict[str, List[Dict]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @staticmethod
    def _get_cache_key(drug_id: str, disease_id: str) -> str:
        return f"{drug_id.lower()}:{disease_id.lower()}"

    def explain(self, drug_id: str, disease_id: str,
                drug_name: Optional[str] = None, disease_name: Optional[str] = None) -> List[Dict]:
        key = self._get_cache_key(drug_id, disease_id)
        if key in self._explanation_cache:
            self._cache_hits += 1
            return self._explanation_cache[key]
        self._cache_misses += 1
        explanations = self._compute_explanations(drug_id, disease_id, drug_name, disease_name)
        self._explanation_cache[key] = explanations
        return explanations

    def _compute_explanations(self, drug_id, disease_id, drug_name, disease_name) -> List[Dict]:
        ev = pair_evidence(drug_id, disease_id)
        drug = {"id": drug_id, "name": drug_name or drug_id}
        disease = {"id": disease_id, "name": disease_name or disease_id}

        def node(row, id_key="id", name_key="name"):
            return {"id": row[id_key], "name": row[name_key]}

        def item(rule, path, examples):
            return {"rule": rule, "confidence": 1.0, "path": path, "examples": examples}

        out: List[Dict] = []
        binds = [g for g in ev["genes"] if g["rel"] == "BINDS"]
        regs = [g for g in ev["genes"] if g["rel"] in ("UPREGULATES", "DOWNREGULATES")]
        if binds:
            g = binds[0]
            out.append(item("Mechanism of Action",
                            [drug, "BINDS", node(g), "ASSOCIATES", disease], len(binds)))
        if regs:
            g = regs[0]
            out.append(item("Gene Regulation",
                            [drug, g["rel"], node(g), "ASSOCIATES", disease], len(regs)))
        if ev["similar_drugs"]:
            o = ev["similar_drugs"][0]
            out.append(item("Similar Compound Strategy",
                            [drug, "RESEMBLES", node(o), "TREATS", disease], len(ev["similar_drugs"])))
        if ev["classes"]:
            c = ev["classes"][0]
            out.append(item("Pharmacologic Class",
                            [drug, "MEMBER_OF", {"id": c["pid"], "name": c["pname"]},
                             "INCLUDES", node(c), "TREATS", disease], len(ev["classes"])))
        if ev["similar_diseases"]:
            o = ev["similar_diseases"][0]
            out.append(item("Similar Disease Strategy",
                            [drug, "TREATS", node(o), "RESEMBLES", disease], len(ev["similar_diseases"])))
        return out

    def explain_batch(self, pairs: List[Tuple[str, str]]) -> List[List[Dict]]:
        return [self.explain(d, s) for d, s in pairs]

    def clear_cache(self):
        self._explanation_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self) -> Dict:
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "cache_size": len(self._explanation_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": f"{hit_rate:.2%}",
        }


symbolic_reasoner = SymbolicReasoner()
