"""Tests for the /predict pipeline (NeuroSymbolicService) using a fake database.
Run from the repo root:  python -m tests.test_predict"""
import backend.services.analysis as an
import backend.services.neurosymbolic as ns
from backend.services.symbolic import symbolic_reasoner


class FakeDB:
    def query(self, q, params=None):
        params = params or {}
        if "RETURN n.name AS name" in q:
            return [{"name": "breast cancer"}] if params.get("id") == "DOID:1612" else []
        if "count(DISTINCT g) AS evidence" in q:
            if params.get("id") != "DOID:1612":
                return []
            return [{"id": "DB1", "name": "DrugA", "evidence": 12},
                    {"id": "DB2", "name": "DrugB", "evidence": 3},
                    {"id": "DB3", "name": "DrugC", "evidence": 1}]
        if "type(r) AS rel" in q and "g.id AS id" in q:
            if params["cid"] == "DB1":
                return [{"id": "g1", "name": "MAPK1", "rel": "BINDS"},
                        {"id": "g2", "name": "AKT1", "rel": "UPREGULATES"}]
            if params["cid"] == "DB2":
                return [{"id": "g3", "name": "TP53", "rel": "BINDS"}]
            return []  # DB3 has no evidence for this pair
        if "-[:RESEMBLES]-(o:Compound)" in q:
            return [{"id": "DB9", "name": "DrugZ"}] if params["cid"] == "DB1" else []
        if "pc.id AS pid" in q:
            return [{"pid": "N1", "pname": "Kinase inhibitors", "id": "DB8", "name": "DrugY"}] if params["cid"] == "DB1" else []
        return []  # similar diseases


def fresh(trained: bool):
    an.db = FakeDB()
    an._gnn_is_trained = lambda: trained
    symbolic_reasoner.clear_cache()
    ns.service.clear_all_caches = lambda: None  # not needed here
    ns.service._result_cache.clear()


def check_path(path):
    # flat [node, REL, node, ...] with {"id","name"} nodes, as GraphView expects
    assert len(path) % 2 == 1
    for i, el in enumerate(path):
        if i % 2 == 0:
            assert isinstance(el, dict) and el["id"] and el["name"], el
        else:
            assert isinstance(el, str) and el.isupper(), el


def run():
    # 1. No trained model: graph-evidence scores, never the hash fallback
    fresh(trained=False)
    class Boom:
        def predict_batch(self, pairs): raise AssertionError("hash fallback must not be used")
    an.gnn_predictor = Boom()
    res = ns.service.predict_repurposing("DOID:1612", top_k=2)
    assert [r["drug_name"] for r in res] == ["DrugA", "DrugB"], res
    assert all(r["score_source"] == "graph-evidence" for r in res)
    assert abs(res[0]["score"] - 12 / 17) < 1e-6          # no symbolic boost in this mode
    assert res[0]["shared_genes"] == 12
    rules = {e["rule"] for e in res[0]["explanations"]}
    assert rules == {"Mechanism of Action", "Gene Regulation", "Similar Compound Strategy", "Pharmacologic Class"}, rules
    for e in res[0]["explanations"]:
        check_path(e["path"])
        assert e["path"][0]["name"] == "DrugA" and e["path"][-1]["name"] == "breast cancer"
    assert [e["rule"] for e in res[1]["explanations"]] == ["Mechanism of Action"]
    print("ok  graph-evidence mode (no placeholder scores)")

    # 2. Trained model: GNN scores plus the +0.2 boost when explained
    fresh(trained=True)
    class G:
        def predict_batch(self, pairs): return [0.5, 0.6, 0.95][:len(pairs)]
    an.gnn_predictor = G()
    res = ns.service.predict_repurposing("DOID:1612", top_k=3)
    by = {r["drug_name"]: r for r in res}
    assert all(r["score_source"] == "gnn" for r in res)
    assert abs(by["DrugA"]["score"] - 0.7) < 1e-9          # 0.5 + 0.2
    assert abs(by["DrugB"]["score"] - 0.8) < 1e-9          # 0.6 + 0.2
    assert abs(by["DrugC"]["score"] - 0.95) < 1e-9         # no explanation, no boost
    assert res[0]["drug_name"] == "DrugC" or res[0]["score"] >= res[-1]["score"]
    assert [r["score"] for r in res] == sorted((r["score"] for r in res), reverse=True)
    print("ok  trained GNN mode (+0.2 boost only when explained)")

    # 3. Unknown disease and empty results are not cached
    fresh(trained=False)
    assert ns.service.predict_repurposing("DOID:nope") == []
    assert ns.service._result_cache == {}
    print("ok  unknown disease returns [] and is not cached")
    print("\nall tests passed")


if __name__ == "__main__":
    run()
