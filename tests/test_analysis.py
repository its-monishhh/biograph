"""Logic tests for backend/services/analysis.py using a fake database.
Run from the repo root:  python -m tests.test_analysis
(These check the Python logic. Cypher syntax needs a real Neo4j.)"""
import backend.services.analysis as an


class FakeDB:
    def __init__(self, has_candidates=True):
        self.has_candidates = has_candidates

    def query(self, q, params=None):
        params = params or {}
        if "WHERE n.id = $q" in q:
            if "n:Compound" in q:
                return [{"id": "DB00331", "name": "Metformin", "rank": 0},
                        {"id": "DB00999", "name": "Metformin XR", "rank": 1}] if "metformin" in params["q"].lower() else []
            return [{"id": "DOID:1612", "name": "breast cancer", "rank": 0}] if "breast" in params["q"].lower() else []
        if "count(DISTINCT g) AS evidence" in q:
            if not self.has_candidates:
                return []
            if "RETURN d.id AS id" in q:
                return [{"id": "DOID:1612", "name": "breast cancer", "evidence": 9},
                        {"id": "DOID:162", "name": "cancer", "evidence": 4}]
            return [{"id": "DB00331", "name": "Metformin", "evidence": 9},
                    {"id": "DB00945", "name": "Aspirin", "evidence": 2}]
        if "type(r) AS rel" in q and "g.id AS id" in q:
            return [{"id": "5594", "name": "MAPK1", "rel": "BINDS"},
                    {"id": "207", "name": "AKT1", "rel": "UPREGULATES"}]
        if "-[:RESEMBLES]-(o:Compound)" in q:
            return [{"id": "DB00227", "name": "Phenformin"}]
        if "(o:Disease)-[:RESEMBLES]-" in q:
            return []
        if "pc.id AS pid" in q:
            return []
        if "startNode(r) = n" in q:
            return [{"rel": "BINDS", "mtype": "Gene", "mid": f"g{i}", "mname": f"GENE{i}", "outgoing": True} for i in range(9)] + \
                   [{"rel": "TREATS", "mtype": "Disease", "mid": "DOID:9352", "mname": "type 2 diabetes", "outgoing": True},
                    {"rel": "INCLUDES", "mtype": "PharmacologicClass", "mid": "N01", "mname": "Biguanides", "outgoing": False}]
        raise AssertionError("unexpected query: " + q[:80])


def check_graph(g):
    ids = {n["id"] for n in g["nodes"]}
    assert len(ids) == len(g["nodes"]), "duplicate node ids"
    for l in g["links"]:
        assert l["source"] in ids and l["target"] in ids, f"dangling link {l}"
    assert all(n["type"] in {"Compound", "Disease", "Gene", "PharmacologicClass", "Pathway", "Anatomy", "Symptom"} for n in g["nodes"])
    assert 3 <= len(g["nodes"]) <= 30


def run():
    # 1. drug entity, no trained GNN -> graph-evidence scoring
    an.db = FakeDB(); an._cache.clear(); an._gnn_is_trained = lambda: False
    r = an.analyze_entity("metformin", "drug")
    assert r["status"] == "success", r
    p = r["data"]["prediction"]
    assert p["source"] == "Metformin" and p["target"] == "breast cancer"
    assert p["scoreSource"] == "graph-evidence" and abs(p["neuralScore"] - 9 / 14) < 1e-3
    assert p["symbolicConfidence"] == 0.5          # genes + similar_drugs of 4 rule types
    assert p["noveltyScore"] == 0.8                # one analogue-based evidence type
    assert p["reasoningChain"] == [{"step": "MAPK1", "logic": "binds"}], p["reasoningChain"]
    assert "acts on 9 genes associated with breast cancer" in p["explanation"]
    assert "Phenformin resembles Metformin" in p["explanation"]
    check_graph(r["data"]["graph"])
    assert len(r["data"]["alternatives"]) == 1
    print("ok  drug entity (graph-evidence)")

    # 2. disease entity: orientation flips, target is a drug
    an._cache.clear()
    r = an.analyze_entity("breast", "disease")
    p = r["data"]["prediction"]
    assert p["source"] == "breast cancer" and p["target"] == "Metformin"
    assert p["reasoningChain"] == [{"step": "MAPK1", "logic": "is associated with"}], p["reasoningChain"]
    check_graph(r["data"]["graph"])
    print("ok  disease entity (orientation)")

    # 3. auto mode picks the drug first when both could match
    an._cache.clear()
    assert an.analyze_entity("metformin", "auto")["data"]["entity"]["kind"] == "drug"
    an._cache.clear()
    assert an.analyze_entity("breast", "auto")["data"]["entity"]["kind"] == "disease"
    print("ok  auto resolution")

    # 4. trained GNN is used when loaded
    an._cache.clear(); an._gnn_is_trained = lambda: True
    class G:  # prefers the second candidate
        def predict_batch(self, pairs): return [0.2, 0.9][:len(pairs)]
    an.gnn_predictor = G()
    p = an.analyze_entity("metformin", "drug")["data"]["prediction"]
    assert p["scoreSource"] == "gnn" and p["target"] == "cancer" and p["neuralScore"] == 0.9
    print("ok  trained GNN path")

    # 5. not found / no evidence / empty input
    an._cache.clear()
    assert an.analyze_entity("zzzz", "drug")["status"] == "not_found"
    an.db = FakeDB(has_candidates=False); an._cache.clear()
    assert an.analyze_entity("metformin", "drug")["status"] == "no_evidence"
    assert an.analyze_entity("  ", "drug")["status"] == "error"
    print("ok  error cases")
    print("\nall tests passed")


if __name__ == "__main__":
    run()
