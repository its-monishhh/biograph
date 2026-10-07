"""Quick connectivity check. Usage: python verify_neo4j.py  (reads .env)"""
from backend.database import db

if __name__ == "__main__":
    db.connect()
    print("labels:", [r["label"] for r in db.query("CALL db.labels() YIELD label RETURN label")])
    print("node count:", db.query("MATCH (n) RETURN count(n) AS c")[0]["c"])
    db.close()
