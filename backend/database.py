import os
from typing import Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


def _clean(value):
    """Strip whitespace/newlines and wrapping quotes that often sneak in when
    pasting values into a hosting dashboard."""
    if value is None:
        return None
    value = value.strip().strip('"').strip("'").strip()
    return value or None


def _require(name: str) -> str:
    value = _clean(os.getenv(name))
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Copy .env.example to .env and fill it in."
        )
    return value


class Neo4jClient:
    def __init__(self):
        self.driver = None
        self.database = _clean(os.getenv("NEO4J_DATABASE"))

    def connect(self):
        if not self.driver:
            uri = _require("NEO4J_URI")
            user = _clean(os.getenv("NEO4J_USERNAME")) or "neo4j"
            password = _require("NEO4J_PASSWORD")
            print(f"Connecting to Neo4j at {uri} as user={user!r}, "
                  f"password_length={len(password)}, database={self.database!r}")
            self.driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                connection_timeout=30.0,
                max_connection_lifetime=200,  # Aura drops idle connections
            )
            self.driver.verify_connectivity()
            print("Connected to Neo4j.")

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None

    def query(self, query: str, parameters: Optional[dict] = None):
        if not self.driver:
            self.connect()
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]


# Singleton instance
db = Neo4jClient()
