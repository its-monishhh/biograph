import os
from typing import Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Copy .env.example to .env and fill it in."
        )
    return value


class Neo4jClient:
    def __init__(self):
        self.driver = None
        self.database = os.getenv("NEO4J_DATABASE") or None

    def connect(self):
        if not self.driver:
            uri = _require("NEO4J_URI")
            user = os.getenv("NEO4J_USERNAME", "neo4j")
            password = _require("NEO4J_PASSWORD")
            print(f"Connecting to Neo4j at {uri}...")
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
