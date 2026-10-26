"""
Load Hetionet into Neo4j in the schema this app expects.

Run from the repo root (reads credentials from .env):

    python -m backend.import_hetionet              # "lite": fits Aura Free (~165k relationships)
    python -m backend.import_hetionet --full       # + gene-gene interactions & disease up/down-regulation (~330k rels)
    python -m backend.import_hetionet --wipe       # delete everything in the DB first

Data: https://github.com/hetio/hetionet (CC0). Downloaded once into ./hetionet_data/.
"""
import argparse
import csv
import gzip
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# The nodes file is a normal git file; the edges file lives in Git LFS (media.* host).
# Try several sources per file and keep the first real download.
SOURCES = [
    "https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/",
    "https://media.githubusercontent.com/media/hetio/hetionet/main/hetnet/tsv/",
    "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/",
]
NODES_FILE = "hetionet-v1.0-nodes.tsv"
EDGES_FILE = "hetionet-v1.0-edges.sif.gz"
BATCH = 5000

# Hetionet "kind" -> Neo4j label. Only the kinds the app queries are loaded.
KIND_TO_LABEL = {
    "Compound": "Compound",
    "Disease": "Disease",
    "Gene": "Gene",
    "Anatomy": "Anatomy",
    "Pathway": "Pathway",
    "Pharmacologic Class": "PharmacologicClass",
    "Symptom": "Symptom",
}

# metaedge -> (relationship type, reverse direction?)
# DaG is flipped to Gene->Disease because the app's directed queries (and README) use that direction.
LITE_EDGES = {
    "CtD": ("TREATS", False),
    "CpD": ("PALLIATES", False),
    "CbG": ("BINDS", False),
    "CuG": ("UPREGULATES", False),
    "CdG": ("DOWNREGULATES", False),
    "DaG": ("ASSOCIATES", True),
    "CrC": ("RESEMBLES", False),
    "DrD": ("RESEMBLES", False),
    "DlA": ("LOCALIZES", False),
    "DpS": ("PRESENTS", False),
    "PCiC": ("INCLUDES", False),
    "GpPW": ("PARTICIPATES", False),
}
FULL_EXTRA_EDGES = {
    "GiG": ("INTERACTS", False),
    "DuG": ("UPREGULATES", False),
    "DdG": ("DOWNREGULATES", False),
}


def download(name: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 1_000:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for base in SOURCES:
        url = base + name
        print(f"Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:  # 404 / network error: try the next source
            errors.append(f"{url}: {e}")
            continue
        # Git-LFS pointer files are ~130 bytes; a real file is much larger
        if dest.stat().st_size > 1_000:
            return
        errors.append(f"{url}: got a Git-LFS pointer instead of the data")
        dest.unlink()
    sys.exit(f"Could not download {name}:\n  " + "\n  ".join(errors))


def strip_kind(prefixed: str) -> tuple[str, str]:
    kind, _, ident = prefixed.partition("::")
    return kind, ident


def load_nodes(path: Path) -> dict[str, list[dict]]:
    """label -> [{id, name}]"""
    by_label = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            label = KIND_TO_LABEL.get(row["kind"].strip())
            if not label:
                continue
            _, ident = strip_kind(row["id"].strip())
            by_label[label].append({"id": ident, "name": row["name"].strip()})
    return by_label


def load_edges(path: Path, wanted: dict, valid: dict[str, set]):
    """(srcLabel, relType, dstLabel) -> [{s, t}]  (already direction-corrected)"""
    groups = defaultdict(list)
    skipped = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: source metaedge target
        for row in reader:
            if len(row) != 3 or row[1] not in wanted:
                continue
            src_kind, src = strip_kind(row[0])
            dst_kind, dst = strip_kind(row[2])
            rel, reverse = wanted[row[1]]
            sl, dl = KIND_TO_LABEL.get(src_kind), KIND_TO_LABEL.get(dst_kind)
            if not sl or not dl or src not in valid[sl] or dst not in valid[dl]:
                skipped[row[1]] += 1
                continue
            if reverse:
                sl, dl, src, dst = dl, sl, dst, src
            groups[(sl, rel, dl)].append({"s": src, "t": dst})
    return groups, skipped


def batches(rows, n=BATCH):
    for i in range(0, len(rows), n):
        yield rows[i : i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also import GiG/DuG/DdG (needs ~330k relationship capacity)")
    ap.add_argument("--wipe", action="store_true", help="delete all existing nodes first")
    ap.add_argument("--data-dir", default="hetionet_data")
    args = ap.parse_args()

    data = Path(args.data_dir)
    download(NODES_FILE, data / NODES_FILE)
    download(EDGES_FILE, data / EDGES_FILE)

    wanted = dict(LITE_EDGES)
    if args.full:
        wanted.update(FULL_EXTRA_EDGES)

    nodes = load_nodes(data / NODES_FILE)
    valid = {label: {n["id"] for n in rows} for label, rows in nodes.items()}
    groups, skipped = load_edges(data / EDGES_FILE, wanted, valid)

    n_nodes = sum(len(v) for v in nodes.values())
    n_rels = sum(len(v) for v in groups.values())
    print(f"\nPlan: {n_nodes:,} nodes, {n_rels:,} relationships ({'full' if args.full else 'lite'})")
    for label, rows in sorted(nodes.items()):
        print(f"  node {label:<20}{len(rows):>8,}")
    for (sl, rel, dl), rows in sorted(groups.items()):
        print(f"  edge ({sl})-[{rel}]->({dl}){len(rows):>10,}")
    if skipped:
        print("  skipped (endpoint not loaded):", dict(skipped))
    if n_rels == 0:
        sys.exit("No relationships parsed - edges file is probably a Git-LFS pointer. Delete hetionet_data/ and retry.")

    from backend.database import db  # imported late so --help works without .env

    db.connect()
    with db.driver.session(database=db.database) as session:
        if args.wipe:
            print("\nWiping existing data ...")
            while session.run("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS c").single()["c"]:
                pass
        existing = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        if existing and not args.wipe:
            sys.exit(f"Database already has {existing:,} nodes. Re-run with --wipe to replace it.")

        for label in nodes:
            session.run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE").consume()

        t0 = time.time()
        for label, rows in nodes.items():
            for chunk in batches(rows):
                session.execute_write(
                    lambda tx, c=chunk, l=label: tx.run(
                        f"UNWIND $rows AS r CREATE (n:{l} {{id: r.id, name: r.name}})", rows=c
                    ).consume()
                )
            print(f"  loaded {len(rows):>7,} {label}")

        for (sl, rel, dl), rows in groups.items():
            for chunk in batches(rows):
                session.execute_write(
                    lambda tx, c=chunk, a=sl, r=rel, b=dl: tx.run(
                        f"UNWIND $rows AS r MATCH (a:{a} {{id: r.s}}) MATCH (b:{b} {{id: r.t}}) CREATE (a)-[:{r}]->(b)",
                        rows=c,
                    ).consume()
                )
            print(f"  loaded {len(rows):>7,} (:{sl})-[:{rel}]->(:{dl})")

        got_n = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        got_r = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    db.close()
    print(f"\nDone in {time.time() - t0:.0f}s. Database now has {got_n:,} nodes and {got_r:,} relationships.")
    if got_n != n_nodes or got_r != n_rels:
        print("WARNING: counts differ from the plan. If you hit your plan's limit, Aura silently rejects writes; use lite mode or a bigger instance.")


if __name__ == "__main__":
    main()
