"""
Post-deploy smoke test. Checks a live deployment end to end.

    python scripts/smoke_test.py https://YOUR-API.onrender.com https://YOUR-APP.vercel.app

The first request may take up to a minute while a free Render server wakes up.
Exit code is 0 when everything passes, 1 otherwise.
"""
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

TIMEOUT = 90
results = []


def call(url, method="GET", headers=None):
    req = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            body = r.read().decode("utf-8", "replace")
            return r.status, dict(r.headers), body
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode("utf-8", "replace")


def as_json(body):
    try:
        return json.loads(body)
    except ValueError:
        return {}


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    api = sys.argv[1].rstrip("/")
    web = sys.argv[2].rstrip("/") if len(sys.argv) > 2 else None

    code, _, body = call(f"{api}/health")
    check("backend is up (/health)", code == 200 and as_json(body).get("status") == "ok", f"HTTP {code}")

    code, _, body = call(f"{api}/test-db")
    j = as_json(body)
    nodes = j.get("node_count", 0)
    check("database reachable and loaded (/test-db)", j.get("status") == "success" and nodes > 0,
          f"{nodes} nodes" if nodes else j.get("message", f"HTTP {code}"))

    code, _, body = call(f"{api}/diseases")
    diseases = as_json(body).get("data") or []
    check("disease list loads (/diseases)", code == 200 and len(diseases) > 0, f"{len(diseases)} diseases")

    if diseases:
        did = diseases[0]["id"]
        code, _, body = call(f"{api}/predict?disease_id={urllib.parse.quote(did)}")
        preds = as_json(body).get("data") or []
        top = preds[0] if preds else {}
        check("predictions work (/predict)", code == 200 and len(preds) > 0 and "explanations" in top,
              f"{diseases[0]['name']}: top = {top.get('drug_name')} ({top.get('score_source')})")

    code, _, body = call(f"{api}/analyze?q=metformin&entity_type=drug")
    j = as_json(body)
    check("analysis works (/analyze)", code == 200 and j.get("status") == "success",
          j.get("message") or f"target = {j.get('data', {}).get('prediction', {}).get('target')}")

    code, _, body = call(f"{api}/analyze?q=zzzz-not-a-drug&entity_type=drug")
    check("unknown name is reported, not invented", as_json(body).get("status") == "not_found")

    code, _, _ = call(f"{api}/cache/clear", method="POST")
    check("admin endpoints are locked", code in (401, 403), f"HTTP {code}")

    if web:
        _, h, _ = call(f"{api}/health", headers={"Origin": web})
        allowed = {k.lower(): v for k, v in h.items()}.get("access-control-allow-origin")
        check("CORS allows your frontend", allowed == web, f"allow-origin = {allowed}")
        _, h, _ = call(f"{api}/health", headers={"Origin": "https://evil.example"})
        blocked = {k.lower(): v for k, v in h.items()}.get("access-control-allow-origin")
        check("CORS blocks other sites", blocked is None, f"allow-origin = {blocked}")
        code, _, body = call(web)
        check("frontend loads", code == 200 and 'id="root"' in body, f"HTTP {code}")

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
