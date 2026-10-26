"""Tests for admin protection and rate limiting. Run: python -m tests.test_security"""
import os
from fastapi import HTTPException
import backend.main as m


class Req:
    def __init__(self, ip="1.2.3.4", xff=""):
        self.headers = {"x-forwarded-for": xff} if xff else {}
        self.client = type("C", (), {"host": ip})()


def expect(exc, fn, *a):
    try:
        fn(*a)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


def run():
    # admin endpoints: disabled without ADMIN_TOKEN, locked with it
    os.environ.pop("ADMIN_TOKEN", None)
    assert expect(HTTPException, m.require_admin, "").status_code == 403
    os.environ["ADMIN_TOKEN"] = "s3cret"
    assert expect(HTTPException, m.require_admin, "wrong").status_code == 401
    assert expect(HTTPException, m.require_admin, "").status_code == 401
    assert m.require_admin("s3cret") is None
    print("ok  admin token (403 disabled, 401 wrong, pass when correct)")

    # rate limit: allows N, blocks N+1, separate per IP, window slides, 0 disables
    os.environ["RATE_LIMIT_PER_MIN"] = "3"
    m._rate_hits.clear()
    for _ in range(3):
        m.rate_limit(Req("9.9.9.9"))
    expect(m.RateLimited, m.rate_limit, Req("9.9.9.9"))
    m.rate_limit(Req("8.8.8.8"))                         # different client unaffected
    m.rate_limit(Req("7.7.7.7", xff="5.5.5.5, 10.0.0.1"))  # forwarded IP is used
    for _ in range(2):
        m.rate_limit(Req("7.7.7.7", xff="5.5.5.5"))
    expect(m.RateLimited, m.rate_limit, Req("0.0.0.0", xff="5.5.5.5"))
    m._rate_hits["9.9.9.9"].clear()
    m._rate_hits["9.9.9.9"].extend([0.0, 0.0, 0.0])     # old hits fall out of the window
    m.rate_limit(Req("9.9.9.9"))
    os.environ["RATE_LIMIT_PER_MIN"] = "0"
    for _ in range(100):
        m.rate_limit(Req("9.9.9.9"))
    print("ok  rate limiting (limit, per-IP, sliding window, disable)")
    print("\nall tests passed")


if __name__ == "__main__":
    run()
