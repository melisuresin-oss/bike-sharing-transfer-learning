"""Cryptographic binding verifier used for independent tamper tests."""
from __future__ import annotations
from pathlib import Path
from . import core

def verify_entries(entries, resolver=None):
    resolver = resolver or (lambda value: core.path(value))
    seen=set()
    for entry in entries:
        if entry["path"] in seen: raise PermissionError("duplicate bound path: " + entry["path"])
        seen.add(entry["path"]); value=Path(resolver(entry["path"]))
        if not value.is_file() or core.sha256_path(value) != entry["sha256"] or ("bytes" in entry and value.stat().st_size != entry["bytes"]):
            raise PermissionError("bound artifact changed: " + entry["path"])
    return {"status":"PASS","entries":len(entries)}
