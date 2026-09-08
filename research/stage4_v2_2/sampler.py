"""Reuse the exact sealed EqualCitySampler class without loading V2.1 readers."""
import ast
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

SOURCE=Path(__file__).resolve().parents[2]/"research/data/budget_sampler.py"
EXPECTED="07c2b5060c29ea9e0e629c065c3ec5ed5dbaa1700f560e5043e7aa59bcdb2edd"
raw=SOURCE.read_bytes()
if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise RuntimeError("Frozen EqualCitySampler source changed")
tree=ast.parse(raw)
classes=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="EqualCitySampler"]
if len(classes)!=1:raise RuntimeError("Missing sealed sampler class")
# Compile only the original, hash-pinned class. The legacy BudgetLoader,
# ProtocolArtifactRegistry and their imports are never executed or instantiated.
scope={"__name__":__name__,"random":random,"dataclass":dataclass}
exec(compile(ast.Module(body=classes,type_ignores=[]),str(SOURCE),"exec"),scope)
EqualCitySampler=scope["EqualCitySampler"]
