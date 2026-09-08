"""Fail-closed V2.2 identities and allowlisted artifact access. No pickle loader."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from . import FEATURE_FUNCTION_VERSION, PROTOCOL_VERSION
from .contract import COHORT_SHA256, SEAL_SHA256, SPEC_SHA256, DEVELOPMENT, FINAL_TARGETS, Contract, canonical_bytes, sha256_bytes, utc_us

ROLES = frozenset({"predictor_cache", "fit_snapshot", "checkpoint", "graph"})


@dataclass(frozen=True)
class ArtifactIdentity:
    artifact_role: str
    cutoff_us: int
    city_ids: tuple[int, ...]
    roster_sha256: str
    graph_binding_sha256: str | None = None
    model_family: str = "none"
    protocol_version: str = PROTOCOL_VERSION
    specification_sha256: str = SPEC_SHA256
    specification_seal_sha256: str = SEAL_SHA256
    cohort_static_sha256: str = COHORT_SHA256
    feature_function_version: str = FEATURE_FUNCTION_VERSION
    schema_version: str = "1.0"

    def __post_init__(self):
        if (self.protocol_version != PROTOCOL_VERSION or self.specification_sha256 != SPEC_SHA256
                or self.specification_seal_sha256 != SEAL_SHA256 or self.cohort_static_sha256 != COHORT_SHA256
                or self.feature_function_version != FEATURE_FUNCTION_VERSION or self.schema_version != "1.0"):
            raise ValueError("Only the exact sealed V2.2 identity is supported")
        if self.artifact_role not in ROLES or type(self.cutoff_us) is not int:
            raise ValueError("Invalid artifact role or cutoff")
        if not self.city_ids or self.city_ids != tuple(sorted(set(self.city_ids))) or not set(self.city_ids) <= set(DEVELOPMENT+FINAL_TARGETS):
            raise ValueError("Invalid fixed city scope")
        if self.model_family not in {"none", "graph_gru", "vanilla_gru"}:
            raise ValueError("Unsupported model family")
        if (self.artifact_role == "graph" or self.model_family == "graph_gru") and not self.graph_binding_sha256:
            raise ValueError("Graph binding required")
        for digest in (self.roster_sha256, self.graph_binding_sha256):
            if digest is not None and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Invalid identity digest")

    @classmethod
    def create(cls, contract: Contract, role: str, cutoff, city_ids, *,
               model_family="none", graph_binding_sha256=None):
        if role not in ROLES or model_family not in {"none", "graph_gru", "vanilla_gru"}:
            raise ValueError("Unapproved artifact role or model family")
        cities = tuple(sorted(set(city_ids)))
        if not cities:
            raise ValueError("Artifact requires an explicit city scope")
        for city in cities:
            contract.city(city)
        if (role == "graph" or model_family == "graph_gru") and not graph_binding_sha256:
            raise ValueError("Geographic graph binding is required")
        if graph_binding_sha256 is not None and (
                len(graph_binding_sha256) != 64 or any(c not in "0123456789abcdef" for c in graph_binding_sha256)):
            raise ValueError("Invalid graph SHA256")
        return cls(role, utc_us(cutoff), cities, contract.roster_sha256,
                   graph_binding_sha256, model_family)

    def as_json(self):
        value = asdict(self)
        value["city_ids"] = list(self.city_ids)
        return value

    def validate(self, received):
        if received != self.as_json():
            raise ValueError("V2.2 identity mismatch: version/specification/cohort/function/cutoff/roster/graph")


def envelope(identity: ArtifactIdentity, payload, *, purpose: str):
    if purpose not in {"NON_SCIENTIFIC_TEST_ONLY", "SCIENTIFIC_ARTIFACT"}:
        raise ValueError("Explicit artifact purpose required")
    return {"identity": identity.as_json(), "purpose": purpose,
            "payload_sha256": sha256_bytes(canonical_bytes(payload)), "payload": payload}


def validate_envelope(value, expected: ArtifactIdentity):
    if set(value) != {"identity", "purpose", "payload_sha256", "payload"}:
        raise ValueError("Unsupported artifact schema")
    expected.validate(value["identity"])
    if value["purpose"] not in {"NON_SCIENTIFIC_TEST_ONLY", "SCIENTIFIC_ARTIFACT"}:
        raise ValueError("Invalid artifact purpose")
    if sha256_bytes(canonical_bytes(value["payload"])) != value["payload_sha256"]:
        raise ValueError("Artifact content hash mismatch")
    return value["payload"]


class ArtifactLoader:
    """Names must be registered in an externally hash-bound V2.2 index.

    Evaluation labels are a different authority and intentionally have no role
    here. Neither arbitrary paths nor legacy scientific-artifact exceptions exist.
    """
    def __init__(self, root: Path, index: dict, expected_index_sha256: str):
        if sha256_bytes(canonical_bytes(index)) != expected_index_sha256:
            raise ValueError("Artifact index is not bound")
        if index.get("protocol_version") != "2.2" or index.get("specification_seal_sha256") != SEAL_SHA256:
            raise ValueError("Legacy/unbound artifact index")
        self.root = root.resolve()
        self.entries = json.loads(canonical_bytes(index["artifacts"]))

    def path(self, name: str, expected: ArtifactIdentity) -> Path:
        entry = self.entries[name]
        expected.validate(entry["identity"])
        relative = entry["path"]
        if not isinstance(relative, str) or "\\" in relative:
            raise ValueError("Require a relative POSIX path")
        lowered = relative.lower()
        if any(x in lowered for x in ("final_label", "evaluation_label", "sealed_final", "v2_1", "protocol_v2_1")):
            raise PermissionError("Final labels and V2.1 scientific paths are forbidden")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise PermissionError("Artifact path escape")
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError("Resolved artifact escapes registry root") from exc
        if path.suffix != ".json":
            raise ValueError("Only explicit JSON envelopes are supported; no unsafe deserialization")
        return path

    def load(self, name: str, expected: ArtifactIdentity):
        data = self.path(name, expected).read_bytes()
        if sha256_bytes(data) != self.entries[name]["sha256"]:
            raise ValueError("Artifact byte hash mismatch")
        return validate_envelope(json.loads(data), expected)
