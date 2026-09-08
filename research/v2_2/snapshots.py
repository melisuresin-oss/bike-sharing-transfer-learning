"""Cutoff-specific fitting logic, with no optimizer, HA estimator or evaluator."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .artifacts import ArtifactIdentity
from .contract import DEVELOPMENT, FINAL_TARGETS, HOUR, PSEUDO_TARGETS, Contract, canonical_bytes, hour_us, sha256_bytes
from .history import CausalHistory, count_available, coverage


@dataclass(frozen=True)
class FitRequest:
    phase: str
    kind: str
    budget: str
    start: int
    cutoff: int
    city_ids: tuple[int, ...]

    @classmethod
    def registered(cls, contract: Contract, *, phase: str, kind: str, budget="full", target_city=None):
        if phase not in {"development", "final"} or kind not in {"source", "adaptation"}:
            raise ValueError("Unregistered phase/fit kind")
        if budget not in {"zero", "1", "7", "30", "full"}:
            raise ValueError("Unknown elapsed-time budget")
        cutoff = contract.boundaries["HD" if phase == "development" else "HF"]
        allowed_target = PSEUDO_TARGETS if phase == "development" else FINAL_TARGETS
        if kind == "source":
            if budget != "full" or (phase == "development" and target_city not in PSEUDO_TARGETS):
                raise ValueError("Source fit requires full history and a valid development holdout")
            cities = tuple(c for c in DEVELOPMENT if phase != "development" or c != target_city)
        else:
            if target_city not in allowed_target:
                raise ValueError("Target not registered for this phase")
            cities = (target_city,)
        start = contract.boundaries["H0"] if budget == "full" else cutoff if budget == "zero" else cutoff - int(budget)*24*HOUR
        return cls(phase, kind, budget, start, cutoff, cities)

    def validate(self, contract):
        target = (next((c for c in PSEUDO_TARGETS if c not in self.city_ids), None)
                  if self.kind == "source" and self.phase == "development"
                  else self.city_ids[0] if self.kind == "adaptation" and self.city_ids else None)
        if self != self.registered(contract, phase=self.phase, kind=self.kind,
                                   budget=self.budget, target_city=target):
            raise ValueError("Fit window/scope differs from registered elapsed-time policy")


@dataclass(frozen=True)
class FitRow:
    city_id: int
    station_id: int
    origin: int
    count: int
    origin_feature_sha256: str


@dataclass(frozen=True)
class FitSnapshot:
    request: FitRequest
    identity: ArtifactIdentity
    rows: tuple[FitRow, ...]
    purpose: str

    def payload(self):
        request = asdict(self.request)
        request["city_ids"] = list(request["city_ids"])
        return {"request": request, "rows": [asdict(r) for r in self.rows], "purpose": self.purpose}


def build_fit_snapshot(engine: CausalHistory, request: FitRequest, *, candidate_origins=None) -> FitSnapshot:
    """Implement full logic; explicit subsets are always non-scientific fixtures.

    This function is not invoked over the full registered dataset by any test or
    verifier. There is deliberately no CLI that automatically constructs a panel.
    """
    request.validate(engine.contract)
    if candidate_origins is None:
        origins = range(request.start, request.cutoff, HOUR)
        purpose = "SCIENTIFIC_ARTIFACT"
    else:
        origins = sorted(set(hour_us(t) for t in candidate_origins))
        if any(not request.start <= t < request.cutoff for t in origins) and request.budget != "zero":
            raise ValueError("Fixture origins cannot extend fitting window")
        purpose = "NON_SCIENTIFIC_TEST_ONLY"
    identity = ArtifactIdentity.create(engine.contract, "fit_snapshot", request.cutoff, request.city_ids)
    rows = []
    if request.budget != "zero":
        for city in request.city_ids:
            stations = engine.contract.city(city)
            for origin in origins:
                if not request.start <= origin < request.cutoff or not count_available(origin, request.cutoff):
                    continue
                cv = coverage(engine.contract, engine.status, city, origin, request.cutoff)
                if not any(cv.observed):
                    continue
                # Predictor time remains origin, independent of certification at C.
                feature_hash = engine.for_training(city, origin).fingerprint()
                for station, observed in zip(stations, cv.observed):
                    if observed:
                        value = engine.counts.count(station.station_id, origin, request.cutoff)
                        rows.append(FitRow(city, station.station_id, origin, value, feature_hash))
    return FitSnapshot(request, identity, tuple(sorted(rows, key=lambda r:(r.city_id, r.station_id, r.origin))), purpose)


def seal_training_keys(snapshot: FitSnapshot):
    """Return a content-addressed manifest; persistence/authorization is external."""
    if not snapshot.rows and snapshot.request.budget != "zero":
        raise ValueError("Empty positive-budget fit is non-estimable; do not extend its window")
    keys = [(r.city_id, r.station_id, r.origin) for r in snapshot.rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate training keys")
    manifest = {"identity": snapshot.identity.as_json(), "status": "SEALED_TRAINING_KEYS",
                "purpose": snapshot.purpose, "row_count": len(keys),
                "keys_sha256": sha256_bytes(canonical_bytes(keys)),
                "snapshot_sha256": sha256_bytes(canonical_bytes(snapshot.payload()))}
    return manifest, sha256_bytes(canonical_bytes(manifest))


def require_training_keys(snapshot: FitSnapshot, manifest, expected_sha256: str, *, scientific: bool):
    """Required precondition for every future V2.2 scientific fitting entry point."""
    if manifest is None or not expected_sha256:
        raise PermissionError("A sealed training-key manifest is required before fitting")
    expected, digest = seal_training_keys(snapshot)
    if manifest != expected or expected_sha256 != digest:
        raise PermissionError("Training keys, values, features or cutoff are not bound")
    if scientific and snapshot.purpose != "SCIENTIFIC_ARTIFACT":
        raise PermissionError("Non-scientific fixtures cannot authorize scientific fitting")
    if snapshot.request.budget == "zero":
        raise PermissionError("Zero budget reuses a checkpoint; it cannot start a target fit")
    return snapshot.rows


def historical_average_rows(snapshot, manifest, expected_sha256, *, scientific=True):
    """Same guarded sufficient-statistic rows; deliberately does not fit a mean."""
    return require_training_keys(snapshot, manifest, expected_sha256, scientific=scientific)
