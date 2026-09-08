"""Pinned specification, immutable cohort, and exact UTC time conversion."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[2]
BASE = "research/results/causal_history_spec_v2_2/"
SPEC_SHA256 = "c85c8fdbcab26e7239bfb4528b570b720e7d31ea5933934d9258b63ad90f9277"
SEAL_SHA256 = "170d2c3c8b3371dd6c6c3d0606733d6d9a6b93e83449808dd7ccb98a904a86fb"
COHORT_SHA256 = "24f25a4d18dafe888dffc74acc279e59e51ef5c68c03eaa9e8148c632ae4a046"
HOUR = 3_600_000_000
BRACKET = 12 * HOUR
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
DEVELOPMENT = (129, 194, 438, 467, 476, 532, 619, 658)
PSEUDO_TARGETS = (476, 532, 619, 658)
FINAL_TARGETS = (195, 199, 237, 617)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def utc_us(value: datetime | str | int) -> int:
    """Integers are already UTC microseconds; no floating-point timestamp math."""
    if type(value) is int:
        return value
    if isinstance(value, str):
        if re.search(r"\.\d{7,}", value):
            raise ValueError("Sub-microsecond timestamps cannot be silently truncated")
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("An aware timestamp or integer UTC microseconds is required")
    delta = value.astimezone(timezone.utc) - EPOCH
    return ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds)


def hour_us(value) -> int:
    value = utc_us(value)
    if value % HOUR:
        raise ValueError("Origin/hour boundary must be aligned to an elapsed UTC hour")
    return value


@dataclass(frozen=True)
class Station:
    city_id: int
    station_id: int
    node_index: int
    latitude: float
    longitude: float
    bike_racks: int | None
    timezone: str


class Contract:
    """Only the accepted fixed cohort can construct production/test engines.

    Tests use real roster identities with synthetic observations, never a new
    eligibility rule. No scientific data artifact is opened here.
    """
    def __init__(self, root: Path = ROOT):
        self.root = root.resolve()
        def pinned(name, expected):
            data = (self.root / BASE / name).read_bytes()
            if sha256_bytes(data) != expected:
                raise ValueError("Sealed binding mismatch: " + name)
            return json.loads(data)
        seal = pinned("causal_history_spec_seal.json", SEAL_SHA256)
        spec = pinned("v2_2_specification.json", SPEC_SHA256)
        cohort = pinned("fixed_cohort_static_manifest.json", COHORT_SHA256)
        if seal["protocol_version"] != "2.2" or spec["count_feed"]["Q"] != "1[e <= a]":
            raise ValueError("Unsupported contract")
        self.stations = tuple(Station(**row) for row in cohort["stations"])
        if len(self.stations) != 799 or len({s.station_id for s in self.stations}) != 799:
            raise ValueError("Fixed roster differs from 799 identities")
        self.by_id = MappingProxyType({s.station_id: s for s in self.stations})
        self.by_city = MappingProxyType({c: tuple(s for s in self.stations if s.city_id == c)
                                        for c in sorted({s.city_id for s in self.stations})})
        self.boundaries = MappingProxyType({k: hour_us(v) for k, v in
                                           spec["carried_forward_design"]["boundaries"].items()})
        self.roster_sha256 = sha256_bytes(canonical_bytes(
            [(s.city_id, s.station_id, s.node_index) for s in self.stations]))

    def city(self, city_id: int) -> tuple[Station, ...]:
        try:
            return self.by_city[city_id]
        except KeyError as exc:
            raise ValueError("City outside fixed cohort") from exc

    def city_roster_hash(self, city_id: int) -> str:
        return sha256_bytes(canonical_bytes([(s.station_id, s.node_index) for s in self.city(city_id)]))

    def validate_static_rows(self, city_id: int, rows) -> None:
        expected = tuple(self.city(city_id))
        received = tuple(Station(**r) if isinstance(r, dict) else r for r in rows)
        if received != expected:
            raise ValueError("Static fields, membership or node order changed")
