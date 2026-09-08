"""One pure as-of history function. It has no label-store or panel dependency."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import timedelta
import math
import struct
from types import MappingProxyType
from zoneinfo import ZoneInfo

from .contract import BRACKET, EPOCH, HOUR, Contract, canonical_bytes, hour_us, sha256_bytes, utc_us
from .artifacts import ArtifactIdentity


@dataclass(frozen=True)
class StatusEvent:
    station_id: int
    timestamp: int | None


class StatusIndex:
    def __init__(self, contract: Contract, events):
        values = {s.station_id: set() for s in contract.stations}
        for event in events:
            if not isinstance(event, StatusEvent):
                raise TypeError("Explicit StatusEvent records are required")
            if event.timestamp is None or event.station_id not in values:
                continue
            values[event.station_id].add(utc_us(event.timestamp))
        self.timestamps = MappingProxyType({i: tuple(sorted(v)) for i, v in values.items()})
        self.roster_sha256 = contract.roster_sha256

    def evidence(self, station_id: int, start: int, cutoff: int):
        """Every selected timestamp is inside the prefix, including extrema."""
        times = self.timestamps[station_id]
        stop = bisect_right(times, cutoff)
        if stop == 0:
            return None, None, None, None
        first, last = times[0], times[stop - 1]
        pi = bisect_right(times, start, 0, stop) - 1
        ni = bisect_left(times, start + HOUR, 0, stop)
        return (times[pi] if pi >= 0 else None,
                times[ni] if ni < stop else None, first, last)


@dataclass(frozen=True)
class StationEvidence:
    station_id: int
    previous: int | None
    next: int | None
    first: int | None
    last: int | None
    lifetime: bool
    bracket: bool


@dataclass(frozen=True)
class Coverage:
    city_id: int
    hour_start: int
    cutoff: int
    evidence: tuple[StationEvidence, ...]
    denominator: int
    numerator: int
    city_observed: bool
    observed: tuple[bool, ...]


def city_gate(numerator: int, denominator: int) -> bool:
    if type(numerator) is not int or type(denominator) is not int or not 0 <= numerator <= denominator:
        raise ValueError("Invalid integer network counts")
    return denominator > 0 and 2 * numerator >= denominator


def coverage(contract: Contract, status: StatusIndex, city_id: int, hour_start, cutoff) -> Coverage:
    start, cutoff = hour_us(hour_start), utc_us(cutoff)
    if status.roster_sha256 != contract.roster_sha256:
        raise ValueError("Status index roster mismatch")
    evidence = []
    for station in contract.city(city_id):
        p, n, first, last = status.evidence(station.station_id, start, cutoff)
        life = first is not None and first <= start <= last
        bracket = (p is not None and n is not None and
                   0 <= start - p <= BRACKET and 0 <= n - (start + HOUR) <= BRACKET)
        evidence.append(StationEvidence(station.station_id, p, n, first, last, life, bracket))
    dn = sum(int(x.lifetime) for x in evidence)
    nn = sum(int(x.lifetime and x.bracket) for x in evidence)
    gate = city_gate(nn, dn)
    return Coverage(city_id, start, cutoff, tuple(evidence), dn, nn, gate,
                    tuple(bool(x.lifetime and x.bracket and gate) for x in evidence))


def count_available(hour_start, cutoff) -> bool:
    return hour_us(hour_start) + HOUR <= utc_us(cutoff)


class IdealizedCountFeed:
    """Explicit completed-count ledger; absence inside its domain means zero.

    The caller attests that this is an exhaustive finalized reconstructed-count
    ledger for the declared stations and half-open interval. Missing coverage of
    the feed is an error, never an implicit observed zero. No target masks accepted.
    """
    def __init__(self, contract: Contract, station_ids, start, end, counts):
        self.start, self.end = hour_us(start), hour_us(end)
        self.station_ids = frozenset(station_ids)
        if self.start >= self.end or not self.station_ids or not self.station_ids <= contract.by_id.keys():
            raise ValueError("Invalid count-feed domain")
        copied = {}
        for (station, hour), count in counts.items():
            hour = hour_us(hour)
            if station not in self.station_ids or not self.start <= hour < self.end:
                raise ValueError("Count outside declared feed domain")
            if type(count) is not int or count < 0:
                raise ValueError("A nonnegative integer completed count is required")
            copied[(station, hour)] = count
        self._counts = MappingProxyType(copied)

    def count(self, station_id: int, start: int, cutoff: int) -> int:
        if not count_available(start, cutoff):
            raise ValueError("Unfinished hour is not count-available")
        if station_id not in self.station_ids or not self.start <= start < self.end:
            raise ValueError("Count-feed domain does not certify this hour")
        return self._counts.get((station_id, start), 0)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


@dataclass(frozen=True)
class History:
    identity: ArtifactIdentity
    city_id: int
    origin: int
    station_ids: tuple[int, ...]
    x_hist: tuple  # [24,N,2], storage lag 1..24, compatible with registered model flip
    m_hist: tuple  # [24,N], exactly the indicator channel
    x_week: tuple  # [N,2]
    x_static: tuple
    x_calendar: tuple
    lineage: tuple[Coverage, ...]

    @property
    def chronological(self):
        """Recurrent order t-24,...,t-1. Do not feed this to a model that flips again."""
        return self.x_hist[::-1]

    def tensor_bytes(self) -> bytes:
        values = [v for lag in self.x_hist for pair in lag for v in pair]
        values += [v for pair in self.x_week for v in pair]
        values += [v for pair in self.x_static for v in pair] + list(self.x_calendar)
        return struct.pack("<" + "f" * len(values), *values) + bytes(
            int(v) for lag in self.m_hist for v in lag)

    def fingerprint(self) -> str:
        return sha256_bytes(canonical_bytes((self.identity.as_json(), self.city_id, self.origin, self.station_ids)) + self.tensor_bytes())

    def arrays(self):
        """Non-model adapter: fresh float32/Boolean arrays; no labels included."""
        import numpy as np
        return {"x_hist": np.array(self.x_hist, dtype=np.float32),
                "m_hist": np.array(self.m_hist, dtype=bool),
                "x_week": np.array(self.x_week, dtype=np.float32),
                "x_static": np.array(self.x_static, dtype=np.float32),
                "x_calendar": np.array(self.x_calendar, dtype=np.float32)}


class CausalHistory:
    def __init__(self, contract: Contract, status: StatusIndex, counts: IdealizedCountFeed):
        if type(contract) is not Contract or type(status) is not StatusIndex or type(counts) is not IdealizedCountFeed:
            raise TypeError("Predictors require the bound contract, status index and explicit idealized count feed")
        self.contract, self.status, self.counts = contract, status, counts

    def at_origin(self, city_id: int, origin) -> History:
        origin = hour_us(origin)
        stations = self.contract.city(city_id)
        if not {s.station_id for s in stations} <= self.counts.station_ids:
            raise ValueError("Feed must cover the entire frozen city roster")
        pairs, lineage = [], []
        for lag in (*range(1, 25), 168):
            start = origin - lag * HOUR
            cv = coverage(self.contract, self.status, city_id, start, origin)
            lineage.append(cv)
            row = []
            for station, observed in zip(stations, cv.observed):
                usable = observed and start >= self.contract.boundaries["H0"] and count_available(start, origin)
                row.append((_f32(math.log1p(self.counts.count(station.station_id, start, origin))), 1.0)
                           if usable else (0.0, 0.0))
            pairs.append(tuple(row))
        local = (EPOCH + timedelta(microseconds=origin)).astimezone(ZoneInfo(stations[0].timezone))
        u, d = local.hour, local.weekday()
        calendar = tuple(_f32(v) for v in (math.sin(2*math.pi*u/24), math.cos(2*math.pi*u/24),
                         math.sin(2*math.pi*d/7), math.cos(2*math.pi*d/7), float(d >= 5),
                         local.utcoffset().total_seconds()/43200))
        static = tuple((_f32(math.log1p(s.bike_racks)), 1.0) if s.bike_racks is not None and s.bike_racks > 0
                       else (0.0, 0.0) for s in stations)
        hist = tuple(pairs[:24])
        identity = ArtifactIdentity.create(self.contract, "predictor_cache", origin, [city_id])
        return History(identity, city_id, origin, tuple(s.station_id for s in stations), hist,
                       tuple(tuple(bool(v[1]) for v in row) for row in hist), pairs[24],
                       static, calendar, tuple(lineage))

    def for_training(self, city_id, origin):
        return self.at_origin(city_id, origin)

    def for_inference(self, city_id, origin):
        return self.at_origin(city_id, origin)
