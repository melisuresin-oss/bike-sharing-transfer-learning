"""Registered causal deterministic baselines using Phase-A predictors/fitting rows only."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
from . import core

def persistence(panel):
    return np.array(panel.x_hist[:, 0, :, 0], np.float32), np.array(panel.x_hist[:, 0, :, 1] > .5, bool)

def seasonal_naive(panel):
    return np.array(panel.x_week[:, :, 0], np.float32), np.array(panel.x_week[:, :, 1] > .5, bool)

def _hour_of_week(city_id, origin_us):
    zone = ZoneInfo(core.TIMEZONES[city_id])
    return np.fromiter((datetime.fromtimestamp(int(t) / 1_000_000, timezone.utc).astimezone(zone).weekday() * 24 +
                        datetime.fromtimestamp(int(t) / 1_000_000, timezone.utc).astimezone(zone).hour for t in origin_us),
                       dtype=np.int16, count=len(origin_us))

def _snapshot(entry):
    with np.load(core.path(entry["path"]), allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in ("station_id", "origin_us", "count", "observed")}

def _means(rows, city_id):
    station_how = defaultdict(list); station_all = defaultdict(list)
    how = _hour_of_week(city_id, rows["origin_us"])
    for station, bucket, value, observed in zip(rows["station_id"], how, rows["count"], rows["observed"]):
        if bool(observed) and np.isfinite(value):
            station_how[(int(station), int(bucket))].append(float(value)); station_all[int(station)].append(float(value))
    sh = {k: float(np.mean(v)) for k,v in station_how.items()}; sa = {k: float(np.mean(v)) for k,v in station_all.items()}
    city_how = {}; buckets = {b for _,b in sh}
    for bucket in buckets:
        values = [value for (station,b),value in sh.items() if b == bucket]
        if values: city_how[bucket] = float(np.mean(values))
    return {"station_how": sh, "station_all": sa, "city_how": city_how,
            "city_all": float(np.mean(list(sa.values()))) if sa else None}

def _source_fallbacks(data_manifest):
    cities = {}
    for city in core.SOURCES:
        entry = next(x["data"] for x in data_manifest["source_refit_partitions"] if x["request"]["city_ids"] == [city])
        cities[city] = _means(_snapshot(entry), city)
    source_how = {}
    for bucket in range(168):
        values = [cities[c]["city_how"][bucket] for c in core.SOURCES if bucket in cities[c]["city_how"]]
        if values: source_how[bucket] = float(np.mean(values))
    globals_ = [cities[c]["city_all"] for c in core.SOURCES if cities[c]["city_all"] is not None]
    return source_how, float(np.mean(globals_)) if globals_ else None

def historical_average(panel, budget, *, source_only=False):
    data_manifest = core.read_json(core.DATA_MANIFEST); source_how, source_global = _source_fallbacks(data_manifest)
    target_stats = None
    if not source_only:
        rec = next(x for x in data_manifest["targets"] if x["city_id"] == panel.city_id)
        entry = next(x["data"] for x in rec["adaptations"] if x["request"]["budget"] == budget)
        target_stats = _means(_snapshot(entry), panel.city_id)
    output = np.zeros((len(panel.origin_us), len(panel.station_ids)), np.float32); available = np.zeros(output.shape, bool)
    buckets = _hour_of_week(panel.city_id, panel.origin_us)
    for ti,bucket in enumerate(buckets):
        for si,station in enumerate(panel.station_ids):
            value = None
            if target_stats is not None:
                value = target_stats["station_how"].get((int(station), int(bucket)))
                if value is None: value = target_stats["station_all"].get(int(station))
                if value is None: value = target_stats["city_how"].get(int(bucket))
                if value is None: value = target_stats["city_all"]
            if value is None: value = source_how.get(int(bucket))
            if value is None: value = source_global
            if value is not None: output[ti,si] = value; available[ti,si] = True
    return output, available

def predict(method, panel, budget):
    if method == "persistence": return persistence(panel)
    if method == "seasonal_naive": return seasonal_naive(panel)
    if method == "HA_SOURCE": return historical_average(panel, "0", source_only=True)
    if method == "HA_TARGET": return historical_average(panel, budget, source_only=False)
    raise ValueError("unknown deterministic baseline")

