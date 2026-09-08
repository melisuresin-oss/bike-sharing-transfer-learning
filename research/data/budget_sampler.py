from __future__ import annotations

import random
from dataclasses import dataclass

from .manifests import ProtocolArtifactRegistry, sql_path


BUDGET_COLUMNS = {
    "parameter_zero": None,
    "1_day": "budget_1_day",
    "7_days": "budget_7_days",
    "30_days": "budget_30_days",
    "full": "budget_full",
}


class BudgetLoader:
    """Deterministic, manifest-reconciled access to adaptation-label keys."""

    def __init__(self, registry: ProtocolArtifactRegistry) -> None:
        self.registry = registry

    def count(self, city_id: int, budget: str) -> int:
        if budget not in BUDGET_COLUMNS:
            raise KeyError(budget)
        if budget == "parameter_zero":
            return 0
        path = self.registry.panel_for_city(city_id)
        column = BUDGET_COLUMNS[budget]
        con = self.registry.connect()
        try:
            return int(
                con.sql(
                    f"SELECT count(*) FROM read_parquet('{sql_path(path)}') "
                    f"WHERE city_id={int(city_id)} AND {column} "
                    "AND coverage_observed_12h"
                ).fetchone()[0]
            )
        finally:
            con.close()

    def validate_all(self) -> list[dict[str, int | str | bool]]:
        results = []
        for (city_id, budget), expected in sorted(
            self.registry.expected_budget_counts.items()
        ):
            actual = self.count(city_id, budget)
            result = {
                "city_id": city_id,
                "budget": budget,
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
            }
            results.append(result)
            if actual != expected:
                raise RuntimeError(
                    f"Frozen budget mismatch for city={city_id}, budget={budget}: "
                    f"expected {expected}, got {actual}"
                )
        return results

    def keys(self, city_id: int, budget: str) -> list[tuple[int, int, object]]:
        if budget == "parameter_zero":
            return []
        if budget not in BUDGET_COLUMNS:
            raise KeyError(budget)
        path = self.registry.panel_for_city(city_id)
        column = BUDGET_COLUMNS[budget]
        con = self.registry.connect()
        try:
            return con.sql(
                f"SELECT city_id,station_id,timestamp_utc "
                f"FROM read_parquet('{sql_path(path)}') "
                f"WHERE city_id={int(city_id)} AND {column} "
                "AND coverage_observed_12h ORDER BY station_id,timestamp_utc"
            ).fetchall()
        finally:
            con.close()


@dataclass(frozen=True)
class EqualCitySampler:
    city_ids: tuple[int, ...]
    seed: int

    def sequence(self, updates: int) -> list[int]:
        if updates < 0:
            raise ValueError("updates must be nonnegative")
        if not self.city_ids and updates:
            raise ValueError("at least one city is required")
        base = [self.city_ids[index % len(self.city_ids)] for index in range(updates)]
        rng = random.Random(self.seed)
        rng.shuffle(base)
        counts = {city_id: base.count(city_id) for city_id in self.city_ids}
        if counts and max(counts.values()) - min(counts.values()) > 1:
            raise RuntimeError("Equal-city sequence is unexpectedly imbalanced")
        return base
