from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifests import ProtocolArtifactRegistry, sql_path


HIST_VALUE_COLUMNS = [f"hist_lag_{lag:03d}_log1p" for lag in range(1, 25)]
HIST_MASK_COLUMNS = [f"hist_lag_{lag:03d}_observed" for lag in range(1, 25)]
WEEK_COLUMNS = ["week_lag_168_log1p", "week_lag_168_observed"]
STATIC_COLUMNS = ["static_bike_racks_log1p", "static_bike_racks_valid"]
CALENDAR_COLUMNS = [
    "calendar_local_hour_sin",
    "calendar_local_hour_cos",
    "calendar_local_weekday_sin",
    "calendar_local_weekday_cos",
    "calendar_weekend",
    "calendar_utc_offset_div_12",
]
MODEL_FEATURE_COLUMNS = [
    item
    for pair in zip(HIST_VALUE_COLUMNS, HIST_MASK_COLUMNS)
    for item in pair
] + WEEK_COLUMNS + STATIC_COLUMNS + CALENDAR_COLUMNS


@dataclass(frozen=True)
class PanelScope:
    city_id: int
    artifact: Path
    start_utc: str | None = None
    end_utc: str | None = None
    require_target: bool = False

    def where_sql(self) -> str:
        clauses = [f"city_id={int(self.city_id)}"]
        if self.start_utc:
            clauses.append(f"timestamp_utc >= TIMESTAMPTZ '{self.start_utc}'")
        if self.end_utc:
            clauses.append(f"timestamp_utc < TIMESTAMPTZ '{self.end_utc}'")
        if self.require_target:
            clauses.append("coverage_observed_12h")
        return " AND ".join(clauses)


class ProtocolPanel:
    """Lazy, fixed-artifact panel reader used by model-development loaders."""

    def __init__(
        self,
        registry: ProtocolArtifactRegistry,
        city_id: int,
        *,
        start_utc: str | None = None,
        end_utc: str | None = None,
        require_target: bool = False,
    ) -> None:
        self.registry = registry
        self.role = registry.city_roles[int(city_id)]
        self.scope = PanelScope(
            city_id=int(city_id),
            artifact=registry.panel_for_city(city_id),
            start_utc=start_utc,
            end_utc=end_utc,
            require_target=require_target,
        )

    @property
    def relation_sql(self) -> str:
        return (
            f"read_parquet('{sql_path(self.scope.artifact)}') "
            f"WHERE {self.scope.where_sql()}"
        )

    def count(self) -> int:
        con = self.registry.connect()
        try:
            return int(con.sql(f"SELECT count(*) FROM {self.relation_sql}").fetchone()[0])
        finally:
            con.close()

    def query(self, select_sql: str, suffix_sql: str = "") -> list[tuple[Any, ...]]:
        con = self.registry.connect()
        try:
            return con.sql(
                f"SELECT {select_sql} FROM {self.relation_sql} {suffix_sql}"
            ).fetchall()
        finally:
            con.close()


class FinalInferencePanel:
    """Label-free final feature and prediction-key access."""

    def __init__(self, registry: ProtocolArtifactRegistry, city_id: int) -> None:
        role = registry.city_roles[int(city_id)]
        if not role["final_target"]:
            raise ValueError("Final inference is defined only for sealed final targets")
        self.registry = registry
        self.city_id = int(city_id)
        self.features = registry.path("final_features")
        self.keys = registry.path("final_prediction_keys")

    @property
    def relation_sql(self) -> str:
        return (
            f"read_parquet('{sql_path(self.features)}') "
            f"WHERE city_id={self.city_id}"
        )

    def count(self) -> int:
        con = self.registry.connect()
        try:
            return int(con.sql(f"SELECT count(*) FROM {self.relation_sql}").fetchone()[0])
        finally:
            con.close()
