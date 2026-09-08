from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .manifests import ProtocolArtifactRegistry, sql_path
from .protocol_dataset import (
    CALENDAR_COLUMNS,
    HIST_MASK_COLUMNS,
    HIST_VALUE_COLUMNS,
    STATIC_COLUMNS,
    WEEK_COLUMNS,
    PanelScope,
)


def _array(value: Any, dtype=None, fill=np.nan) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        if dtype is not None:
            value = value.astype(dtype)
        value = np.ma.filled(value, fill)
    return np.asarray(value, dtype=dtype)


@dataclass(frozen=True)
class StationWindowBatch:
    city_ids: np.ndarray
    station_ids: np.ndarray
    timestamps: np.ndarray
    x_hist: np.ndarray
    m_hist: np.ndarray
    x_week: np.ndarray
    x_static: np.ndarray
    x_calendar: np.ndarray
    y: np.ndarray | None
    m_target: np.ndarray


class StationWindowLoader:
    """Lazy station-level samples from approved label-bearing panels only."""

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

    def __len__(self) -> int:
        con = self.registry.connect()
        try:
            return int(con.sql(f"SELECT count(*) FROM {self.relation_sql}").fetchone()[0])
        finally:
            con.close()

    def fetch(self, offset: int, batch_size: int) -> StationWindowBatch:
        if offset < 0 or batch_size <= 0:
            raise ValueError("offset must be nonnegative and batch_size positive")
        columns = [
            "city_id",
            "station_id",
            "timestamp_utc",
            *[item for pair in zip(HIST_VALUE_COLUMNS, HIST_MASK_COLUMNS) for item in pair],
            *WEEK_COLUMNS,
            *STATIC_COLUMNS,
            *CALENDAR_COLUMNS,
            "target_12h",
            "coverage_observed_12h",
        ]
        con = self.registry.connect()
        try:
            data = con.execute(
                f"SELECT {','.join(columns)} FROM {self.relation_sql} "
                f"ORDER BY station_id,timestamp_utc LIMIT {int(batch_size)} OFFSET {int(offset)}"
            ).fetchnumpy()
        finally:
            con.close()
        size = len(data["city_id"])
        hist_values = np.column_stack(
            [_array(data[column], np.float32, 0.0) for column in HIST_VALUE_COLUMNS]
        ) if size else np.empty((0, 24), dtype=np.float32)
        hist_masks = np.column_stack(
            [_array(data[column], bool, False) for column in HIST_MASK_COLUMNS]
        ) if size else np.empty((0, 24), dtype=bool)
        x_hist = np.stack(
            [hist_values, hist_masks.astype(np.float32)], axis=-1
        )
        x_week = np.column_stack(
            [
                _array(data[WEEK_COLUMNS[0]], np.float32, 0.0),
                _array(data[WEEK_COLUMNS[1]], np.float32, 0.0),
            ]
        )
        x_static = np.column_stack(
            [_array(data[column], np.float32, 0.0) for column in STATIC_COLUMNS]
        )
        x_calendar = np.column_stack(
            [_array(data[column], np.float32, 0.0) for column in CALENDAR_COLUMNS]
        )
        target = _array(data["target_12h"], np.float32, np.nan)
        mask = _array(data["coverage_observed_12h"], bool, False)
        target = np.where(mask, target, np.nan).astype(np.float32)
        return StationWindowBatch(
            city_ids=_array(data["city_id"], np.int64),
            station_ids=_array(data["station_id"], np.int64),
            timestamps=_array(data["timestamp_utc"]),
            x_hist=x_hist,
            m_hist=hist_masks,
            x_week=x_week,
            x_static=x_static,
            x_calendar=x_calendar,
            y=target,
            m_target=mask,
        )


class FinalStationInferenceLoader:
    """Station samples from label-free final features; Y is structurally absent."""

    def __init__(self, registry: ProtocolArtifactRegistry, city_id: int) -> None:
        role = registry.city_roles[int(city_id)]
        if not role["final_target"]:
            raise ValueError("Final inference loader accepts only final targets")
        self.registry = registry
        self.city_id = int(city_id)
        self.artifact = registry.path("final_features")

    @property
    def relation_sql(self) -> str:
        return (
            f"read_parquet('{sql_path(self.artifact)}') WHERE city_id={self.city_id}"
        )

    def __len__(self) -> int:
        con = self.registry.connect()
        try:
            return int(con.sql(f"SELECT count(*) FROM {self.relation_sql}").fetchone()[0])
        finally:
            con.close()

    def fetch(self, offset: int, batch_size: int) -> StationWindowBatch:
        columns = [
            "city_id",
            "station_id",
            "timestamp_utc",
            *[item for pair in zip(HIST_VALUE_COLUMNS, HIST_MASK_COLUMNS) for item in pair],
            *WEEK_COLUMNS,
            *STATIC_COLUMNS,
            *CALENDAR_COLUMNS,
            "coverage_observed_12h",
        ]
        con = self.registry.connect()
        try:
            data = con.execute(
                f"SELECT {','.join(columns)} FROM {self.relation_sql} "
                f"ORDER BY station_id,timestamp_utc LIMIT {int(batch_size)} OFFSET {int(offset)}"
            ).fetchnumpy()
        finally:
            con.close()
        size = len(data["city_id"])
        hist_values = np.column_stack(
            [_array(data[column], np.float32, 0.0) for column in HIST_VALUE_COLUMNS]
        ) if size else np.empty((0, 24), dtype=np.float32)
        hist_masks = np.column_stack(
            [_array(data[column], bool, False) for column in HIST_MASK_COLUMNS]
        ) if size else np.empty((0, 24), dtype=bool)
        return StationWindowBatch(
            city_ids=_array(data["city_id"], np.int64),
            station_ids=_array(data["station_id"], np.int64),
            timestamps=_array(data["timestamp_utc"]),
            x_hist=np.stack([hist_values, hist_masks.astype(np.float32)], axis=-1),
            m_hist=hist_masks,
            x_week=np.column_stack(
                [
                    _array(data[WEEK_COLUMNS[0]], np.float32, 0.0),
                    _array(data[WEEK_COLUMNS[1]], np.float32, 0.0),
                ]
            ),
            x_static=np.column_stack(
                [_array(data[column], np.float32, 0.0) for column in STATIC_COLUMNS]
            ),
            x_calendar=np.column_stack(
                [_array(data[column], np.float32, 0.0) for column in CALENDAR_COLUMNS]
            ),
            y=None,
            m_target=_array(data["coverage_observed_12h"], bool, False),
        )
