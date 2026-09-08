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
)
from .window_dataset import _array


@dataclass(frozen=True)
class GraphWindowBatch:
    city_id: int
    station_ids: np.ndarray
    timestamps: np.ndarray
    x_hist: np.ndarray
    m_hist: np.ndarray
    x_week: np.ndarray
    x_static: np.ndarray
    x_calendar: np.ndarray
    y: np.ndarray | None
    m_target: np.ndarray


class GraphCityHourLoader:
    """City-homogeneous, frozen-node-order graph samples.

    This class creates no adjacency and contains no graph-learning logic.
    """

    def __init__(
        self,
        registry: ProtocolArtifactRegistry,
        city_id: int,
        *,
        start_utc: str | None = None,
        end_utc: str | None = None,
        eligible_anchors_only: bool = False,
        final_inference: bool = False,
    ) -> None:
        self.registry = registry
        self.city_id = int(city_id)
        self.final_inference = bool(final_inference)
        role = registry.city_roles[self.city_id]
        if final_inference:
            if not role["final_target"]:
                raise ValueError("Final inference accepts only sealed final targets")
            self.artifact = registry.path("final_features")
        else:
            self.artifact = registry.panel_for_city(self.city_id)
        self.station_artifact = registry.station_manifest(self.city_id)
        clauses = [f"city_id={self.city_id}"]
        if start_utc:
            clauses.append(f"timestamp_utc >= TIMESTAMPTZ '{start_utc}'")
        if end_utc:
            clauses.append(f"timestamp_utc < TIMESTAMPTZ '{end_utc}'")
        self.where_sql = " AND ".join(clauses)
        con = registry.connect()
        try:
            having = "HAVING count_if(coverage_observed_12h)>0" if eligible_anchors_only else ""
            self._timestamps = [
                row[0]
                for row in con.sql(
                    f"SELECT timestamp_utc FROM read_parquet('{sql_path(self.artifact)}') "
                    f"WHERE {self.where_sql} GROUP BY timestamp_utc {having} "
                    "ORDER BY timestamp_utc"
                ).fetchall()
            ]
            manifest_rows = con.sql(
                f"SELECT station_id FROM read_parquet('{sql_path(self.station_artifact)}') "
                "ORDER BY node_index"
            ).fetchall()
            self.station_ids = np.asarray([row[0] for row in manifest_rows], dtype=np.int64)
        finally:
            con.close()

    def __len__(self) -> int:
        return len(self._timestamps)

    @property
    def node_count(self) -> int:
        return len(self.station_ids)

    def fetch(self, offset: int, batch_size: int) -> GraphWindowBatch:
        if offset < 0 or batch_size <= 0:
            raise ValueError("offset must be nonnegative and batch_size positive")
        timestamps = self._timestamps[offset : offset + batch_size]
        if not timestamps:
            empty = np.empty((0, 24, self.node_count, 2), dtype=np.float32)
            return GraphWindowBatch(
                city_id=self.city_id,
                station_ids=self.station_ids.copy(),
                timestamps=np.empty(0, dtype="datetime64[us]"),
                x_hist=empty,
                m_hist=np.empty((0, 24, self.node_count), dtype=bool),
                x_week=np.empty((0, self.node_count, 2), dtype=np.float32),
                x_static=np.empty((self.node_count, 2), dtype=np.float32),
                x_calendar=np.empty((0, 6), dtype=np.float32),
                y=None if self.final_inference else np.empty((0, self.node_count)),
                m_target=np.empty((0, self.node_count), dtype=bool),
            )
        placeholders = ",".join("?" for _ in timestamps)
        target_columns = [] if self.final_inference else ["p.target_12h"]
        columns = [
            "p.timestamp_utc",
            "p.station_id",
            "m.node_index",
            *[f"p.{item}" for pair in zip(HIST_VALUE_COLUMNS, HIST_MASK_COLUMNS) for item in pair],
            *[f"p.{item}" for item in WEEK_COLUMNS],
            *[f"p.{item}" for item in STATIC_COLUMNS],
            *[f"p.{item}" for item in CALENDAR_COLUMNS],
            *target_columns,
            "p.coverage_observed_12h",
        ]
        con = self.registry.connect()
        try:
            data = con.execute(
                f"SELECT {','.join(columns)} "
                f"FROM read_parquet('{sql_path(self.artifact)}') p "
                f"JOIN read_parquet('{sql_path(self.station_artifact)}') m "
                "USING(city_id,station_id) "
                f"WHERE p.city_id={self.city_id} AND p.timestamp_utc IN ({placeholders}) "
                "ORDER BY p.timestamp_utc,m.node_index",
                timestamps,
            ).fetchnumpy()
        finally:
            con.close()
        batch = len(timestamps)
        expected = batch * self.node_count
        if len(data["station_id"]) != expected:
            raise RuntimeError(
                f"Incomplete graph batch for city {self.city_id}: expected {expected} rows"
            )
        observed_station_order = _array(data["station_id"], np.int64).reshape(
            batch, self.node_count
        )
        if not np.all(observed_station_order == self.station_ids[None, :]):
            raise RuntimeError("Frozen node ordering differs inside graph batch")
        hist_values = np.stack(
            [
                _array(data[column], np.float32, 0.0).reshape(batch, self.node_count)
                for column in HIST_VALUE_COLUMNS
            ],
            axis=1,
        )
        hist_masks = np.stack(
            [
                _array(data[column], bool, False).reshape(batch, self.node_count)
                for column in HIST_MASK_COLUMNS
            ],
            axis=1,
        )
        x_hist = np.stack([hist_values, hist_masks.astype(np.float32)], axis=-1)
        x_week = np.stack(
            [
                _array(data[WEEK_COLUMNS[0]], np.float32, 0.0).reshape(batch, self.node_count),
                _array(data[WEEK_COLUMNS[1]], np.float32, 0.0).reshape(batch, self.node_count),
            ],
            axis=-1,
        )
        static_cube = np.stack(
            [
                _array(data[column], np.float32, 0.0).reshape(batch, self.node_count)
                for column in STATIC_COLUMNS
            ],
            axis=-1,
        )
        if not np.allclose(static_cube, static_cube[0:1], equal_nan=True):
            raise RuntimeError("Static station features changed across hours")
        calendar_cube = np.stack(
            [
                _array(data[column], np.float32, 0.0).reshape(batch, self.node_count)
                for column in CALENDAR_COLUMNS
            ],
            axis=-1,
        )
        if not np.allclose(calendar_cube, calendar_cube[:, 0:1, :], equal_nan=True):
            raise RuntimeError("Calendar features differ across stations in a city-hour")
        m_target = _array(data["coverage_observed_12h"], bool, False).reshape(
            batch, self.node_count
        )
        y = None
        if not self.final_inference:
            y_raw = _array(data["target_12h"], np.float32, np.nan).reshape(
                batch, self.node_count
            )
            y = np.where(m_target, y_raw, np.nan).astype(np.float32)
        return GraphWindowBatch(
            city_id=self.city_id,
            station_ids=self.station_ids.copy(),
            timestamps=np.asarray(timestamps),
            x_hist=x_hist,
            m_hist=hist_masks,
            x_week=x_week,
            x_static=static_cube[0],
            x_calendar=calendar_cube[:, 0, :],
            y=y,
            m_target=m_target,
        )
