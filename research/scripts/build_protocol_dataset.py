#!/usr/bin/env python3
"""Build the frozen V2.1 station-hour protocol dataset.

This script is deliberately pre-model.  It constructs the label-independent
coverage masks, nullable targets, causal fixed features, split-safe artifacts,
and validation evidence registered by RESEARCH_PROTOCOL_V2_1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]


def _import_duckdb():
    try:
        import duckdb  # type: ignore

        return duckdb
    except ModuleNotFoundError:
        local = ROOT / "tmp" / "protocol_build" / "pydeps"
        if local.exists():
            sys.path.insert(0, str(local))
            import duckdb  # type: ignore

            return duckdb
        raise SystemExit(
            "DuckDB is required. Install research/requirements.txt before running."
        )


duckdb = _import_duckdb()

PROTOCOL_VERSION = "2.1"
PINNED_REVISION = "d8e897ee0f3a58457cbcfa63bf7454142d1ad54e"
H0 = "2022-08-29T05:00:00Z"
HD = "2023-02-15T22:00:00Z"
HF = "2023-04-16T22:00:00Z"
HT = "2023-05-11T15:00:00Z"
HE = "2023-07-15T20:00:00Z"
STATION_THRESHOLD = 0.50
BRACKET_HOURS_PRIMARY = 12
BRACKET_HOURS_SENSITIVITY = 24
CITY_BRACKET_FRACTION = 0.50

# city_id, canonical city, country, source, pseudo-target, sealed final target
CITY_ROLES: tuple[tuple[int, str, str, bool, bool, bool], ...] = (
    (129, "Dortmund", "DE", True, False, False),
    (194, "Heidelberg", "DE", True, False, False),
    (195, "Mannheim", "DE", False, False, True),
    (199, "Innsbruck", "AT", False, False, True),
    (237, "Glasgow", "GB", False, False, True),
    (438, "Marburg", "DE", True, False, False),
    (467, "Gießen", "DE", True, False, False),
    (476, "Cardiff", "GB", True, True, False),
    (532, "Bilbao", "ES", True, True, False),
    (617, "Split", "HR", False, False, True),
    (619, "Freiburg", "DE", True, True, False),
    (658, "Göteborg", "SE", True, True, False),
)

EXPECTED_STATIONS = {
    129: 76,
    194: 47,
    195: 78,
    199: 45,
    237: 93,
    438: 43,
    467: 28,
    476: 73,
    532: 43,
    617: 60,
    619: 87,
    658: 126,
}

PROTOCOL_FILES = (
    "RESEARCH_PROTOCOL_V2_1.md",
    "COVERAGE_AND_TARGET_DEFINITION_V2_1.md",
    "FEATURE_PROTOCOL_V2_1.md",
    "MODEL_SPECIFICATION_V2_1.md",
    "DEVELOPMENT_SELECTION_PROTOCOL.md",
    "TRAINING_BUDGET_PROTOCOL.md",
    "COMPUTATION_DAG_V2_1.md",
    "CITY_SPLIT_PROTOCOL_V2.md",
    "TEMPORAL_SPLIT_PROTOCOL_V2.md",
    "SCALE_AND_LOSS_PROTOCOL.md",
    "EVALUATION_PROTOCOL_V2.md",
    "city_roles_v2.csv",
    "station_eligibility_summary.csv",
)


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_hash(rows: Iterable[Sequence[Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        fields = []
        for value in row:
            if value is None:
                fields.append("<NULL>")
            elif isinstance(value, float):
                fields.append(format(value, ".17g"))
            elif isinstance(value, datetime):
                fields.append(value.astimezone(timezone.utc).isoformat())
            else:
                fields.append(str(value))
        digest.update(("\x1f".join(fields) + "\n").encode("utf-8"))
    return digest.hexdigest()


def query_logical_hash(con, query: str, batch_size: int = 100_000) -> str:
    cursor = con.execute(query)
    digest = hashlib.sha256()
    while True:
        batch = cursor.fetchmany(batch_size)
        if not batch:
            break
        for row in batch:
            fields = []
            for value in row:
                if value is None:
                    fields.append("<NULL>")
                elif isinstance(value, float):
                    fields.append(format(value, ".17g"))
                elif isinstance(value, datetime):
                    if value.tzinfo is None:
                        fields.append(value.isoformat())
                    else:
                        fields.append(value.astimezone(timezone.utc).isoformat())
                else:
                    fields.append(str(value))
            digest.update(("\x1f".join(fields) + "\n").encode("utf-8"))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv_query(con, query: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cursor = con.execute(query)
    headers = [item[0] for item in cursor.description]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        while True:
            batch = cursor.fetchmany(100_000)
            if not batch:
                break
            writer.writerows(batch)


def copy_parquet(con, query: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY ({query}) TO '{sql_path(path)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )


def primary_mask_rule(
    station_in_lifetime: bool,
    station_bracketed: bool,
    bracketed_city_stations: int,
    lifetime_city_stations: int,
) -> bool:
    """Pure registered Rule B; exact-hour status events are intentionally absent."""
    required = math.ceil(CITY_BRACKET_FRACTION * lifetime_city_stations)
    return bool(
        station_in_lifetime
        and station_bracketed
        and lifetime_city_stations > 0
        and bracketed_city_stations >= required
    )


def feature_schema() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for lag in range(1, 25):
        result.extend(
            [
                {
                    "column": f"hist_lag_{lag:03d}_log1p",
                    "dtype": "DOUBLE",
                    "tensor": "X_hist",
                    "channel": "value",
                    "transform": "log1p when observed; zero placeholder otherwise",
                    "fitted_statistics_required": False,
                    "information_timestamp": f"t-{lag}h",
                    "offset_hours": -lag,
                },
                {
                    "column": f"hist_lag_{lag:03d}_observed",
                    "dtype": "BOOLEAN",
                    "tensor": "X_hist",
                    "channel": "observation_indicator",
                    "transform": "none",
                    "fitted_statistics_required": False,
                    "information_timestamp": f"t-{lag}h",
                    "offset_hours": -lag,
                },
            ]
        )
    result.extend(
        [
            {
                "column": "week_lag_168_log1p",
                "dtype": "DOUBLE",
                "tensor": "X_week",
                "channel": "value",
                "transform": "log1p when observed; zero placeholder otherwise",
                "fitted_statistics_required": False,
                "information_timestamp": "t-168h",
                "offset_hours": -168,
            },
            {
                "column": "week_lag_168_observed",
                "dtype": "BOOLEAN",
                "tensor": "X_week",
                "channel": "observation_indicator",
                "transform": "none",
                "fitted_statistics_required": False,
                "information_timestamp": "t-168h",
                "offset_hours": -168,
            },
            {
                "column": "static_bike_racks_log1p",
                "dtype": "DOUBLE",
                "tensor": "X_static",
                "channel": "rack_value",
                "transform": "log1p when bike_racks > 0; zero otherwise",
                "fitted_statistics_required": False,
                "information_timestamp": "static/pre-forecast metadata",
                "offset_hours": None,
            },
            {
                "column": "static_bike_racks_valid",
                "dtype": "BOOLEAN",
                "tensor": "X_static",
                "channel": "rack_validity",
                "transform": "bike_racks > 0",
                "fitted_statistics_required": False,
                "information_timestamp": "static/pre-forecast metadata",
                "offset_hours": None,
            },
            {
                "column": "calendar_local_hour_sin",
                "dtype": "DOUBLE",
                "tensor": "X_calendar",
                "channel": "local_hour_sine",
                "transform": "sin(2*pi*local_hour/24)",
                "fitted_statistics_required": False,
                "information_timestamp": "known at t",
                "offset_hours": 0,
            },
            {
                "column": "calendar_local_hour_cos",
                "dtype": "DOUBLE",
                "tensor": "X_calendar",
                "channel": "local_hour_cosine",
                "transform": "cos(2*pi*local_hour/24)",
                "fitted_statistics_required": False,
                "information_timestamp": "known at t",
                "offset_hours": 0,
            },
            {
                "column": "calendar_local_weekday_sin",
                "dtype": "DOUBLE",
                "tensor": "X_calendar",
                "channel": "local_weekday_sine",
                "transform": "sin(2*pi*Monday-based-weekday/7)",
                "fitted_statistics_required": False,
                "information_timestamp": "known at t",
                "offset_hours": 0,
            },
            {
                "column": "calendar_local_weekday_cos",
                "dtype": "DOUBLE",
                "tensor": "X_calendar",
                "channel": "local_weekday_cosine",
                "transform": "cos(2*pi*Monday-based-weekday/7)",
                "fitted_statistics_required": False,
                "information_timestamp": "known at t",
                "offset_hours": 0,
            },
            {
                "column": "calendar_weekend",
                "dtype": "BOOLEAN",
                "tensor": "X_calendar",
                "channel": "weekend",
                "transform": "Monday-based weekday >= 5",
                "fitted_statistics_required": False,
                "information_timestamp": "known at t",
                "offset_hours": 0,
            },
            {
                "column": "calendar_utc_offset_div_12",
                "dtype": "DOUBLE",
                "tensor": "X_calendar",
                "channel": "utc_offset_scaled",
                "transform": "UTC offset hours / 12",
                "fitted_statistics_required": False,
                "information_timestamp": "known at t",
                "offset_hours": 0,
            },
        ]
    )
    return result


FEATURE_SCHEMA = feature_schema()
FEATURE_COLUMNS = [item["column"] for item in FEATURE_SCHEMA]


def lag_select_sql() -> str:
    expressions: list[str] = []
    for lag in range(1, 25):
        target = f"lag(target_12h, {lag}) OVER lag_window"
        observed = f"lag(coverage_observed_12h, {lag}) OVER lag_window"
        expressions.extend(
            [
                f"CASE WHEN {observed} IS TRUE THEN ln(1.0 + {target}) "
                f"ELSE 0.0 END::DOUBLE AS hist_lag_{lag:03d}_log1p",
                f"coalesce({observed}, FALSE)::BOOLEAN "
                f"AS hist_lag_{lag:03d}_observed",
            ]
        )
    target = "lag(target_12h, 168) OVER lag_window"
    observed = "lag(coverage_observed_12h, 168) OVER lag_window"
    expressions.extend(
        [
            f"CASE WHEN {observed} IS TRUE THEN ln(1.0 + {target}) "
            "ELSE 0.0 END::DOUBLE AS week_lag_168_log1p",
            f"coalesce({observed}, FALSE)::BOOLEAN AS week_lag_168_observed",
        ]
    )
    return ",\n            ".join(expressions)


class BuildLog:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, message: str) -> None:
        print(message, flush=True)
        self.lines.append(message)

    def write(self, path: Path, build_timestamp: str) -> None:
        path.write_text(
            f"build_timestamp_utc={build_timestamp}\n"
            + "\n".join(self.lines)
            + "\n",
            encoding="utf-8",
        )


def assert_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{message}: expected {expected!r}, got {actual!r}")


def prepare_output(output: Path, overwrite: bool) -> None:
    resolved = output.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise SystemExit("Output directory must be inside the repository.") from exc
    if resolved == ROOT.resolve():
        raise SystemExit("Refusing to use the repository root as output.")
    if output.exists():
        if not overwrite:
            raise SystemExit(f"Output exists: {output}. Pass --overwrite to rebuild.")
        shutil.rmtree(output)
    for name in (
        "development",
        "final_adaptation",
        "final_features",
        "final_labels",
        "station_manifests",
        "manifests",
        "audits",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)


def rows_as_dicts(cursor) -> list[dict[str, Any]]:
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def markdown_table(rows: list[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    prepare_output(output, args.overwrite)
    log = BuildLog()
    build_timestamp = datetime.now(timezone.utc).isoformat()

    source_base = ROOT / "feasibility_test" / "full_audit" / "data"
    status_base = (
        ROOT
        / "feasibility_test"
        / "full_audit"
        / "coverage"
        / "data"
        / "station_status"
    )
    cities_glob = source_base / "cities" / "*.parquet"
    stations_glob = source_base / "stations" / "*.parquet"
    trips_glob = source_base / "trips" / "*.parquet"
    status_glob = status_base / "*.parquet"
    for required in (cities_glob.parent, stations_glob.parent, trips_glob.parent, status_base):
        if not required.exists():
            raise SystemExit(f"Required local input is missing: {required}")

    work_key = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    work_dir = ROOT / "tmp" / "protocol_build"
    work_dir.mkdir(parents=True, exist_ok=True)
    work_db = work_dir / f"protocol_{work_key}.duckdb"
    if work_db.exists():
        work_db.unlink()
    con = duckdb.connect(str(work_db))
    con.execute(f"SET threads={max(1, args.threads)}")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute("SET TimeZone='UTC'")

    try:
        log.add("[1/12] Registering frozen roles, boundaries, and source views")
        roles_values = ",\n".join(
            "(" + ",".join(
                [
                    str(city_id),
                    "'" + city.replace("'", "''") + "'",
                    "'" + country + "'",
                    str(source).upper(),
                    str(pseudo).upper(),
                    str(final).upper(),
                ]
            ) + ")"
            for city_id, city, country, source, pseudo, final in CITY_ROLES
        )
        con.execute(
            f"""
            CREATE TABLE roles(
                city_id BIGINT,
                city VARCHAR,
                country VARCHAR,
                development_source BOOLEAN,
                pseudo_target BOOLEAN,
                final_target BOOLEAN
            );
            INSERT INTO roles VALUES {roles_values};
            CREATE VIEW cities_raw AS
                SELECT * FROM read_parquet('{sql_path(cities_glob)}');
            CREATE VIEW stations_raw AS
                SELECT * FROM read_parquet('{sql_path(stations_glob)}');
            CREATE VIEW trips_raw AS
                SELECT * FROM read_parquet('{sql_path(trips_glob)}');
            CREATE VIEW status_raw AS
                SELECT * FROM read_parquet('{sql_path(status_glob)}');
            """
        )
        role_count = con.sql("SELECT count(*) FROM roles").fetchone()[0]
        assert_equal(role_count, 12, "Frozen city count differs")

        log.add("[2/12] Reconstructing the pre-HF q_s roster independently")
        con.execute(
            f"""
            CREATE TABLE candidate_stations AS
            SELECT
                s.id::BIGINT AS station_id,
                s.city_id::BIGINT AS city_id,
                s.lat::DOUBLE AS latitude,
                s.lon::DOUBLE AS longitude,
                s.bike_racks::BIGINT AS bike_racks
            FROM stations_raw s
            JOIN roles r ON s.city_id = r.city_id
            WHERE isfinite(s.lat) AND isfinite(s.lon);

            CREATE TABLE status_pretest AS
            SELECT
                st.station_id::BIGINT AS station_id,
                to_timestamp(st.time) AS status_utc
            FROM status_raw st
            JOIN candidate_stations c ON st.station_id = c.station_id
            WHERE st.time IS NOT NULL
              AND isfinite(st.time)
              AND st.time < epoch(TIMESTAMPTZ '{HF}')
            ORDER BY station_id, status_utc;

            CREATE TABLE pretest_lifetimes AS
            SELECT station_id, min(status_utc) AS first_status_utc,
                   max(status_utc) AS last_status_utc
            FROM status_pretest
            GROUP BY station_id;

            CREATE TABLE pretest_grid AS
            SELECT
                c.station_id,
                c.city_id,
                h.timestamp_utc,
                h.timestamp_utc + INTERVAL 1 HOUR AS hour_end
            FROM candidate_stations c
            CROSS JOIN generate_series(
                TIMESTAMPTZ '{H0}',
                TIMESTAMPTZ '{HF}' - INTERVAL 1 HOUR,
                INTERVAL 1 HOUR
            ) h(timestamp_utc);

            CREATE TABLE pretest_previous AS
            SELECT g.*, p.status_utc AS previous_status_utc
            FROM pretest_grid g
            ASOF LEFT JOIN status_pretest p
              ON g.station_id = p.station_id
             AND g.timestamp_utc >= p.status_utc;

            CREATE TABLE pretest_brackets AS
            SELECT
                gp.station_id,
                gp.city_id,
                gp.timestamp_utc,
                coalesce(gp.timestamp_utc >= l.first_status_utc
                  AND gp.timestamp_utc <= l.last_status_utc, FALSE)
                    AS station_in_lifetime,
                coalesce(date_diff('minute', gp.previous_status_utc, gp.timestamp_utc)
                    BETWEEN 0 AND 720
                  AND date_diff('minute', gp.hour_end, n.status_utc)
                    BETWEEN 0 AND 720, FALSE) AS station_bracketed_12h
            FROM pretest_previous gp
            ASOF LEFT JOIN status_pretest n
              ON gp.station_id = n.station_id
             AND gp.hour_end <= n.status_utc
            LEFT JOIN pretest_lifetimes l ON gp.station_id = l.station_id;

            CREATE TABLE station_q AS
            SELECT
                c.station_id,
                c.city_id,
                coalesce(
                    count_if(b.station_in_lifetime AND b.station_bracketed_12h)::DOUBLE
                    / nullif(count_if(b.station_in_lifetime), 0),
                    0.0
                )::DOUBLE AS q_s,
                count_if(b.station_in_lifetime)::BIGINT AS pretest_lifetime_hours,
                count_if(b.station_in_lifetime AND b.station_bracketed_12h)::BIGINT
                    AS pretest_bracketed_hours
            FROM candidate_stations c
            LEFT JOIN pretest_brackets b ON c.station_id = b.station_id
            GROUP BY c.station_id, c.city_id;

            CREATE TABLE eligible_stations AS
            SELECT
                c.*,
                q.q_s,
                q.pretest_lifetime_hours,
                q.pretest_bracketed_hours,
                TRUE::BOOLEAN AS frozen_station_roster_indicator
            FROM candidate_stations c
            JOIN station_q q USING(station_id, city_id)
            WHERE q.q_s >= {STATION_THRESHOLD};
            """
        )
        actual_station_rows = con.sql(
            "SELECT city_id, count(*) FROM eligible_stations GROUP BY city_id ORDER BY city_id"
        ).fetchall()
        actual_stations = {int(city_id): int(count) for city_id, count in actual_station_rows}
        if actual_stations != EXPECTED_STATIONS:
            raise RuntimeError(
                "STOP: reconstructed station roster differs from frozen V2.1 roster. "
                f"Expected {EXPECTED_STATIONS}; got {actual_stations}."
            )
        assert_equal(sum(actual_stations.values()), 799, "Frozen roster total differs")
        log.add("      PASS: exact registered per-city counts and 799-station total")

        log.add("[3/12] Building valid trip departures and exclusion diagnostics")
        con.execute(
            f"""
            CREATE TABLE trip_scope AS
            SELECT
                t.city_id::BIGINT AS trip_city_id,
                date_trunc('hour', to_timestamp(t.time_start)) AS timestamp_utc,
                t.station_id_start,
                CASE
                    WHEN t.station_id_start IS NOT NULL
                     AND isfinite(t.station_id_start)
                     AND t.station_id_start = trunc(t.station_id_start)
                    THEN t.station_id_start::BIGINT
                    ELSE NULL
                END AS start_station_id,
                CASE
                    WHEN t.station_id_start IS NULL THEN 'null_station_reference'
                    WHEN NOT isfinite(t.station_id_start)
                      OR t.station_id_start != trunc(t.station_id_start)
                        THEN 'malformed_station_reference'
                    ELSE 'integer_station_reference'
                END AS reference_form
            FROM trips_raw t
            JOIN roles r ON t.city_id = r.city_id
            WHERE t.time_start IS NOT NULL
              AND isfinite(t.time_start)
              AND t.time_start >= epoch(TIMESTAMPTZ '{H0}')
              AND t.time_start < epoch(TIMESTAMPTZ '{HE}');

            CREATE TABLE trip_classification AS
            SELECT
                t.*,
                s.city_id AS station_city_id,
                e.station_id IS NOT NULL AS station_is_eligible,
                CASE
                    WHEN t.reference_form = 'null_station_reference'
                        THEN 'null_station_reference'
                    WHEN t.reference_form = 'malformed_station_reference'
                        THEN 'malformed_station_reference'
                    WHEN s.id IS NULL THEN 'orphan_station_reference'
                    WHEN s.city_id != t.trip_city_id THEN 'cross_city_station_reference'
                    WHEN e.station_id IS NULL THEN 'station_below_q_threshold'
                    ELSE 'accepted_eligible_departure'
                END AS disposition
            FROM trip_scope t
            LEFT JOIN stations_raw s ON t.start_station_id = s.id
            LEFT JOIN eligible_stations e
              ON t.start_station_id = e.station_id
             AND t.trip_city_id = e.city_id;

            CREATE TABLE hourly_departures AS
            SELECT
                trip_city_id AS city_id,
                start_station_id AS station_id,
                timestamp_utc,
                count(*)::BIGINT AS raw_departure_count
            FROM trip_classification
            WHERE disposition = 'accepted_eligible_departure'
            GROUP BY city_id, station_id, timestamp_utc;
            """
        )
        timestamp_diagnostics = rows_as_dicts(
            con.execute(
                f"""
                SELECT
                    count(*)::BIGINT AS protocol_city_trip_rows_all_time,
                    count_if(time_start IS NULL OR NOT isfinite(time_start)
                        OR time_start < 0 OR time_start > 4102444800)::BIGINT
                        AS invalid_start_timestamp_rows,
                    count_if(time_start IS NOT NULL AND isfinite(time_start)
                        AND (time_start < epoch(TIMESTAMPTZ '{H0}')
                          OR time_start >= epoch(TIMESTAMPTZ '{HE}')))::BIGINT
                        AS valid_timestamp_outside_analysis_rows
                FROM trips_raw t
                JOIN roles r ON t.city_id = r.city_id
                """
            )
        )[0]
        trip_exclusions = rows_as_dicts(
            con.execute(
                """
                SELECT disposition, count(*)::BIGINT AS trip_rows
                FROM trip_classification
                GROUP BY disposition ORDER BY disposition
                """
            )
        )

        log.add("[4/12] Reproducing Rule B 12h coverage and registered sensitivities")
        con.execute(
            f"""
            CREATE TABLE status_selected AS
            SELECT
                st.station_id::BIGINT AS station_id,
                to_timestamp(st.time) AS status_utc,
                st.maintenance::BOOLEAN AS maintenance
            FROM status_raw st
            JOIN eligible_stations e ON st.station_id = e.station_id
            WHERE st.time IS NOT NULL AND isfinite(st.time)
            ORDER BY station_id, status_utc;

            CREATE TABLE full_lifetimes AS
            SELECT station_id, min(status_utc) AS first_status_utc,
                   max(status_utc) AS last_status_utc
            FROM status_selected GROUP BY station_id;

            CREATE TABLE station_hour_grid AS
            SELECT
                e.city_id,
                e.station_id,
                h.timestamp_utc,
                h.timestamp_utc + INTERVAL 1 HOUR AS hour_end
            FROM eligible_stations e
            CROSS JOIN generate_series(
                TIMESTAMPTZ '{H0}',
                TIMESTAMPTZ '{HE}' - INTERVAL 1 HOUR,
                INTERVAL 1 HOUR
            ) h(timestamp_utc);

            CREATE TABLE grid_previous AS
            SELECT
                g.*,
                p.status_utc AS previous_status_utc,
                p.maintenance AS previous_maintenance
            FROM station_hour_grid g
            ASOF LEFT JOIN status_selected p
              ON g.station_id = p.station_id
             AND g.timestamp_utc >= p.status_utc;

            CREATE TABLE station_hour_brackets AS
            SELECT
                gp.city_id,
                gp.station_id,
                gp.timestamp_utc,
                gp.hour_end,
                gp.previous_status_utc,
                n.status_utc AS next_status_utc,
                gp.previous_maintenance,
                l.first_status_utc,
                l.last_status_utc,
                coalesce(gp.timestamp_utc >= l.first_status_utc
                  AND gp.timestamp_utc <= l.last_status_utc, FALSE)
                    AS station_in_status_lifetime,
                coalesce(date_diff('minute', gp.previous_status_utc, gp.timestamp_utc)
                    BETWEEN 0 AND 720
                  AND date_diff('minute', gp.hour_end, n.status_utc)
                    BETWEEN 0 AND 720, FALSE) AS station_status_bracketed_12h,
                coalesce(date_diff('minute', gp.previous_status_utc, gp.timestamp_utc)
                    BETWEEN 0 AND 1440
                  AND date_diff('minute', gp.hour_end, n.status_utc)
                    BETWEEN 0 AND 1440, FALSE) AS station_status_bracketed_24h
            FROM grid_previous gp
            ASOF LEFT JOIN status_selected n
              ON gp.station_id = n.station_id
             AND gp.hour_end <= n.status_utc
            LEFT JOIN full_lifetimes l ON gp.station_id = l.station_id;

            CREATE TABLE city_hour_coverage AS
            SELECT
                city_id,
                timestamp_utc,
                count_if(station_in_status_lifetime)::BIGINT
                    AS eligible_lifetime_stations,
                count_if(station_in_status_lifetime
                    AND station_status_bracketed_12h)::BIGINT
                    AS bracketed_stations_12h,
                count_if(station_in_status_lifetime
                    AND station_status_bracketed_24h)::BIGINT
                    AS bracketed_stations_24h,
                ceil(count_if(station_in_status_lifetime) * {CITY_BRACKET_FRACTION})::BIGINT
                    AS required_bracketed_stations,
                count_if(station_in_status_lifetime) > 0
                  AND count_if(station_in_status_lifetime
                    AND station_status_bracketed_12h)
                    >= ceil(count_if(station_in_status_lifetime) * {CITY_BRACKET_FRACTION})
                    AS city_observable_12h,
                count_if(station_in_status_lifetime) > 0
                  AND count_if(station_in_status_lifetime
                    AND station_status_bracketed_24h)
                    >= ceil(count_if(station_in_status_lifetime) * {CITY_BRACKET_FRACTION})
                    AS city_observable_24h
            FROM station_hour_brackets
            GROUP BY city_id, timestamp_utc;
            """
        )

        log.add("[5/12] Constructing nullable targets, local time, roles, and budgets")
        con.execute(
            f"""
            CREATE TABLE panel_core AS
            WITH masked AS (
                SELECT
                    b.*,
                    c.eligible_lifetime_stations,
                    c.bracketed_stations_12h,
                    c.bracketed_stations_24h,
                    c.required_bracketed_stations,
                    c.city_observable_12h,
                    c.city_observable_24h,
                    coalesce(d.raw_departure_count, 0)::BIGINT AS raw_departure_count,
                    coalesce((b.station_in_status_lifetime
                     AND b.station_status_bracketed_12h
                     AND c.city_observable_12h), FALSE)::BOOLEAN AS coverage_observed_12h,
                    coalesce((b.station_in_status_lifetime
                     AND b.station_status_bracketed_24h
                     AND c.city_observable_24h), FALSE)::BOOLEAN AS coverage_observed_24h
                FROM station_hour_brackets b
                JOIN city_hour_coverage c USING(city_id, timestamp_utc)
                LEFT JOIN hourly_departures d USING(city_id, station_id, timestamp_utc)
            ), targets AS (
                SELECT
                    m.*,
                    CASE WHEN coverage_observed_12h
                         THEN raw_departure_count ELSE NULL END::BIGINT AS target_12h,
                    CASE WHEN coverage_observed_24h
                         THEN raw_departure_count ELSE NULL END::BIGINT AS target_24h,
                    coalesce((coverage_observed_12h
                     AND previous_maintenance IS FALSE), FALSE)::BOOLEAN
                        AS coverage_observed_12h_no_maintenance,
                    CASE WHEN coverage_observed_12h
                               AND previous_maintenance IS FALSE
                         THEN raw_departure_count ELSE NULL END::BIGINT
                        AS target_12h_no_maintenance,
                    CASE
                        WHEN coverage_observed_12h AND raw_departure_count > 0
                            THEN 'coverage_qualified_positive'
                        WHEN coverage_observed_12h AND raw_departure_count = 0
                            THEN 'coverage_qualified_zero_demand'
                        WHEN NOT coverage_observed_12h AND raw_departure_count > 0
                            THEN 'unobserved_positive'
                        ELSE 'unknown_no_trip'
                    END AS coverage_state_12h,
                    CASE
                        WHEN previous_status_utc IS NULL THEN 'no_prior_status'
                        WHEN previous_maintenance IS TRUE THEN 'maintenance'
                        ELSE 'not_maintenance'
                    END AS maintenance_state
                FROM masked m
            ), enriched AS (
                SELECT
                    t.*,
                    e.q_s,
                    e.latitude,
                    e.longitude,
                    e.bike_racks,
                    e.frozen_station_roster_indicator,
                    r.city,
                    r.country,
                    r.development_source,
                    r.pseudo_target,
                    r.final_target,
                    cr.timezone,
                    timezone(cr.timezone, t.timestamp_utc) AS local_timestamp,
                    date_part('hour', timezone(cr.timezone, t.timestamp_utc))::UTINYINT
                        AS local_hour,
                    (date_part('isodow', timezone(cr.timezone, t.timestamp_utc)) - 1)::UTINYINT
                        AS weekday,
                    epoch(
                        timezone(cr.timezone, t.timestamp_utc)
                        - timezone('UTC', t.timestamp_utc)
                    ) / 3600.0 AS utc_offset_hours
                FROM targets t
                JOIN eligible_stations e USING(city_id, station_id)
                JOIN roles r USING(city_id)
                JOIN cities_raw cr ON r.city_id = cr.id
            )
            SELECT
                enriched.*,
                CASE
                    WHEN timestamp_utc >= TIMESTAMPTZ '{H0}'
                     AND timestamp_utc < TIMESTAMPTZ '{HD}' THEN 'H0_HD'
                    WHEN timestamp_utc >= TIMESTAMPTZ '{HD}'
                     AND timestamp_utc < TIMESTAMPTZ '{HF}' THEN 'HD_HF'
                    WHEN timestamp_utc >= TIMESTAMPTZ '{HF}'
                     AND timestamp_utc < TIMESTAMPTZ '{HT}' THEN 'HF_HT_embargo'
                    WHEN timestamp_utc >= TIMESTAMPTZ '{HT}'
                     AND timestamp_utc < TIMESTAMPTZ '{HE}' THEN 'HT_HE_final_evaluation'
                    ELSE 'OUT_OF_PROTOCOL'
                END AS temporal_phase,
                FALSE::BOOLEAN AS budget_parameter_zero,
                (
                    (pseudo_target AND timestamp_utc >= TIMESTAMPTZ '{HD}' - INTERVAL 1 DAY
                                   AND timestamp_utc < TIMESTAMPTZ '{HD}')
                    OR
                    (final_target AND timestamp_utc >= TIMESTAMPTZ '{HF}' - INTERVAL 1 DAY
                                  AND timestamp_utc < TIMESTAMPTZ '{HF}')
                )::BOOLEAN AS budget_1_day,
                (
                    (pseudo_target AND timestamp_utc >= TIMESTAMPTZ '{HD}' - INTERVAL 7 DAY
                                   AND timestamp_utc < TIMESTAMPTZ '{HD}')
                    OR
                    (final_target AND timestamp_utc >= TIMESTAMPTZ '{HF}' - INTERVAL 7 DAY
                                  AND timestamp_utc < TIMESTAMPTZ '{HF}')
                )::BOOLEAN AS budget_7_days,
                (
                    (pseudo_target AND timestamp_utc >= TIMESTAMPTZ '{HD}' - INTERVAL 30 DAY
                                   AND timestamp_utc < TIMESTAMPTZ '{HD}')
                    OR
                    (final_target AND timestamp_utc >= TIMESTAMPTZ '{HF}' - INTERVAL 30 DAY
                                  AND timestamp_utc < TIMESTAMPTZ '{HF}')
                )::BOOLEAN AS budget_30_days,
                (
                    (pseudo_target AND timestamp_utc >= TIMESTAMPTZ '{H0}'
                                   AND timestamp_utc < TIMESTAMPTZ '{HD}')
                    OR
                    (final_target AND timestamp_utc >= TIMESTAMPTZ '{H0}'
                                  AND timestamp_utc < TIMESTAMPTZ '{HF}')
                )::BOOLEAN AS budget_full,
                CASE WHEN bike_racks > 0 THEN ln(1.0 + bike_racks)
                     ELSE 0.0 END::DOUBLE AS static_bike_racks_log1p,
                (bike_racks > 0)::BOOLEAN AS static_bike_racks_valid,
                sin(2.0 * pi() * local_hour / 24.0)::DOUBLE
                    AS calendar_local_hour_sin,
                cos(2.0 * pi() * local_hour / 24.0)::DOUBLE
                    AS calendar_local_hour_cos,
                sin(2.0 * pi() * weekday / 7.0)::DOUBLE
                    AS calendar_local_weekday_sin,
                cos(2.0 * pi() * weekday / 7.0)::DOUBLE
                    AS calendar_local_weekday_cos,
                (weekday >= 5)::BOOLEAN AS calendar_weekend,
                (utc_offset_hours / 12.0)::DOUBLE AS calendar_utc_offset_div_12
            FROM enriched;
            """
        )

        log.add("[6/12] Running hard-stop panel, coverage, split, and reconciliation checks")
        validation: list[dict[str, Any]] = []

        def check(name: str, query: str, expected: Any = 0, warning: bool = False) -> Any:
            actual = con.sql(query).fetchone()[0]
            passed = actual == expected
            validation.append(
                {
                    "name": name,
                    "status": "PASS" if passed else ("WARNING" if warning else "FAIL"),
                    "actual": actual,
                    "expected": expected,
                }
            )
            if not passed and not warning:
                raise RuntimeError(
                    f"STOP: validation failed: {name}; expected {expected}, got {actual}"
                )
            return actual

        check(
            "unique_primary_panel_keys",
            "SELECT count(*) - count(DISTINCT (city_id, station_id, timestamp_utc)) FROM panel_core",
        )
        check(
            "only_frozen_q_roster",
            f"SELECT count(*) FROM panel_core WHERE q_s < {STATION_THRESHOLD} "
            "OR NOT frozen_station_roster_indicator",
        )
        check(
            "target_nonnull_iff_primary_mask",
            "SELECT count(*) FROM panel_core WHERE (target_12h IS NOT NULL) "
            "IS DISTINCT FROM coverage_observed_12h",
        )
        check(
            "coverage_masks_are_explicit_booleans",
            "SELECT count(*) FROM panel_core WHERE coverage_observed_12h IS NULL "
            "OR coverage_observed_24h IS NULL "
            "OR coverage_observed_12h_no_maintenance IS NULL",
        )
        check(
            "target_nonnegative",
            "SELECT count(*) FROM panel_core WHERE target_12h < 0 OR target_24h < 0",
        )
        check(
            "uncovered_positive_target_is_null",
            "SELECT count(*) FROM panel_core WHERE raw_departure_count > 0 "
            "AND NOT coverage_observed_12h AND target_12h IS NOT NULL",
        )
        check(
            "primary_mask_reproduces_rule_b",
            "SELECT count(*) FROM panel_core WHERE coverage_observed_12h IS DISTINCT FROM "
            "(station_in_status_lifetime AND station_status_bracketed_12h "
            "AND city_observable_12h)",
        )
        check(
            "24h_mask_independently_reproducible",
            "SELECT count(*) FROM panel_core WHERE coverage_observed_24h IS DISTINCT FROM "
            "(station_in_status_lifetime AND station_status_bracketed_24h "
            "AND city_observable_24h)",
        )
        check(
            "no_maintenance_symmetric_mask",
            "SELECT count(*) FROM panel_core WHERE "
            "coverage_observed_12h_no_maintenance IS DISTINCT FROM "
            "(coverage_observed_12h AND previous_maintenance IS FALSE) "
            "OR ((target_12h_no_maintenance IS NOT NULL) IS DISTINCT FROM "
            "coverage_observed_12h_no_maintenance)",
        )
        check(
            "registered_temporal_phases_only",
            "SELECT count(*) FROM panel_core WHERE temporal_phase='OUT_OF_PROTOCOL'",
        )
        check(
            "budget_nesting",
            "SELECT count(*) FROM panel_core WHERE budget_1_day AND NOT budget_7_days "
            "OR budget_7_days AND NOT budget_30_days "
            "OR budget_30_days AND NOT budget_full",
        )
        check(
            "parameter_zero_contains_no_labels",
            "SELECT count(*) FROM panel_core WHERE budget_parameter_zero",
        )
        accepted_departures = con.sql(
            "SELECT count(*) FROM trip_classification "
            "WHERE disposition='accepted_eligible_departure'"
        ).fetchone()[0]
        panel_departures = con.sql(
            "SELECT sum(raw_departure_count) FROM panel_core"
        ).fetchone()[0]
        if accepted_departures != panel_departures:
            raise RuntimeError(
                "STOP: raw panel departures do not reconcile with accepted trips: "
                f"{panel_departures} vs {accepted_departures}."
            )
        validation.append(
            {
                "name": "raw_departure_reconciliation",
                "status": "PASS",
                "actual": panel_departures,
                "expected": accepted_departures,
            }
        )
        if not primary_mask_rule(True, True, 5, 10):
            raise RuntimeError("STOP: bracket-only synthetic nighttime fixture failed.")
        validation.append(
            {
                "name": "nighttime_bracket_without_same_hour_event_is_observed",
                "status": "PASS",
                "actual": True,
                "expected": True,
                "note": "Rule B has no exact-hour event input.",
            }
        )
        final_gate_rows = rows_as_dicts(
            con.execute(
                f"""
                SELECT
                    r.city_id,
                    r.city,
                    avg(c.city_observable_12h::INTEGER)::DOUBLE
                        AS final_city_hour_coverage_fraction
                FROM city_hour_coverage c
                JOIN roles r USING(city_id)
                WHERE r.final_target
                  AND c.timestamp_utc >= TIMESTAMPTZ '{HT}'
                  AND c.timestamp_utc < TIMESTAMPTZ '{HE}'
                GROUP BY r.city_id, r.city ORDER BY r.city_id
                """
            )
        )
        failed_gate = [row for row in final_gate_rows if row["final_city_hour_coverage_fraction"] < 0.90]
        if failed_gate:
            raise RuntimeError(f"STOP: sealed final target failed 90% coverage gate: {failed_gate}")
        validation.append(
            {
                "name": "all_final_targets_pass_90_percent_city_hour_gate",
                "status": "PASS",
                "actual": len(final_gate_rows),
                "expected": 4,
            }
        )

        # The repository records Europe/Berlin for Bilbao. Europe/Madrid has the
        # same offsets and local calendar over this interval, so this metadata
        # warning does not change a registered feature value.
        bilbao_tz_difference = con.sql(
            f"""
            SELECT count(*) FROM generate_series(
                TIMESTAMPTZ '{H0}', TIMESTAMPTZ '{HE}' - INTERVAL 1 HOUR,
                INTERVAL 1 HOUR
            ) h(t)
            WHERE timezone('Europe/Berlin', t) != timezone('Europe/Madrid', t)
            """
        ).fetchone()[0]
        if bilbao_tz_difference != 0:
            raise RuntimeError(
                "STOP: unresolved Bilbao timezone metadata changes calendar features."
            )
        validation.append(
            {
                "name": "bilbao_timezone_alias_feature_equivalence",
                "status": "WARNING",
                "actual": bilbao_tz_difference,
                "expected": 0,
                "note": "Source says Europe/Berlin; Europe/Madrid is semantically canonical but feature-equivalent during the frozen interval.",
            }
        )

        log.add("[7/12] Writing split-safe feature and sealed-label Parquet artifacts")
        lag_sql = lag_select_sql()
        common_lagged = f"""
            SELECT p.*, {lag_sql}
            FROM panel_core p
            WINDOW lag_window AS (
                PARTITION BY city_id, station_id ORDER BY timestamp_utc
            )
        """
        development_query = f"""
            SELECT * FROM ({common_lagged}) q
            WHERE development_source
              AND timestamp_utc >= TIMESTAMPTZ '{H0}'
              AND timestamp_utc < TIMESTAMPTZ '{HF}'
            ORDER BY city_id, station_id, timestamp_utc
        """
        final_adaptation_query = f"""
            SELECT * FROM ({common_lagged}) q
            WHERE final_target
              AND timestamp_utc >= TIMESTAMPTZ '{H0}'
              AND timestamp_utc < TIMESTAMPTZ '{HF}'
            ORDER BY city_id, station_id, timestamp_utc
        """
        safe_final_columns = [
            "city_id",
            "station_id",
            "timestamp_utc",
            "local_timestamp",
            "local_hour",
            "weekday",
            "utc_offset_hours",
            "city",
            "country",
            "timezone",
            "q_s",
            "latitude",
            "longitude",
            "bike_racks",
            "frozen_station_roster_indicator",
            "final_target",
            "temporal_phase",
            "coverage_observed_12h",
            *FEATURE_COLUMNS,
        ]
        final_feature_query = f"""
            SELECT {', '.join(safe_final_columns)}
            FROM ({common_lagged}) q
            WHERE final_target
              AND timestamp_utc >= TIMESTAMPTZ '{HT}'
              AND timestamp_utc < TIMESTAMPTZ '{HE}'
            ORDER BY city_id, station_id, timestamp_utc
        """
        final_label_columns = [
            "city_id",
            "station_id",
            "timestamp_utc",
            "raw_departure_count",
            "coverage_observed_12h",
            "target_12h",
            "coverage_state_12h",
            "coverage_observed_24h",
            "target_24h",
            "previous_maintenance AS maintenance_at_hour_start",
            "maintenance_state",
            "coverage_observed_12h_no_maintenance",
            "target_12h_no_maintenance",
        ]
        final_label_query = f"""
            SELECT {', '.join(final_label_columns)}
            FROM panel_core
            WHERE final_target
              AND timestamp_utc >= TIMESTAMPTZ '{HT}'
              AND timestamp_utc < TIMESTAMPTZ '{HE}'
            ORDER BY city_id, station_id, timestamp_utc
        """
        final_key_query = f"""
            SELECT city_id, station_id, timestamp_utc
            FROM panel_core
            WHERE final_target AND coverage_observed_12h
              AND timestamp_utc >= TIMESTAMPTZ '{HT}'
              AND timestamp_utc < TIMESTAMPTZ '{HE}'
            ORDER BY city_id, station_id, timestamp_utc
        """
        development_path = output / "development" / "development_panel.parquet"
        final_adaptation_path = (
            output / "final_adaptation" / "final_adaptation_panel.parquet"
        )
        final_features_path = (
            output / "final_features" / "final_evaluation_features.parquet"
        )
        final_keys_path = (
            output / "final_features" / "final_prediction_keys.parquet"
        )
        final_labels_path = (
            output / "final_labels" / "SEALED_final_evaluation_labels.parquet"
        )
        copy_parquet(con, development_query, development_path)
        copy_parquet(con, final_adaptation_query, final_adaptation_path)
        copy_parquet(con, final_feature_query, final_features_path)
        copy_parquet(con, final_key_query, final_keys_path)
        copy_parquet(con, final_label_query, final_labels_path)

        # Hard physical/logical seal assertions based on output schemas.
        dev_columns = {
            row[0]
            for row in con.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{sql_path(development_path)}')"
            ).fetchall()
        }
        final_feature_columns = {
            row[0]
            for row in con.sql(
                f"DESCRIBE SELECT * FROM read_parquet('{sql_path(final_features_path)}')"
            ).fetchall()
        }
        forbidden_final_feature_columns = {
            "raw_departure_count",
            "target_12h",
            "target_24h",
            "target_12h_no_maintenance",
            "coverage_state_12h",
            "next_status_utc",
            "previous_maintenance",
        }
        leaked = sorted(final_feature_columns & forbidden_final_feature_columns)
        if leaked:
            raise RuntimeError(f"STOP: final feature artifact contains label fields: {leaked}")
        dev_final_rows = con.sql(
            f"SELECT count(*) FROM read_parquet('{sql_path(development_path)}') "
            "WHERE final_target"
        ).fetchone()[0]
        assert_equal(dev_final_rows, 0, "Final-target rows leaked into development artifact")
        validation.extend(
            [
                {
                    "name": "no_final_targets_in_development_artifact",
                    "status": "PASS",
                    "actual": dev_final_rows,
                    "expected": 0,
                },
                {
                    "name": "no_labels_or_future_status_in_final_features",
                    "status": "PASS",
                    "actual": leaked,
                    "expected": [],
                },
                {
                    "name": "development_contains_registered_feature_columns",
                    "status": "PASS" if set(FEATURE_COLUMNS) <= dev_columns else "FAIL",
                    "actual": len(set(FEATURE_COLUMNS) & dev_columns),
                    "expected": len(FEATURE_COLUMNS),
                },
            ]
        )
        if not set(FEATURE_COLUMNS) <= dev_columns:
            raise RuntimeError("STOP: development output is missing registered features.")

        log.add("[8/12] Creating graph-ready station-order manifests")
        station_manifest_summary: list[dict[str, Any]] = []
        for city_id, city, _country, _source, _pseudo, _final in CITY_ROLES:
            slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in city)
            station_path = (
                output
                / "station_manifests"
                / f"city_{city_id}_{slug}_stations.parquet"
            )
            query = f"""
                SELECT
                    row_number() OVER (ORDER BY station_id) - 1 AS node_index,
                    city_id,
                    station_id,
                    latitude,
                    longitude,
                    bike_racks,
                    CASE WHEN bike_racks > 0 THEN ln(1.0 + bike_racks)
                         ELSE 0.0 END::DOUBLE AS static_bike_racks_log1p,
                    (bike_racks > 0)::BOOLEAN AS static_bike_racks_valid,
                    q_s,
                    frozen_station_roster_indicator
                FROM eligible_stations
                WHERE city_id = {city_id}
                ORDER BY station_id
            """
            copy_parquet(con, query, station_path)
            roster_rows = con.sql(
                f"SELECT station_id FROM eligible_stations WHERE city_id={city_id} "
                "ORDER BY station_id"
            ).fetchall()
            coordinate_rows = con.sql(
                f"SELECT station_id, latitude, longitude FROM eligible_stations "
                f"WHERE city_id={city_id} ORDER BY station_id"
            ).fetchall()
            station_manifest_summary.append(
                {
                    "city_id": city_id,
                    "city": city,
                    "station_count": len(roster_rows),
                    "roster_hash_sha256": logical_hash(roster_rows),
                    "coordinate_hash_sha256": logical_hash(coordinate_rows),
                    "artifact": station_path.relative_to(output).as_posix(),
                }
            )
        write_json(
            output / "station_manifests" / "station_manifest.json",
            {
                "ordering": "station_id ascending; node_index is zero-based",
                "graph_inputs": "coordinates only; no trip/demand input",
                "cities": station_manifest_summary,
            },
        )

        log.add("[9/12] Materializing split, budget, row-count, and cohort manifests")
        city_summary_query = f"""
            SELECT
                p.city_id,
                p.city,
                max(cs.listed_stations)::BIGINT AS listed_stations,
                count(DISTINCT p.station_id)::BIGINT AS eligible_stations,
                count(DISTINCT p.timestamp_utc)::BIGINT AS analysis_hours,
                count(*)::BIGINT AS raw_station_hours,
                count_if(p.coverage_observed_12h)::BIGINT AS primary_observed_station_hours,
                count_if(p.coverage_observed_12h AND p.raw_departure_count=0)::BIGINT
                    AS coverage_qualified_zeros,
                count_if(p.coverage_observed_12h AND p.raw_departure_count>0)::BIGINT
                    AS positive_station_hours,
                sum(p.raw_departure_count)::BIGINT AS total_eligible_departures,
                sum(CASE WHEN p.coverage_observed_12h THEN p.raw_departure_count ELSE 0 END)::BIGINT
                    AS primary_observed_departures,
                count_if(NOT p.coverage_observed_12h)::BIGINT AS masked_station_hours,
                count_if(NOT p.coverage_observed_12h AND p.raw_departure_count>0)::BIGINT
                    AS positive_station_hours_excluded_by_coverage,
                sum(CASE WHEN NOT p.coverage_observed_12h THEN p.raw_departure_count ELSE 0 END)::BIGINT
                    AS departures_excluded_by_coverage,
                count_if(NOT p.coverage_observed_12h AND p.raw_departure_count=0)::BIGINT
                    AS candidate_zeros_excluded_unknown,
                count_if(p.coverage_observed_24h)::BIGINT AS observed_station_hours_24h,
                count_if(p.coverage_observed_12h_no_maintenance)::BIGINT
                    AS observed_station_hours_no_maintenance
            FROM panel_core p
            JOIN (
                SELECT c.city_id, count(*)::BIGINT AS listed_stations
                FROM candidate_stations c GROUP BY c.city_id
            ) cs USING(city_id)
            GROUP BY p.city_id, p.city
            ORDER BY p.city_id
        """
        phase_summary_query = """
            SELECT
                temporal_phase,
                count(DISTINCT timestamp_utc)::BIGINT AS hours,
                count(*)::BIGINT AS station_hours,
                count_if(coverage_observed_12h)::BIGINT AS primary_observed_station_hours,
                count_if(coverage_observed_12h AND raw_departure_count=0)::BIGINT
                    AS coverage_qualified_zeros,
                count_if(coverage_observed_12h AND raw_departure_count>0)::BIGINT
                    AS positive_station_hours,
                sum(raw_departure_count)::BIGINT AS total_eligible_departures,
                count_if(NOT coverage_observed_12h)::BIGINT AS masked_station_hours,
                count_if(coverage_observed_24h)::BIGINT AS observed_station_hours_24h,
                count_if(coverage_observed_12h_no_maintenance)::BIGINT
                    AS observed_station_hours_no_maintenance
            FROM panel_core GROUP BY temporal_phase
            ORDER BY min(timestamp_utc)
        """
        budget_query = """
            SELECT city_id, city,
                   CASE WHEN pseudo_target THEN 'pseudo_target'
                        ELSE 'final_target' END AS target_role,
                   budget,
                   CASE budget
                     WHEN 'parameter_zero' THEN 0
                     WHEN '1_day' THEN count_if(budget_1_day AND coverage_observed_12h)
                     WHEN '7_days' THEN count_if(budget_7_days AND coverage_observed_12h)
                     WHEN '30_days' THEN count_if(budget_30_days AND coverage_observed_12h)
                     WHEN 'full' THEN count_if(budget_full AND coverage_observed_12h)
                   END::BIGINT AS eligible_target_labels
            FROM panel_core
            CROSS JOIN (VALUES ('parameter_zero'),('1_day'),('7_days'),('30_days'),('full')) b(budget)
            WHERE pseudo_target OR final_target
            GROUP BY city_id, city, pseudo_target, budget
            ORDER BY city_id,
                CASE budget WHEN 'parameter_zero' THEN 0 WHEN '1_day' THEN 1
                WHEN '7_days' THEN 2 WHEN '30_days' THEN 3 ELSE 4 END
        """
        final_cohort_query = f"""
            SELECT city_id, city,
                   count_if(coverage_observed_12h)::BIGINT AS primary_evaluation_keys,
                   count_if(coverage_observed_12h AND raw_departure_count>0)::BIGINT
                       AS positive_station_hours,
                   count_if(coverage_observed_12h AND raw_departure_count=0)::BIGINT
                       AS coverage_qualified_zeros,
                   count_if(coverage_observed_24h)::BIGINT AS sensitivity_24h_keys,
                   count_if(coverage_observed_12h_no_maintenance)::BIGINT
                       AS no_maintenance_keys
            FROM panel_core
            WHERE final_target
              AND timestamp_utc >= TIMESTAMPTZ '{HT}'
              AND timestamp_utc < TIMESTAMPTZ '{HE}'
            GROUP BY city_id, city ORDER BY city_id
        """
        city_summary = rows_as_dicts(con.execute(city_summary_query))
        phase_summary = rows_as_dicts(con.execute(phase_summary_query))
        budget_counts = rows_as_dicts(con.execute(budget_query))
        final_cohort = rows_as_dicts(con.execute(final_cohort_query))
        write_csv_query(con, city_summary_query, output / "manifests" / "city_panel_summary.csv")
        write_csv_query(con, phase_summary_query, output / "manifests" / "temporal_phase_summary.csv")
        write_csv_query(con, budget_query, output / "manifests" / "budget_label_counts.csv")
        write_csv_query(con, final_cohort_query, output / "manifests" / "final_cohort_summary.csv")
        copy_parquet(con, city_summary_query, output / "manifests" / "city_panel_summary.parquet")
        copy_parquet(con, budget_query, output / "manifests" / "budget_label_counts.parquet")

        split_manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "interval_semantics": "UTC, left-closed, right-open",
            "boundaries": {"H0": H0, "HD": HD, "HF": HF, "HT": HT, "HE": HE},
            "city_roles": [
                {
                    "city_id": city_id,
                    "city": city,
                    "country": country,
                    "development_source": source,
                    "pseudo_target": pseudo,
                    "final_target": final,
                }
                for city_id, city, country, source, pseudo, final in CITY_ROLES
            ],
            "budgets": {
                "pseudo_target_cutoff": HD,
                "final_target_cutoff": HF,
                "parameter_zero": None,
                "elapsed_windows": ["1 day", "7 days", "30 days", "full from H0"],
                "no_window_extension_for_missing_labels": True,
            },
            "final_evaluation": {"start": HT, "end": HE, "hours": 1565},
            "embargo": {"start": HF, "end": HT, "hours": 593},
        }
        write_json(output / "manifests" / "split_manifest.json", split_manifest)
        write_json(
            output / "manifests" / "budget_manifest.json",
            {
                "definition": split_manifest["budgets"],
                "actual_eligible_label_counts": budget_counts,
            },
        )
        write_json(
            output / "manifests" / "feature_manifest.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "ordered_features": FEATURE_SCHEMA,
                "learned_predictor_exclusions": [
                    "city_id",
                    "station_id",
                    "coordinates",
                    "country",
                    "station status values",
                    "weather",
                    "holidays/events",
                    "future information",
                ],
                "dst_indicator_registered": False,
            },
        )

        log.add("[10/12] Auditing all registered feature channels")
        union_features = (
            f"read_parquet(['{sql_path(development_path)}',"
            f"'{sql_path(final_adaptation_path)}','{sql_path(final_features_path)}'], "
            "union_by_name=true)"
        )
        aggregate_expressions: list[str] = ["count(*)::BIGINT AS total_rows"]
        for index, item in enumerate(FEATURE_SCHEMA):
            column = item["column"]
            alias = f"f{index:02d}"
            value = f"cast({column} AS DOUBLE)"
            aggregate_expressions.extend(
                [
                    f"count_if({column} IS NULL)::BIGINT AS {alias}_missing",
                    f"count_if({column} IS NOT NULL AND isfinite({value}))::BIGINT AS {alias}_finite",
                    f"min({value})::DOUBLE AS {alias}_min",
                    f"median({value})::DOUBLE AS {alias}_median",
                    f"max({value})::DOUBLE AS {alias}_max",
                ]
            )
        feature_cursor = con.execute(
            "SELECT " + ",".join(aggregate_expressions) + f" FROM {union_features}"
        )
        feature_values = dict(zip([x[0] for x in feature_cursor.description], feature_cursor.fetchone()))
        feature_total = int(feature_values["total_rows"])
        feature_audit: list[dict[str, Any]] = []
        for index, item in enumerate(FEATURE_SCHEMA):
            alias = f"f{index:02d}"
            missing = int(feature_values[f"{alias}_missing"])
            finite = int(feature_values[f"{alias}_finite"])
            feature_audit.append(
                {
                    **item,
                    "rows": feature_total,
                    "missing_count": missing,
                    "missing_fraction": missing / feature_total if feature_total else None,
                    "finite_fraction": finite / (feature_total - missing)
                    if feature_total > missing
                    else None,
                    "min": feature_values[f"{alias}_min"],
                    "median": feature_values[f"{alias}_median"],
                    "max": feature_values[f"{alias}_max"],
                }
            )
        lag_inconsistency = con.sql(
            "SELECT "
            + " + ".join(
                [
                    f"count_if(NOT hist_lag_{lag:03d}_observed "
                    f"AND hist_lag_{lag:03d}_log1p != 0)"
                    for lag in range(1, 25)
                ]
                + [
                    "count_if(NOT week_lag_168_observed AND week_lag_168_log1p != 0)"
                ]
            )
            + f" FROM {union_features}"
        ).fetchone()[0]
        if lag_inconsistency != 0:
            raise RuntimeError("STOP: missing lag placeholders are inconsistent with masks.")
        validation.append(
            {
                "name": "lag_placeholder_mask_consistency",
                "status": "PASS",
                "actual": lag_inconsistency,
                "expected": 0,
            }
        )
        lag_offsets_bad = [
            item["column"]
            for item in FEATURE_SCHEMA
            if item["tensor"] in ("X_hist", "X_week")
            and (item["offset_hours"] is None or item["offset_hours"] >= 0)
        ]
        if lag_offsets_bad:
            raise RuntimeError(f"STOP: non-causal registered lag offsets: {lag_offsets_bad}")
        validation.append(
            {
                "name": "all_dynamic_feature_offsets_strictly_precede_target",
                "status": "PASS",
                "actual": lag_offsets_bad,
                "expected": [],
            }
        )
        rack_extremes = rows_as_dicts(
            con.execute(
                """
                SELECT city_id, station_id, bike_racks
                FROM eligible_stations
                ORDER BY bike_racks DESC, city_id, station_id LIMIT 20
                """
            )
        )
        write_json(
            output / "audits" / "feature_statistics.json",
            {
                "feature_statistics": feature_audit,
                "lag_mask_inconsistencies": lag_inconsistency,
                "largest_rack_values": rack_extremes,
                "timezone_warnings": [
                    {
                        "city_id": 532,
                        "source_timezone": "Europe/Berlin",
                        "canonical_geographic_timezone": "Europe/Madrid",
                        "calendar_feature_differences_in_frozen_interval": bilbao_tz_difference,
                    }
                ],
            },
        )

        log.add("[11/12] Writing audit reports and machine-readable validation evidence")
        write_json(output / "audits" / "validation_results.json", validation)
        city_cols = [
            "city_id",
            "city",
            "listed_stations",
            "eligible_stations",
            "analysis_hours",
            "raw_station_hours",
            "primary_observed_station_hours",
            "coverage_qualified_zeros",
            "positive_station_hours",
            "total_eligible_departures",
            "masked_station_hours",
            "observed_station_hours_24h",
            "observed_station_hours_no_maintenance",
        ]
        phase_cols = [
            "temporal_phase",
            "hours",
            "station_hours",
            "primary_observed_station_hours",
            "coverage_qualified_zeros",
            "positive_station_hours",
            "total_eligible_departures",
            "masked_station_hours",
        ]
        budget_cols = ["city_id", "city", "target_role", "budget", "eligible_target_labels"]
        exclusion_map = {row["disposition"]: row["trip_rows"] for row in trip_exclusions}
        totals = {
            "panel_rows": con.sql("SELECT count(*) FROM panel_core").fetchone()[0],
            "primary_observed": con.sql(
                "SELECT count_if(coverage_observed_12h) FROM panel_core"
            ).fetchone()[0],
            "zeros": con.sql(
                "SELECT count_if(coverage_observed_12h AND raw_departure_count=0) FROM panel_core"
            ).fetchone()[0],
            "positives": con.sql(
                "SELECT count_if(coverage_observed_12h AND raw_departure_count>0) FROM panel_core"
            ).fetchone()[0],
            "positive_excluded": con.sql(
                "SELECT count_if(NOT coverage_observed_12h AND raw_departure_count>0) FROM panel_core"
            ).fetchone()[0],
            "unknown_zero_candidates": con.sql(
                "SELECT count_if(NOT coverage_observed_12h AND raw_departure_count=0) FROM panel_core"
            ).fetchone()[0],
            "departures_below_q": exclusion_map.get("station_below_q_threshold", 0),
        }
        model_report = f"""# Model-ready panel audit

## Protocol result

The V2.1 builder retained **12 cities and 799 stations** on an explicit
`[H0, HE)` hourly grid. The primary target is nullable and label-independent:
only Rule B coverage produces a numeric target. The primary rule contains no
same-hour status-event condition. Maintenance is excluded only in the symmetric
sensitivity cohort.

## City panel statistics

{markdown_table(city_summary, city_cols)}

`total_eligible_departures` is the raw count at frozen-roster stations before
coverage masking. The full CSV also reports observed departures, uncovered
positive hours/departures, and unknown candidate zeros.

## Temporal phases

{markdown_table(phase_summary, phase_cols)}

All intervals are UTC, left-closed, and right-open. The embargo is never a fit
period.

## Actual target-label budgets

{markdown_table(budget_counts, budget_cols)}

These are eligible primary station-hour labels inside the elapsed registered
windows. No window was lengthened to replace missing labels.

## Final evaluation cohorts

{markdown_table(final_cohort, ['city_id','city','primary_evaluation_keys','positive_station_hours','coverage_qualified_zeros','sensitivity_24h_keys','no_maintenance_keys'])}

Every final target passes the registered 90% city-hour coverage gate:

{markdown_table(final_gate_rows, ['city_id','city','final_city_hour_coverage_fraction'])}

## Retention and exclusions

- Invalid start timestamps in protocol-city raw trips: **{timestamp_diagnostics['invalid_start_timestamp_rows']}**.
- Null station references in the analysis interval: **{exclusion_map.get('null_station_reference', 0)}**.
- Malformed non-integer/non-finite station references: **{exclusion_map.get('malformed_station_reference', 0)}**.
- Orphan station references: **{exclusion_map.get('orphan_station_reference', 0)}**.
- Cross-city station references: **{exclusion_map.get('cross_city_station_reference', 0)}**.
- Departures at stations below `q_s >= 0.50`: **{exclusion_map.get('station_below_q_threshold', 0)}**.
- Accepted eligible departures: **{exclusion_map.get('accepted_eligible_departure', 0)}**.
- Station-hours excluded by primary coverage: **{totals['panel_rows'] - totals['primary_observed']}**.
- Positive station-hours excluded by coverage: **{totals['positive_excluded']}**.
- No-trip station-hours left unknown rather than changed to zero: **{totals['unknown_zero_candidates']}**.

The raw eligible-departure total exactly reconciles between classified trips
and the explicit station-hour grid. Positive rows outside coverage are null in
the primary target, just like no-trip rows outside coverage.

## Timezone and metadata warnings

Bilbao is stored as `Europe/Berlin`, while `Europe/Madrid` is geographically
canonical. Their UTC offsets and calendar transforms are identical throughout
the frozen interval, so the source value causes zero feature differences and
does not trigger the protocol's timezone hard stop. This warning is preserved;
the source metadata was not silently changed.

Rack capacity is transformed exactly as registered even when the raw value is
nonpositive or extreme. Anomalies are reported in `FEATURE_DATA_AUDIT.md` and
are not automatically repaired.
"""
        (output / "MODEL_READY_PANEL_AUDIT.md").write_text(model_report, encoding="utf-8")

        feature_rows_for_md = []
        for item in feature_audit:
            feature_rows_for_md.append(
                {
                    "feature": item["column"],
                    "dtype": item["dtype"],
                    "tensor": item["tensor"],
                    "missing": item["missing_count"],
                    "finite_fraction": item["finite_fraction"],
                    "min": item["min"],
                    "median": item["median"],
                    "max": item["max"],
                    "transformed": item["transform"],
                    "fitted": item["fitted_statistics_required"],
                    "information_time": item["information_timestamp"],
                }
            )
        feature_report = f"""# Feature data audit

## Frozen schema

The audit covers the exact 58 registered channels: 24 hourly lag values and 24
matching masks, one weekly lag and mask, two rack channels, and six calendar
channels. Identity, coordinates, city/station IDs, status evidence, targets,
coverage state, weather, holidays, and events are not learned predictors.

{markdown_table(feature_rows_for_md, ['feature','dtype','tensor','missing','finite_fraction','min','median','max','transformed','fitted','information_time'])}

## Integrity findings

- Dynamic offsets are strictly negative: `t-1h` through `t-24h` and `t-168h`.
- A missing dynamic value always has mask 0 and numeric placeholder 0; detected inconsistencies: **{lag_inconsistency}**.
- Calendar features use the source IANA timezone after fixing the UTC key.
- The feature set requires no fitted mean, variance, quantile, or target-specific normalizer.
- The future status event used retrospectively for coverage is not a feature.
- A DST indicator is not stored because FEATURE_PROTOCOL_V2_1 does not register one; offset/12 is the registered channel.

## Rack-capacity anomaly evidence

Largest raw values in the frozen roster:

{markdown_table(rack_extremes, ['city_id','station_id','bike_racks'])}

Nonpositive rack values use the registered `(0, validity=0)` representation.
Extreme values are flagged here and remain unchanged pending any future
pre-result amendment.

## Timezone warning

Bilbao's source timezone is `Europe/Berlin`. A complete hourly comparison to
`Europe/Madrid` over `[H0,HE)` found **{bilbao_tz_difference}** local-calendar
differences. The warning is retained because the identifier is semantically
noncanonical even though V2.1 feature values are unchanged.
"""
        (output / "FEATURE_DATA_AUDIT.md").write_text(feature_report, encoding="utf-8")

        log.add("[12/12] Hashing sources and deterministic artifacts")
        source_files = sorted(
            list(cities_glob.parent.glob("*.parquet"))
            + list(stations_glob.parent.glob("*.parquet"))
            + list(trips_glob.parent.glob("*.parquet"))
            + list(status_base.glob("*.parquet"))
        )
        source_hashes = {
            path.relative_to(ROOT).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_files
        }
        base_source_manifest = json.loads(
            (ROOT / "feasibility_test" / "full_audit" / "source_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        status_source_manifest = json.loads(
            (
                ROOT
                / "feasibility_test"
                / "full_audit"
                / "coverage"
                / "station_status_source_manifest.json"
            ).read_text(encoding="utf-8")
        )
        if (
            base_source_manifest.get("revision") != PINNED_REVISION
            or status_source_manifest.get("revision") != PINNED_REVISION
        ):
            raise RuntimeError("STOP: source manifests do not record the frozen revision.")
        recorded_hashes: dict[str, str] = {}
        for item in base_source_manifest["files"]:
            if item["path"].split("/", 1)[0] in {"cities", "stations", "trips"}:
                recorded_hashes[
                    (
                        Path("feasibility_test")
                        / "full_audit"
                        / "data"
                        / Path(item["path"])
                    ).as_posix()
                ] = item["sha256"]
        for item in status_source_manifest["files"]:
            recorded_hashes[
                (
                    Path("feasibility_test")
                    / "full_audit"
                    / "coverage"
                    / "data"
                    / "station_status"
                    / item["path"]
                ).as_posix()
            ] = item["sha256"]
        mismatched_source_hashes = [
            relative
            for relative, evidence in source_hashes.items()
            if recorded_hashes.get(relative) != evidence["sha256"]
        ]
        if mismatched_source_hashes or set(recorded_hashes) != set(source_hashes):
            raise RuntimeError(
                "STOP: local Parquet hashes do not match the pinned source manifests: "
                f"{mismatched_source_hashes}"
            )
        validation.append(
            {
                "name": "all_local_source_hashes_match_pinned_manifests",
                "status": "PASS",
                "actual": len(source_hashes),
                "expected": len(recorded_hashes),
            }
        )
        write_json(output / "audits" / "validation_results.json", validation)
        protocol_hashes = {
            name: sha256_file(ROOT / name) for name in PROTOCOL_FILES
        }
        final_key_logical_hash = query_logical_hash(con, final_key_query)

        row_counts = {
            "canonical_grid": con.sql("SELECT count(*) FROM panel_core").fetchone()[0],
            "development_panel": con.sql(
                f"SELECT count(*) FROM read_parquet('{sql_path(development_path)}')"
            ).fetchone()[0],
            "final_adaptation_panel": con.sql(
                f"SELECT count(*) FROM read_parquet('{sql_path(final_adaptation_path)}')"
            ).fetchone()[0],
            "final_evaluation_features": con.sql(
                f"SELECT count(*) FROM read_parquet('{sql_path(final_features_path)}')"
            ).fetchone()[0],
            "final_prediction_keys": con.sql(
                f"SELECT count(*) FROM read_parquet('{sql_path(final_keys_path)}')"
            ).fetchone()[0],
            "sealed_final_evaluation_labels": con.sql(
                f"SELECT count(*) FROM read_parquet('{sql_path(final_labels_path)}')"
            ).fetchone()[0],
        }

        deterministic_candidates = sorted(
            path
            for path in output.rglob("*")
            if path.is_file()
            and path.name
            not in {
                "PROTOCOL_DATASET_MANIFEST.json",
                "artifact_hashes.json",
                "build.log",
                "reproducibility_check.json",
            }
        )
        deterministic_hashes = {
            path.relative_to(output).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in deterministic_candidates
        }
        manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "dataset_pinned_revision": PINNED_REVISION,
            "build_timestamp_utc": build_timestamp,
            "builder": "research/scripts/build_protocol_dataset.py",
            "duckdb_version": duckdb.__version__,
            "source_file_hashes": source_hashes,
            "protocol_file_hashes": protocol_hashes,
            "city_roles": split_manifest["city_roles"],
            "station_eligibility": {
                "definition": "pre-HF 12h bracketed-lifetime fraction",
                "threshold": STATION_THRESHOLD,
                "finite_coordinates_required": True,
                "station_counts": actual_stations,
                "total": 799,
            },
            "coverage": {
                "primary_rule": "Rule B: lifetime + station 12h bracket + >=50% relevant city stations bracketed",
                "same_hour_event_required": False,
                "primary_bracket_hours_each_side": BRACKET_HOURS_PRIMARY,
                "sensitivity_bracket_hours_each_side": BRACKET_HOURS_SENSITIVITY,
                "city_fraction": CITY_BRACKET_FRACTION,
                "maintenance_in_primary": False,
                "no_maintenance_sensitivity_symmetric": True,
            },
            "temporal_boundaries": split_manifest["boundaries"],
            "target_definition": (
                "Hourly valid city-matched trip departures at a frozen-roster station; "
                "numeric iff the relevant retrospective coverage mask is true, else null."
            ),
            "feature_schema": FEATURE_SCHEMA,
            "final_evaluation_key_logical_sha256": final_key_logical_hash,
            "row_counts": row_counts,
            "station_counts": actual_stations,
            "primary_counts": totals,
            "actual_budget_label_counts": budget_counts,
            "final_cohort": final_cohort,
            "trip_exclusions": {
                "timestamp_diagnostics": timestamp_diagnostics,
                "analysis_interval_dispositions": trip_exclusions,
            },
            "validation_summary": {
                "pass": sum(item["status"] == "PASS" for item in validation),
                "warning": sum(item["status"] == "WARNING" for item in validation),
                "fail": sum(item["status"] == "FAIL" for item in validation),
            },
            "artifact_hashes": deterministic_hashes,
            "final_seal": {
                "feature_artifact": final_features_path.relative_to(output).as_posix(),
                "prediction_key_artifact": final_keys_path.relative_to(output).as_posix(),
                "label_artifact": final_labels_path.relative_to(output).as_posix(),
                "labels_excluded_from_final_features": True,
                "labels_excluded_from_development": True,
                "note": "Final labels are evaluation-only. Dynamic features use strictly earlier observed lags, including causally released history.",
            },
        }
        manifest_path = output / "PROTOCOL_DATASET_MANIFEST.json"
        write_json(manifest_path, manifest)
        manifest_hash = sha256_file(manifest_path)
        write_json(
            output / "artifact_hashes.json",
            {
                "deterministic_artifacts": deterministic_hashes,
                "protocol_dataset_manifest": {
                    "bytes": manifest_path.stat().st_size,
                    "sha256": manifest_hash,
                    "contains_nondeterministic_build_timestamp": True,
                },
                "final_evaluation_key_logical_sha256": final_key_logical_hash,
            },
        )
        log.add(
            f"COMPLETE: {row_counts['canonical_grid']} grid rows; "
            f"{totals['primary_observed']} primary observations; 799 stations."
        )
        log.write(output / "build.log", build_timestamp)
        return manifest
    finally:
        con.close()
        if not args.keep_work_db:
            for suffix in ("", ".wal"):
                candidate = Path(str(work_db) + suffix)
                if candidate.exists():
                    candidate.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "processed" / "protocol_v2_1",
        help="Versioned output directory inside the repository.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=max(1, min(os.cpu_count() or 4, 8)))
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--keep-work-db", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
