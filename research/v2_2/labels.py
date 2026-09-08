"""Separate retrospective-label descriptor; no label file reader or evaluator.

Neither history.py nor snapshots.py imports this module. Actual retrospective
adjudication/scoring belongs to a later separately authorized evaluation path.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrospectiveLabel:
    city_id: int
    station_id: int
    hour_start_us: int
    count: int | None
    observed: bool

    def __post_init__(self):
        if type(self.observed) is not bool:
            raise ValueError("Explicit retrospective observation mask required")
        if self.observed:
            if type(self.count) is not int or self.count < 0:
                raise ValueError("Observed retrospective labels require nonnegative counts")
        elif self.count is not None:
            raise ValueError("Unobserved retrospective labels are nullable")


@dataclass(frozen=True)
class RetrospectiveLabelSnapshot:
    adjudication_cutoff_us: int
    rows: tuple[RetrospectiveLabel, ...]

    def __post_init__(self):
        if not isinstance(self.rows, tuple) or not all(type(r) is RetrospectiveLabel for r in self.rows):
            raise TypeError("Immutable retrospective rows required")
