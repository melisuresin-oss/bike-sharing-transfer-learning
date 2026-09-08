from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from research.data.manifests import ProtocolArtifactRegistry, sql_path


EARTH_RADIUS_KM = 6371.0088


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def haversine_distance_matrix(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    if latitude.ndim != 1 or longitude.shape != latitude.shape:
        raise ValueError("Latitude and longitude must be equal-length vectors")
    lat = np.radians(latitude)
    lon = np.radians(longitude)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(
        lat[None, :]
    ) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


@dataclass(frozen=True)
class GeographicGraph:
    station_ids: np.ndarray
    normalized_adjacency: np.ndarray
    weighted_adjacency_with_loops: np.ndarray
    selected_edge_mask: np.ndarray
    sigma_km: float
    k_requested: int
    k_effective: int
    invalid_coordinate_nodes: tuple[int, ...]
    roster_hash_sha256: str
    coordinate_hash_sha256: str
    adjacency_hash_sha256: str

    @property
    def node_count(self) -> int:
        return int(self.station_ids.size)

    def checkpoint_config(self) -> dict[str, object]:
        return {
            "method": "symmetric_union_geographic_knn_gaussian_gcn_normalization",
            "k": self.k_requested,
            "k_effective": self.k_effective,
            "sigma_km": self.sigma_km,
            "node_count": self.node_count,
            "roster_hash_sha256": self.roster_hash_sha256,
            "coordinate_hash_sha256": self.coordinate_hash_sha256,
            "adjacency_hash_sha256": self.adjacency_hash_sha256,
        }


def build_geographic_graph(
    station_ids: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    k: int,
    roster_hash_sha256: str | None = None,
    coordinate_hash_sha256: str | None = None,
) -> GeographicGraph:
    """Build the registered graph from station IDs and coordinates only."""
    station_ids = np.asarray(station_ids, dtype=np.int64)
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    if station_ids.ndim != 1 or latitude.shape != station_ids.shape or longitude.shape != station_ids.shape:
        raise ValueError("Station IDs and coordinates must be equal-length vectors")
    if len(np.unique(station_ids)) != len(station_ids):
        raise ValueError("Station ordering contains duplicate station IDs")
    if k <= 0:
        raise ValueError("k must be positive and supplied by the caller")
    nodes = station_ids.size
    if nodes == 0:
        raise ValueError("Cannot build an empty graph")
    valid = np.isfinite(latitude) & np.isfinite(longitude)
    distances = haversine_distance_matrix(
        np.where(valid, latitude, 0.0), np.where(valid, longitude, 0.0)
    )
    distances[~valid, :] = np.inf
    distances[:, ~valid] = np.inf
    np.fill_diagonal(distances, np.inf)
    directed = np.zeros((nodes, nodes), dtype=bool)
    valid_count = int(valid.sum())
    k_effective = min(int(k), max(0, valid_count - 1))
    for node in np.flatnonzero(valid):
        candidates = np.flatnonzero(valid)
        candidates = candidates[candidates != node]
        order = np.lexsort((candidates, distances[node, candidates]))
        directed[node, candidates[order[:k_effective]]] = True
    selected = directed | directed.T
    edge_distances = distances[np.triu(selected, k=1)]
    edge_distances = edge_distances[np.isfinite(edge_distances) & (edge_distances > 0)]
    sigma = float(np.median(edge_distances)) if edge_distances.size else 1.0
    weighted = np.zeros((nodes, nodes), dtype=np.float64)
    if selected.any():
        finite_selected = selected & np.isfinite(distances)
        weighted[finite_selected] = np.exp(-((distances[finite_selected] / sigma) ** 2))
    np.fill_diagonal(weighted, 1.0)
    degree = weighted.sum(axis=1)
    inverse_sqrt = np.ones_like(degree)
    np.power(degree, -0.5, out=inverse_sqrt, where=degree > 0)
    normalized = inverse_sqrt[:, None] * weighted * inverse_sqrt[None, :]
    normalized = normalized.astype(np.float32)
    weighted = weighted.astype(np.float32)
    roster_hash = roster_hash_sha256 or _hash_arrays(station_ids)
    coordinate_hash = coordinate_hash_sha256 or _hash_arrays(
        station_ids, latitude, longitude
    )
    adjacency_hash = _hash_arrays(station_ids, normalized)
    invalid = tuple(int(index) for index in np.flatnonzero(~valid))
    return GeographicGraph(
        station_ids=station_ids.copy(),
        normalized_adjacency=normalized,
        weighted_adjacency_with_loops=weighted,
        selected_edge_mask=selected,
        sigma_km=sigma,
        k_requested=int(k),
        k_effective=k_effective,
        invalid_coordinate_nodes=invalid,
        roster_hash_sha256=roster_hash,
        coordinate_hash_sha256=coordinate_hash,
        adjacency_hash_sha256=adjacency_hash,
    )


def build_city_graph(
    registry: ProtocolArtifactRegistry, city_id: int, *, k: int
) -> GeographicGraph:
    """Project only ordering/coordinate columns from an approved station manifest."""
    artifact = registry.station_manifest(city_id)
    entry = registry.station_manifest_entry(city_id)
    con = registry.connect()
    try:
        data = con.execute(
            f"SELECT station_id,latitude,longitude "
            f"FROM read_parquet('{sql_path(artifact)}') ORDER BY node_index"
        ).fetchnumpy()
    finally:
        con.close()
    graph = build_geographic_graph(
        data["station_id"],
        data["latitude"],
        data["longitude"],
        k=k,
        roster_hash_sha256=entry["roster_hash_sha256"],
        coordinate_hash_sha256=entry["coordinate_hash_sha256"],
    )
    if graph.node_count != int(entry["station_count"]):
        raise RuntimeError("Graph node count differs from the frozen station manifest")
    return graph
