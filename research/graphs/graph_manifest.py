from __future__ import annotations

import json
from pathlib import Path

from .geographic_graph import GeographicGraph


def write_graph_manifest(graph: GeographicGraph, city_id: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"protocol_version": "2.1", "city_id": int(city_id), **graph.checkpoint_config()}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
