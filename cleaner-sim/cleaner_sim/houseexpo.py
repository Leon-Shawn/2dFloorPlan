"""HouseExpo dataset access: locating floorplans, reading them, and metadata.

The dataset ships as 35k+ JSON files, each describing one 2D floorplan:

    id             unique house id
    room_num       number of rooms
    bbox           {"min": [x, y], "max": [x, y]} in metres
    verts          [[x, y], ...] a single closed outline polygon in metres
    room_category  {category: [[x1, y1, x2, y2], ...]} coarse room bounding boxes

`verts` traces the whole floor (exterior *and* interior walls) as one polygon, so
door openings appear as gaps in that outline. The room_category boxes are coarse
and overlapping — they are usable for "which room type is this point" lookups but
must not be mistaken for a room segmentation.
"""
from __future__ import annotations

import csv
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

# Where the dataset may live, in priority order, relative to the repo root.
_CANDIDATE_DIRS = (
    "../HouseExpo/samples/json",
    "HouseExpo/samples/json",
    "../HouseExpo/json",
    "HouseExpo/json",
    "../data/json",
    "data/json",
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Floorplan:
    id: str
    path: Path
    room_num: int
    bbox_min: tuple[float, float]
    bbox_max: tuple[float, float]
    verts: list[list[float]]
    room_category: dict[str, list] = field(default_factory=dict)

    @property
    def size_x_m(self) -> float:
        return round(self.bbox_max[0] - self.bbox_min[0], 2)

    @property
    def size_y_m(self) -> float:
        return round(self.bbox_max[1] - self.bbox_min[1], 2)

    @property
    def area_m2(self) -> float:
        return round(self.size_x_m * self.size_y_m, 2)

    @property
    def categories(self) -> list[str]:
        return sorted(self.room_category)

    def meta(self) -> dict:
        return {
            "id": self.id,
            "room_num": self.room_num,
            "size_x_m": self.size_x_m,
            "size_y_m": self.size_y_m,
            "area_m2": self.area_m2,
            "n_verts": len(self.verts),
            "categories": self.categories,
            "tier": ("simple" if self.room_num <= 4
                     else "medium" if self.room_num <= 9 else "complex"),
        }


def resolve_data_dir(explicit: str | None = None) -> Path:
    """Find the directory holding HouseExpo floorplan JSONs."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(f"data directory not found: {p}")
        return p.resolve()

    env = os.environ.get("CLEANER_DATA")
    if env and Path(env).is_dir():
        return Path(env).resolve()

    for rel in _CANDIDATE_DIRS:
        p = (REPO_ROOT / rel).resolve()
        if p.is_dir() and any(p.glob("*.json")):
            return p
    raise FileNotFoundError(
        "no HouseExpo floorplans found. Pass --data <dir> or set CLEANER_DATA, "
        "or point it at a directory of HouseExpo JSON files "
        "(the bundled samples live in HouseExpo/samples/json)."
    )


def load_floorplan(path: Path | str) -> Floorplan:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    for key in ("id", "bbox", "verts"):
        if key not in raw:
            raise ValueError(f"{path.name}: missing required field {key!r}")
    verts = raw["verts"]
    if len(verts) < 3:
        raise ValueError(f"{path.name}: outline has fewer than 3 vertices")
    bbox = raw["bbox"]
    return Floorplan(
        id=str(raw.get("id", path.stem)),
        path=path,
        room_num=int(raw.get("room_num", 0) or 0),
        bbox_min=(float(bbox["min"][0]), float(bbox["min"][1])),
        bbox_max=(float(bbox["max"][0]), float(bbox["max"][1])),
        verts=[[float(v[0]), float(v[1])] for v in verts],
        room_category=raw.get("room_category") or {},
    )


class FloorplanLibrary:
    """Lazy index over a directory of floorplans, with a small metadata cache."""

    def __init__(self, data_dir: str | None = None):
        self.data_dir = resolve_data_dir(data_dir)
        self._plans: dict[str, Floorplan] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()          # 懒加载缓存，多线程下需串行
        self._index()

    def _index(self) -> None:
        # Prefer an externally curated index CSV when present (adds tier ordering
        # for the 45 hand-picked samples), otherwise just scan the directory.
        csv_path = self.data_dir.parent / "sample_index.csv"
        ids: list[str] = []
        if csv_path.is_file():
            with csv_path.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    ids.append(row["id"])

        files = {p.stem: p for p in self.data_dir.glob("*.json")}
        ordered = [i for i in ids if i in files]
        ordered += [k for k in sorted(files) if k not in set(ordered)]
        self._order = ordered
        self._files = files

    def __len__(self) -> int:
        return len(self._order)

    def ids(self) -> list[str]:
        return list(self._order)

    def get(self, fid: str) -> Floorplan:
        with self._lock:
            plan = self._plans.get(fid)
            if plan is not None:
                return plan
            path = self._files.get(fid)
        if path is None:
            raise KeyError(f"unknown floorplan id: {fid}")
        plan = load_floorplan(path)
        with self._lock:
            plan = self._plans.setdefault(fid, plan)
        return plan

    def list_meta(self) -> list[dict]:
        out = []
        for fid in self._order:
            try:
                out.append(self.get(fid).meta())
            except (ValueError, OSError, KeyError):
                continue                      # skip unreadable entries rather than fail
        return out
