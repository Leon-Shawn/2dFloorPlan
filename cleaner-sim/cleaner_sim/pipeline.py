"""End-to-end orchestration: floorplan source + parameters -> everything the web UI needs.

    source (HouseExpo JSON polygon | uploaded floorplan image)
      -> [module 1] rasterise  -> occupancy grid + robot model
      -> [engine]   plan       -> full-coverage route (BCD + lawn-mower + retreat)
      -> [module 4] simulate   -> poses, timing, coverage timeline
      -> [module 5] coverage   -> coverage / dead-zone statistics
      -> [module 7] metrics    -> efficiency scorecard
      -> [module 9] replay     -> compact route for browser playback
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import grid as G
from .coverage import analyse_coverage
from .houseexpo import FloorplanLibrary
from .imageplan import ImageParseParams
from .metrics import efficiency_metrics
from .planner import SPACING_MODES, plan_coverage
from .rasterize import FineMask, build_world, thumbnail_from_fine, thumbnail_png
from .simulate import MotionParams, replay_payload, simulate_motion
from .uploads import UploadStore

DEFAULT_UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"


@dataclass(frozen=True)
class RunParams:
    cell_cm: float = 10.0
    robot_cm: float = 34.0
    target_cov: float = 0.995
    max_retreat: int = 4
    speed_mps: float = 0.25
    turn_time_s: float = 0.6
    supersample: int = 4
    wall_threshold: float = 0.5
    spacing_mode: str = "exact"

    def key(self):
        return (self.cell_cm, self.robot_cm, self.target_cov, self.max_retreat,
                self.speed_mps, self.turn_time_s, self.supersample,
                self.wall_threshold, self.spacing_mode)

    def as_dict(self):
        return asdict(self)


def _mask_b64(mask) -> str:
    return G.b64(G.pack_bits(mask))


MAX_PHASES = 30


def _compact_phases(phases: list[dict]) -> list[dict]:
    """Keep the payload small when a busy floor produces dozens of retreat phases."""
    if len(phases) <= MAX_PHASES:
        return phases
    head, tail = phases[0], phases[-1]
    middle = sorted(phases[1:-1], key=lambda p: p.get("cluster_cells", 0), reverse=True)
    kept = [head] + middle[:MAX_PHASES - 2] + [tail]
    return sorted(kept, key=lambda p: p["start_step"])


class SimulationService:
    """Caches floorplan metadata, thumbnails and simulation runs."""

    def __init__(self, data_dir: str | None = None, upload_dir=None, cache_size: int = 48):
        self.library = FloorplanLibrary(data_dir)
        self.uploads = UploadStore(upload_dir or DEFAULT_UPLOAD_DIR)
        self._runs: dict[tuple, dict] = {}
        self._thumbs: dict[tuple[str, int], bytes] = {}
        self.cache_size = cache_size

    # ------------------------------------------------------------------ metadata
    def data_dir(self) -> str:
        return str(self.library.data_dir)

    def upload_dir(self) -> str:
        return str(self.uploads.dir)

    def floorplans(self) -> list[dict]:
        """Built-in dataset first, then user uploads (newest first)."""
        return self.library.list_meta() + self.uploads.list_meta()

    def is_upload(self, fid: str) -> bool:
        return fid in self.uploads.items

    def source_for(self, fid: str):
        """Resolve an id to a Floorplan (built-in / JSON upload) or FineMask (image)."""
        if fid in self.uploads.items:
            return self.uploads.source(fid)
        return self.library.get(fid)

    def thumbnail(self, floorplan_id: str, width: int = 168) -> bytes:
        key = (floorplan_id, width)
        if key not in self._thumbs:
            if floorplan_id in self.uploads.items:
                self._thumbs[key] = self.uploads.thumbnail(floorplan_id, width)
            else:
                self._thumbs[key] = thumbnail_png(self.library.get(floorplan_id), width)
        return self._thumbs[key]

    def invalidate(self, floorplan_id: str) -> None:
        for key in [k for k in self._runs if k[0] == floorplan_id]:
            del self._runs[key]
        for key in [k for k in self._thumbs if k[0] == floorplan_id]:
            del self._thumbs[key]

    # -------------------------------------------------------------------- uploads
    def add_houseexpo(self, name: str, text: str) -> dict:
        up = self.uploads.add_houseexpo(name, text)
        return {"id": up.id, "meta": up.meta, "kind": up.kind}

    def add_image(self, name: str, gray: bytes, width: int, height: int,
                  params: ImageParseParams) -> dict:
        up = self.uploads.add_image(name, gray, width, height, params)
        return {"id": up.id, "meta": up.meta, "kind": up.kind,
                "diagnostics": up.diagnostics,
                "preview": G.b64(self.uploads.thumbnail(up.id, 320))}

    def reparse_image(self, uid: str, params: ImageParseParams) -> dict:
        up = self.uploads.reparse_image(uid, params)
        self.invalidate(uid)
        return {"id": up.id, "meta": up.meta, "kind": up.kind,
                "diagnostics": up.diagnostics,
                "preview": G.b64(self.uploads.thumbnail(uid, 320))}

    def parse_preview(self, gray: bytes, width: int, height: int,
                      params: ImageParseParams) -> dict:
        """Stateless parse used for the live preview before anything is saved."""
        from .imageplan import parse_floorplan_image
        from .rasterize import png_from_masks
        fine, diag = parse_floorplan_image(gray, width, height, params)
        wall = bytearray(1 if not v else 0 for v in fine.inside)
        png = png_from_masks(wall, fine.width, fine.height)
        return {"diagnostics": diag, "preview": G.b64(png),
                "meta": fine.meta, "size": [fine.width, fine.height]}

    def delete_upload(self, uid: str) -> bool:
        ok = self.uploads.delete(uid)
        if ok:
            self.invalidate(uid)
        return ok

    # ----------------------------------------------------------------- simulation
    def run(self, floorplan_id: str, params: RunParams) -> dict:
        key = (floorplan_id, params.key())
        cached = self._runs.get(key)
        if cached is not None:
            return cached

        t0 = time.perf_counter()
        source = self.source_for(floorplan_id)
        t_parse = time.perf_counter()

        if isinstance(source, FineMask):
            source.meta.setdefault("id", floorplan_id)
        meta = source.meta if isinstance(source, FineMask) else source.meta()

        occ, view = build_world(source, cell_cm=params.cell_cm, robot_cm=params.robot_cm,
                                supersample=params.supersample,
                                wall_threshold=params.wall_threshold)
        t_raster = time.perf_counter()

        plan_result = plan_coverage(occ, view, target_cov=params.target_cov,
                                    max_retreat_passes=params.max_retreat,
                                    spacing_mode=params.spacing_mode)
        t_plan = time.perf_counter()

        if not plan_result.path:
            raise RuntimeError(plan_result.stats.get("error", "planning produced no route"))

        simulation = simulate_motion(
            plan_result, occ, G.count(view.cleanable),
            MotionParams(speed_mps=params.speed_mps, turn_time_s=params.turn_time_s))
        coverage = analyse_coverage(occ, view, plan_result.covered)
        efficiency = efficiency_metrics(occ, coverage, plan_result, simulation)
        t_sim = time.perf_counter()

        if isinstance(source, FineMask):
            parse_detail = (f"上传图像 {occ.stats['source_px'][0]}x{occ.stats['source_px'][1]} px "
                            f"-> {occ.stats['source_px'][0] * source.meters_per_px:.1f}m 宽")
        else:
            parse_detail = f"{len(source.verts)} 个轮廓顶点 / {source.room_num} 个房间"

        payload = {
            "floorplan": meta,
            "params": params.as_dict(),
            "grid": {
                "w": occ.width,
                "h": occ.height,
                "cell_cm": occ.cell_cm,
                "requested_cell_cm": occ.stats.get("requested_cell_cm"),
                "radius_cells": view.radius_cells,
                "start": list(divmod(view.start, occ.width)[::-1]),   # [x, y]
                "wall_b64": _mask_b64(occ.wall),
                "cleanable_b64": _mask_b64(view.cleanable),
                "separated_b64": _mask_b64(view.unreachable),
                "tight_b64": _mask_b64(view.tight),
                "packing": "row-major, LSB-first within each byte",
            },
            "replay": replay_payload(occ, plan_result, plan_result.path[0]),
            "phases": _compact_phases(plan_result.phases),
            "timeline": simulation.coverage_curve,
            "coverage": coverage,
            "metrics": efficiency,
            "pipeline": {
                "stages": [
                    {"name": "解析户型" if not isinstance(source, FineMask) else "解析上传图像",
                     "ms": round((t_parse - t0) * 1000, 1), "detail": parse_detail},
                    {"name": "栅格化 + 机器人建模", "ms": round((t_raster - t_parse) * 1000, 1),
                     "detail": f"{occ.width}x{occ.height} 栅格 @ {occ.cell_cm:g}cm，"
                               f"墙体阈值 {params.wall_threshold:g}"},
                    {"name": "BCD 全覆盖路径规划", "ms": round((t_plan - t_raster) * 1000, 1),
                     "detail": f"{plan_result.stats['steps']} 步 · "
                               f"{plan_result.stats.get('cells', 0)} 个单元 · "
                               f"扫描间距 {plan_result.stats.get('sweep_spacing_cells')} 格"},
                    {"name": "运动仿真 + 指标统计", "ms": round((t_sim - t_plan) * 1000, 1),
                     "detail": f"{len(simulation.poses)} 帧，覆盖曲线 "
                               f"{len(simulation.coverage_curve)} 采样点"},
                ],
                "total_ms": round((t_sim - t0) * 1000, 1),
                "occupancy_stats": occ.stats,
                "planner_stats": plan_result.stats,
                "motion": {
                    "path_length_m": simulation.path_length_m,
                    "turns": simulation.turns,
                    "turn_seconds": simulation.turn_seconds,
                    "duration_s": simulation.duration_s,
                },
            },
        }

        if len(self._runs) >= self.cache_size:
            self._runs.pop(next(iter(self._runs)))
        self._runs[key] = payload
        return payload


__all__ = ["RunParams", "SimulationService", "SPACING_MODES", "ImageParseParams"]
