"""Module 5 — coverage statistics and dead-zone reporting.

Everything the dashboard shows about *how much floor got cleaned* is produced here:
overall coverage, coverage per connected floor region, remaining dirty area, and a
classified dead-zone list (walled off vs. too tight for the robot body).
"""
from __future__ import annotations

from . import grid as G
from .rasterize import Occupancy, RobotView


def _bbox(cells: list[int], width: int):
    ys = [c // width for c in cells]
    xs = [c % width for c in cells]
    return {"x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)}


def analyse_coverage(occ: Occupancy, view: RobotView, covered: bytearray) -> dict:
    width, height = occ.width, occ.height
    area = occ.cell_area_m2

    walkable = occ.walkable
    cleanable = view.cleanable
    covered_walkable = G.intersection(covered, walkable)
    covered_cleanable = G.intersection(covered, cleanable)

    n_walkable = G.count(walkable)
    n_cleanable = G.count(cleanable)
    n_covered = G.count(covered_walkable)
    n_covered_cleanable = G.count(covered_cleanable)
    dirty = G.intersection(cleanable, G.difference(walkable, covered))

    # Per-region breakdown: proves every connected room cluster got serviced, and
    # exposes regions the robot cannot enter at all.
    labels, sizes = G.label_components(walkable, width, height, 4)
    reachable_labels = set()
    for i, ok in enumerate(view.reachable):
        if ok:
            reachable_labels.add(labels[i])

    # single pass instead of re-scanning the grid once per region
    covered_by_label: dict[int, int] = {}
    for i, lab in enumerate(labels):
        if lab and covered_walkable[i]:
            covered_by_label[lab] = covered_by_label.get(lab, 0) + 1

    regions = []
    for label, cells in sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)[:12]:
        covered_here = covered_by_label.get(label, 0)
        regions.append({
            "region": label,
            "cells": cells,
            "area_m2": round(cells * area, 2),
            "covered_cells": covered_here,
            "coverage": round(covered_here / cells, 4),
            "robot_can_enter": label in reachable_labels,
        })

    # Dead zones, classified by cause.
    dead = G.difference(walkable, cleanable)
    dead_labels, dead_sizes = G.label_components(dead, width, height, 4)
    dead_members: dict[int, list[int]] = {}
    for i, lab in enumerate(dead_labels):
        if lab:
            dead_members.setdefault(lab, []).append(i)

    dead_clusters = []
    for label, cells in sorted(dead_sizes.items(), key=lambda kv: kv[1], reverse=True)[:12]:
        idxs = dead_members.get(label, [])
        separated = sum(1 for i in idxs if view.unreachable[i])
        tight = sum(1 for i in idxs if view.tight[i])
        is_separated = separated >= tight
        dead_clusters.append({
            "cells": cells,
            "area_m2": round(cells * area, 3),
            "reason": "separated" if is_separated else "tight",
            "reason_text": ("与起点区域不连通：通道容不下机器人本体，需分区多次清扫"
                            if is_separated else
                            "空间过窄/位于死角，任何合法位姿都够不到"),
            "separated_cells": separated,
            "tight_cells": tight,
            "bbox": _bbox(idxs, width),
        })

    return {
        "walkable_cells": n_walkable,
        "walkable_area_m2": round(n_walkable * area, 2),
        "cleanable_cells": n_cleanable,
        "cleanable_area_m2": round(n_cleanable * area, 2),
        "covered_cells": n_covered,
        "covered_area_m2": round(n_covered * area, 2),
        "covered_cleanable_cells": n_covered_cleanable,
        "coverage_walkable": round(n_covered / max(1, n_walkable), 4),
        "coverage_reachable": round(n_covered_cleanable / max(1, n_cleanable), 4),
        "dirty_cells": G.count(dirty),
        "dirty_area_m2": round(G.count(dirty) * area, 3),
        "dead_cells": G.count(dead),
        "dead_area_m2": round(G.count(dead) * area, 3),
        "dead_ratio": round(G.count(dead) / max(1, n_walkable), 4),
        "separated_cells": G.count(view.unreachable),
        "separated_area_m2": round(G.count(view.unreachable) * area, 3),
        "separated_ratio": round(G.count(view.unreachable) / max(1, n_walkable), 4),
        "tight_cells": G.count(view.tight),
        "tight_area_m2": round(G.count(view.tight) * area, 3),
        "tight_ratio": round(G.count(view.tight) / max(1, n_walkable), 4),
        "robot_regions": len([s for s in view.stats.get("robot_region_sizes", []) if s >= 4]),
        "dead_clusters": dead_clusters,
        "regions": regions,
    }
