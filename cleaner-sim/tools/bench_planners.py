"""Benchmark coverage planners over the whole sample dataset.

Compares the original watershed-region planner against the BCD planner and isolates
the two independent changes (cell decomposition, sweep spacing):

    legacy/overlap   原始实现（watershed 分区 + 扫描间距 2r）
    legacy/exact     仅改扫描间距为 2r+1
    bcd/overlap      仅改单元分解为 BCD
    bcd/exact        新默认（BCD + 精确间距）

Efficiency is measured as ``steps / ideal`` where
``ideal = cleanable_cells / (2r+1)`` — the number of steps an unobstructed
lawn-mower would need at the robot's footprint width. Lower is better; 1.0 is the
floor, and the overlap spacing alone puts a hard floor of (2r+1)/(2r) ≈ 1.25 on any
variant using it.

    python tools/bench_planners.py [--limit N] [--variants bcd/exact,legacy/overlap]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleaner_sim import grid as G
from cleaner_sim.coverage import analyse_coverage
from cleaner_sim.houseexpo import FloorplanLibrary
from cleaner_sim import planner as bcd_planner
from cleaner_sim import planner_watershed as legacy_planner
from cleaner_sim.rasterize import build_world

VARIANTS = {
    "legacy/overlap": (legacy_planner, "overlap"),
    "legacy/exact": (legacy_planner, "exact"),
    "bcd/overlap": (bcd_planner, "overlap"),
    "bcd/exact": (bcd_planner, "exact"),
}


def run_variant(module, spacing_mode, occ, view):
    t0 = time.perf_counter()
    result = module.plan_coverage(occ, view, target_cov=0.995,
                                  max_retreat_passes=4, spacing_mode=spacing_mode)
    elapsed = time.perf_counter() - t0
    if not result.path:
        return None, elapsed
    coverage = analyse_coverage(occ, view, result.covered)
    return (result, coverage), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N floorplans")
    ap.add_argument("--cell", type=float, default=10.0)
    ap.add_argument("--robot", type=float, default=34.0)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()

    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    for n in names:
        if n not in VARIANTS:
            sys.exit(f"unknown variant {n!r}; choose from {list(VARIANTS)}")

    lib = FloorplanLibrary()
    ids = lib.ids()
    if args.limit:
        ids = ids[: args.limit]
    metas = {m["id"]: m for m in lib.list_meta()}

    print(f"benchmarking {len(ids)} floorplans @ {args.cell:g}cm cell / "
          f"{args.robot:g}cm robot   variants: {', '.join(names)}\n")

    rows = []
    for n, fid in enumerate(ids, 1):
        plan = lib.get(fid)
        occ, view = build_world(plan, cell_cm=args.cell, robot_cm=args.robot)
        radius = view.radius_cells
        cleanable = G.count(view.cleanable)
        ideal = cleanable / (2 * radius + 1)
        entry = {"id": fid, "rooms": metas[fid]["room_num"],
                 "size": f"{metas[fid]['size_x_m']}x{metas[fid]['size_y_m']}",
                 "grid": f"{occ.width}x{occ.height}", "cleanable": cleanable,
                 "ideal": ideal, "variants": {}}
        for name in names:
            module, mode = VARIANTS[name]
            try:
                data, elapsed = run_variant(module, mode, occ, view)
            except Exception as exc:                     # noqa: BLE001
                entry["variants"][name] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            if data is None:
                entry["variants"][name] = {"error": "no path"}
                continue
            result, coverage = data
            steps = len(result.path)
            entry["variants"][name] = {
                "steps": steps,
                "ratio": steps / ideal,
                "coverage": coverage["coverage_reachable"],
                "retreat": result.stats.get("retreat_passes", 0),
                "cells": result.stats.get("cells", result.stats.get("regions", 0)),
                "reposition": result.stats.get("reposition_ratio"),
                "seconds": elapsed,
            }
        rows.append(entry)
        print(f"  [{n:>3}/{len(ids)}] {fid[:10]} rooms={entry['rooms']:>2} "
              f"grid={entry['grid']:>9}  " +
              "  ".join(
                  f"{k.split('/')[0][:4]}/{k.split('/')[1][:3]}="
                  + (f"{v['ratio']:.2f}" if "ratio" in v else "FAIL")
                  for k, v in entry["variants"].items()))

    # ------------------------------------------------------------- aggregates
    print("\n" + "=" * 104)
    print(f"{'variant':<18}{'mean ratio':>11}{'median':>9}{'total steps':>13}"
          f"{'vs legacy':>11}{'min cov':>9}{'cov<.999':>9}{'repos%':>8}{'mean s':>9}{'cells':>7}")
    print("-" * 104)

    baseline_total = None
    summary = {}
    for name in names:
        ok = [r["variants"][name] for r in rows if "ratio" in r["variants"][name]]
        if not ok:
            print(f"{name:<18}{'all failed':>11}")
            continue
        ratios = [v["ratio"] for v in ok]
        total_steps = sum(v["steps"] for v in ok)
        if baseline_total is None:
            baseline_total = total_steps
        imperfect = sum(1 for v in ok if v["coverage"] < 0.9999)
        min_cov = min(v["coverage"] for v in ok)
        repos = [v["reposition"] for v in ok if v.get("reposition") is not None]
        summary[name] = {"mean_ratio": statistics.fmean(ratios),
                         "total_steps": total_steps,
                         "min_coverage": min_cov,
                         "mean_seconds": statistics.fmean(v["seconds"] for v in ok),
                         "cells": statistics.fmean(v["cells"] for v in ok)}
        repos_txt = f"{statistics.fmean(repos) * 100:>7.1f}%" if repos else f"{'-':>8}"
        print(f"{name:<18}{statistics.fmean(ratios):>11.3f}"
              f"{statistics.median(ratios):>9.3f}{total_steps:>13,}"
              f"{(total_steps / baseline_total - 1) * 100:>10.1f}%"
              f"{min_cov:>9.4f}{imperfect:>9}{repos_txt}"
              f"{statistics.fmean(v['seconds'] for v in ok):>9.2f}"
              f"{statistics.fmean(v['cells'] for v in ok):>7.1f}")

    if len(names) >= 2 and "legacy/overlap" in summary and "bcd/exact" in summary:
        base = summary["legacy/overlap"]
        new = summary["bcd/exact"]
        print("-" * 96)
        print(f"总步数下降 : {base['total_steps']:,} -> {new['total_steps']:,}  "
              f"({(new['total_steps'] / base['total_steps'] - 1) * 100:+.1f}%)")
        print(f"平均 steps/ideal : {base['mean_ratio']:.3f} -> {new['mean_ratio']:.3f}  "
              f"({(new['mean_ratio'] / base['mean_ratio'] - 1) * 100:+.1f}%)")
        print(f"平均规划耗时 : {base['mean_seconds']:.2f}s -> {new['mean_seconds']:.2f}s")

    # per-floorplan worst/best for the two endpoint variants
    if "legacy/overlap" in names and "bcd/exact" in names:
        deltas = []
        for r in rows:
            a = r["variants"]["legacy/overlap"]
            b = r["variants"]["bcd/exact"]
            if "ratio" in a and "ratio" in b:
                deltas.append((b["ratio"] - a["ratio"], r["id"], r["rooms"]))
        deltas.sort()
        print("\n改善最大的 5 个户型:")
        for d, fid, rooms in deltas[:5]:
            print(f"   {fid[:10]} rooms={rooms:>2}  {d:+.2f} steps/ideal")
        print("改善最小（或变差）的 5 个户型:")
        for d, fid, rooms in deltas[-5:]:
            print(f"   {fid[:10]} rooms={rooms:>2}  {d:+.2f} steps/ideal")


if __name__ == "__main__":
    main()
