"""Module 7 — path efficiency indicators.

Consumes the finished plan, the motion trace and the coverage report and derives the
quantitative scorecard the dashboard renders: steps, path length, turns, duration,
repetition rate, and the two "unit" efficiencies (area cleaned per metre travelled,
time per square metre).
"""
from __future__ import annotations

from .coverage import analyse_coverage
from .rasterize import Occupancy
from .simulate import Simulation


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} 秒"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} 分 {sec:02d} 秒"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} 小时 {minutes:02d} 分"


def efficiency_metrics(occ: Occupancy, coverage: dict, plan_result,
                       simulation: Simulation) -> dict:
    steps = len(plan_result.path)
    unique_covered = coverage["covered_cells"]
    total_sweeps = sum(plan_result.touch)

    repeat_rate = (total_sweeps / unique_covered - 1.0) if unique_covered else 0.0
    effective_ratio = (unique_covered / total_sweeps) if total_sweeps else 0.0

    path_length = simulation.path_length_m
    covered_area = coverage["covered_area_m2"]
    duration = simulation.duration_s

    # Efficiency against the unobstructed lawn-mower lower bound: a robot with a
    # footprint W cells wide must take at least cleanable/W steps, because each step
    # advances the cleaned frontier by at most W.
    spacing = plan_result.stats.get("sweep_spacing_cells") or 1
    ideal_steps = coverage["cleanable_cells"] / spacing

    return {
        "steps": steps,
        "path_length_m": path_length,
        "turns": simulation.turns,
        "turn_seconds": simulation.turn_seconds,
        "duration_s": duration,
        "duration_text": format_duration(duration),
        "algorithm": plan_result.stats.get("algorithm", "-"),
        "cells": plan_result.stats.get("cells"),
        "ideal_steps": round(ideal_steps, 1),
        "path_efficiency": round(ideal_steps / steps, 4) if steps else 0.0,
        "reposition_ratio": plan_result.stats.get("reposition_ratio"),
        # repetition: how many extra passes each cleaned cell received on average
        "disk_sweeps": total_sweeps,
        "repeat_rate": round(repeat_rate, 4),
        "effective_ratio": round(effective_ratio, 4),
        # unit efficiencies
        "area_per_meter": round(covered_area / path_length, 4) if path_length else 0.0,
        "time_per_m2_s": round(duration / covered_area, 3) if covered_area else 0.0,
        "steps_per_m2": round(steps / covered_area, 2) if covered_area else 0.0,
        "area_per_hour_m2": round(covered_area / (duration / 3600.0), 1) if duration else 0.0,
        # plan quality
        "coverage_reachable": coverage["coverage_reachable"],
        "coverage_walkable": coverage["coverage_walkable"],
        "dead_ratio": coverage["dead_ratio"],
        "retreat_passes": plan_result.stats.get("retreat_passes", 0),
        "deferred_intervals": plan_result.stats.get("deferred_intervals", 0),
        "sweep_spacing_cells": plan_result.stats.get("sweep_spacing_cells"),
        "decomposition_axis": plan_result.stats.get("decomposition_axis"),
        "cell_area_m2": round(occ.cell_area_m2, 5),
    }


__all__ = ["efficiency_metrics", "format_duration", "analyse_coverage"]
