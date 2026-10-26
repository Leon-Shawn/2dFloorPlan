"""Module 4 (motion simulation) and module 9 (replay) — trajectory + timeline.

The planner produces a cell-by-cell route. This module turns it into a physical
motion trace: position in metres, heading, elapsed time (including a turn penalty
whenever the heading changes) and a coverage timeline that the browser scrubber and
the dashboard chart are built from.

Replay payload design
---------------------
The route is 4/8-connected, so it is shipped as one *movement code* byte per step
plus the start cell. The browser replays it by walking the codes, and reproduces the
covered area by stamping the same integer square footprint at every visited cell —
the identical rule the server uses, so the animation and the numbers always agree.
That keeps the payload a few kilobytes instead of shipping a grid per frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import grid as G
from .rasterize import Occupancy


@dataclass
class MotionParams:
    speed_mps: float = 0.25
    turn_time_s: float = 0.6
    start_heading_rad: float = 0.0


@dataclass
class Simulation:
    poses: list[tuple[float, float, float]]      # x_m, y_m, heading_rad
    times: list[float]                           # seconds
    path_length_m: float
    turns: int
    turn_seconds: float
    duration_s: float
    coverage_curve: list[tuple[int, float]]      # (step, coverage of cleanable floor)


def simulate_motion(plan_result, occ: Occupancy, cleanable_cells: int,
                    params: MotionParams | None = None) -> Simulation:
    params = params or MotionParams()
    path = plan_result.path
    width = occ.width
    cell_m = occ.cell_m

    poses: list[tuple[float, float, float]] = []
    times: list[float] = []
    length = 0.0
    turns = 0
    turn_seconds = 0.0
    elapsed = 0.0
    previous_dir = None
    running = 0
    coverage_curve: list[tuple[int, float]] = []

    denom = max(1, cleanable_cells)
    sample_every = max(1, len(path) // 240) if path else 1

    for i, idx in enumerate(path):
        y, x = divmod(idx, width)
        heading = params.start_heading_rad
        if i > 0:
            py, px = divmod(path[i - 1], width)
            dx, dy = x - px, y - py
            step_m = math.hypot(dx, dy) * cell_m
            length += step_m
            elapsed += step_m / max(1e-6, params.speed_mps)
            heading = math.atan2(dy, dx)
            direction = (dx, dy)
            if previous_dir is not None and direction != previous_dir:
                turns += 1
                elapsed += params.turn_time_s
                turn_seconds += params.turn_time_s
            previous_dir = direction

        poses.append((x * cell_m, y * cell_m, heading))
        times.append(elapsed)

        running += plan_result.newly[i]
        if i % sample_every == 0 or i == len(path) - 1:
            coverage_curve.append((i, round(running / denom, 5)))

    return Simulation(
        poses=poses,
        times=times,
        path_length_m=round(length, 2),
        turns=turns,
        turn_seconds=round(turn_seconds, 2),
        duration_s=round(elapsed, 2),
        coverage_curve=coverage_curve,
    )


def replay_payload(occ: Occupancy, plan_result, start_cell: int) -> dict:
    """Compact instructions the browser needs to replay the route exactly."""
    width = occ.width
    start_y, start_x = divmod(start_cell, width)
    return {
        "start": [start_x, start_y],
        "dirs_b64": G.b64(G.encode_path(plan_result.path, width)),
        "steps": len(plan_result.path),
    }
