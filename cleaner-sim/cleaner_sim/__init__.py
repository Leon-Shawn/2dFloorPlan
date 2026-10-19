"""Cleaner-sim: 2D floorplan full-coverage cleaning simulation.

Module map (course brief, topic 15)
----------------------------------
1. 户型图转栅格地图 (core)  -> rasterize.py
4. 机器人运动仿真           -> simulate.py
5. 覆盖率计算与可视化 (core) -> coverage.py
7. 路径效率指标             -> metrics.py
9. 仿真过程回放             -> simulate.py (replay payload) + web/app.js

The coverage path planner itself (brief module 3) lives in planner.py and is the
engine the other modules animate and measure.
"""

__version__ = "1.0.0"
