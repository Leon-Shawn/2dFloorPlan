"""Unit tests for cleaner-sim. Standard library only:

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleaner_sim import grid as G
from cleaner_sim.coverage import analyse_coverage
from cleaner_sim.houseexpo import Floorplan, FloorplanLibrary, load_floorplan
from cleaner_sim.imageplan import ImageParseParams, otsu_threshold, parse_floorplan_image
from cleaner_sim.metrics import efficiency_metrics, format_duration
from cleaner_sim.planner import decompose_bcd, merge_cells, plan_coverage
from cleaner_sim.rasterize import (build_occupancy, build_robot_view, build_world,
                                   fine_from_plan)
from cleaner_sim.simulate import MotionParams, replay_payload, simulate_motion
from cleaner_sim.uploads import UploadStore


def rect_plan(w=6.0, h=4.0, name="rect"):
    return Floorplan(id=name, path=Path(f"{name}.json"), room_num=1,
                     bbox_min=(0.0, 0.0), bbox_max=(w, h),
                     verts=[[0, 0], [w, 0], [w, h], [0, h]], room_category={})


def lshape_plan():
    verts = [[0, 0], [6, 0], [6, 2], [3, 2], [3, 4], [0, 4]]
    return Floorplan(id="lshape", path=Path("lshape.json"), room_num=2,
                     bbox_min=(0.0, 0.0), bbox_max=(6.0, 4.0),
                     verts=verts, room_category={})


# --------------------------------------------------------------------- grid.py
class TestGridPrimitives(unittest.TestCase):
    def test_pack_bits_roundtrip(self):
        mask = bytearray((i % 3 == 0) for i in range(101))
        packed = G.pack_bits(mask)
        self.assertEqual(len(packed), 13)                      # ceil(101/8)
        for i, v in enumerate(mask):
            self.assertEqual((packed[i >> 3] >> (i & 7)) & 1, v, f"bit {i}")

    def test_erode_square_matches_bruteforce(self):
        w = h = 15
        src = bytearray(w * h)
        for y in range(3, 12):
            for x in range(3, 12):
                src[y * w + x] = 1
        for r in (0, 1, 2, 3):
            got = G.erode_square(src, w, h, r)
            want = bytearray(w * h)
            for y in range(h):
                for x in range(w):
                    if all(0 <= y + dy < h and 0 <= x + dx < w
                           and src[(y + dy) * w + x + dx]
                           for dy in range(-r, r + 1) for dx in range(-r, r + 1)):
                        want[y * w + x] = 1
            self.assertEqual(list(got), list(want), f"erode r={r}")

    def test_dilate_square_matches_bruteforce(self):
        w = h = 13
        src = bytearray(w * h)
        src[6 * w + 6] = 1
        for r in (1, 2, 3):
            got = G.dilate_square(src, w, h, r)
            want = bytearray(w * h)
            for y in range(h):
                for x in range(w):
                    if any(0 <= y + dy < h and 0 <= x + dx < w
                           and src[(y + dy) * w + x + dx]
                           for dy in range(-r, r + 1) for dx in range(-r, r + 1)):
                        want[y * w + x] = 1
            self.assertEqual(list(got), list(want), f"dilate r={r}")

    def test_label_components(self):
        w = h = 10
        mask = bytearray(w * h)
        for i in (0, 1, w, w + 1):                 # 2x2 block
            mask[i] = 1
        mask[9 * w + 9] = 1                        # isolated cell
        labels, sizes = G.label_components(mask, w, h)
        self.assertEqual(len(sizes), 2)
        self.assertEqual(sorted(sizes.values()), [1, 4])

    def test_astar_path_is_contiguous_and_avoids_walls(self):
        w = h = 12
        passable = bytearray([1]) * (w * h)
        for y in range(12):                        # vertical wall with a gap
            if y != 9:
                passable[y * w + 5] = 0
        path = G.astar(passable, w, h, 0, 11 * w + 11)
        self.assertIsNotNone(path)
        self.assertTrue(all(passable[i] for i in path))
        for a, b in zip(path, path[1:]):
            ay, ax = divmod(a, w)
            by, bx = divmod(b, w)
            self.assertLessEqual(max(abs(ax - bx), abs(ay - by)), 1)

    def test_astar_returns_none_when_walled_off(self):
        w = h = 9
        passable = bytearray([1]) * (w * h)
        for y in range(h):
            passable[y * w + 4] = 0
        self.assertIsNone(G.astar(passable, w, h, 0, 8))

    def test_encode_path_directions(self):
        w = 5
        path = [0, 1, 2, 2 + w, 2 + 2 * w]
        codes = list(G.encode_path(path, w))
        self.assertEqual(codes, [1, 1, 2, 2])      # right, right, down, down
        self.assertEqual(G.DIR_CODES[(1, 0)], 1)


# ---------------------------------------------------------------- rasterize.py
class TestRasterise(unittest.TestCase):
    def test_rectangle_area(self):
        occ = build_occupancy(rect_plan(6.0, 4.0), cell_cm=10.0)
        # 6x4 m at 10 cm cells = 60x40 interior cells (+ padding ring of wall)
        self.assertAlmostEqual(G.count(occ.walkable) * occ.cell_area_m2, 24.0, delta=1.0)
        self.assertEqual(occ.stats["walkable_cells"], G.count(occ.walkable))

    def test_cell_size_scales_grid(self):
        coarse = build_occupancy(rect_plan(), cell_cm=20.0)
        fine = build_occupancy(rect_plan(), cell_cm=10.0)
        self.assertGreater(fine.width, coarse.width)
        self.assertLess(abs(fine.area_m2(fine.walkable)
                            - coarse.area_m2(coarse.walkable)), 1.0)

    def test_lshape_is_single_region(self):
        occ = build_occupancy(lshape_plan(), cell_cm=10.0)
        labels, sizes = G.label_components(occ.walkable, occ.width, occ.height)
        self.assertEqual(len(sizes), 1, "L-shape floor must be one connected region")

    def test_robot_footprint_and_dead_split(self):
        occ = build_occupancy(rect_plan(), cell_cm=10.0)
        for robot_cm in (20.0, 34.0, 60.0):
            view = build_robot_view(occ, robot_cm=robot_cm)
            # cleanable + dead partitions the floor exactly
            self.assertEqual(G.count(view.cleanable) + G.count(view.dead),
                             G.count(occ.walkable))
            self.assertEqual(G.count(view.unreachable) + G.count(view.tight),
                             G.count(view.dead))
            # a bigger robot can never clean more than a smaller one
            self.assertGreater(G.count(view.cleanable), 0)
        small = build_robot_view(occ, robot_cm=20.0)
        big = build_robot_view(occ, robot_cm=60.0)
        self.assertGreaterEqual(G.count(small.cleanable), G.count(big.cleanable))

    def test_narrow_gap_splits_robot_regions(self):
        # Two 6x6 m rooms separated by a wall with only a 20 cm doorway, so a 34 cm
        # robot cannot pass: this is the "窄通道与门洞通过性" case from the brief.
        # The polygon traces the free space, carving the two wall pieces out of the
        # 5.9..6.1 band, which leaves the gap x in [2.9, 3.1].
        verts = [[0, 0], [6, 0], [6, 5.9], [3.1, 5.9], [3.1, 6.1],
                 [6, 6.1], [6, 12], [0, 12], [0, 6.1], [2.9, 6.1],
                 [2.9, 5.9], [0, 5.9]]
        plan = Floorplan(id="gap", path=Path("gap.json"), room_num=2,
                         bbox_min=(0.0, 0.0), bbox_max=(6.0, 12.0),
                         verts=verts, room_category={})
        occ = build_occupancy(plan, cell_cm=10.0)
        # the floor itself is connected (the doorway exists)...
        _, floor_sizes = G.label_components(occ.walkable, occ.width, occ.height)
        self.assertEqual(len(floor_sizes), 1, "floor should be connected through the gap")

        # ...but the robot cannot fit through it
        view = build_robot_view(occ, robot_cm=34.0)
        self.assertGreater(view.stats["robot_regions"], 1,
                           "a 20 cm doorway must split the robot's regions")
        self.assertGreater(view.stats["separated_cells"], 0)
        self.assertGreater(view.stats["dead_ratio"], 0.2)


# ------------------------------------------------------------------ planner.py
class TestPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.occ, cls.view = build_world(rect_plan(7.0, 5.0), cell_cm=10.0, robot_cm=34.0)
        cls.result = plan_coverage(cls.occ, cls.view, target_cov=0.99)

    def test_path_is_contiguous_and_legal(self):
        path = self.result.path
        self.assertGreater(len(path), 10)
        w = self.occ.width
        for a, b in zip(path, path[1:]):
            ay, ax = divmod(a, w)
            by, bx = divmod(b, w)
            self.assertLessEqual(max(abs(ax - bx), abs(ay - by)), 1,
                                 f"non-adjacent step {a}->{b}")
        for i in path:
            self.assertEqual(self.view.reachable[i], 1,
                             "robot must only ever stand on legal positions")

    def test_per_step_bookkeeping_is_consistent(self):
        self.assertEqual(len(self.result.path), len(self.result.newly))
        self.assertEqual(len(self.result.path), len(self.result.touch))
        self.assertTrue(all(t >= n for t, n in zip(self.result.touch, self.result.newly)))
        self.assertEqual(sum(self.result.newly), G.count(self.result.covered))

    def test_coverage_target_reached(self):
        self.assertGreaterEqual(self.result.stats["coverage_reachable"], 0.99)
        self.assertTrue(G.count(G.difference(self.result.covered, self.occ.walkable)) == 0,
                        "cleaned cells must all be walkable")

    def test_bcd_partitions_reachable_cells(self):
        """Cells from both sweep axes must partition the reachable set exactly."""
        for axis in ("x", "y"):
            labels, cells = decompose_bcd(self.view.reachable, self.occ.width,
                                          self.occ.height, axis)
            labelled = sum(1 for i, lab in enumerate(labels)
                           if lab and self.view.reachable[i])
            self.assertEqual(labelled, G.count(self.view.reachable), f"axis={axis}")
            self.assertEqual(sum(c.count for c in cells.values()), labelled,
                             f"cell counts disagree for axis={axis}")
            # every cell is a solid rectangle in a slab, so its cells must all be
            # reachable and inside its own bounding box
            for cell in cells.values():
                self.assertGreaterEqual(cell.count, 1)
                for y in range(cell.y0, cell.y1 + 1):
                    for x in range(cell.x0, cell.x1 + 1):
                        self.assertTrue(self.view.reachable[y * self.occ.width + x],
                                        "BCD cell extends outside the reachable set")

    def test_polygon_source_labels_are_rectangles(self):
        occ = build_occupancy(rect_plan(8.0, 6.0), cell_cm=10.0)
        view = build_robot_view(occ, robot_cm=34.0)
        labels, cells = decompose_bcd(view.reachable, occ.width, occ.height, "x")
        self.assertEqual(len(cells), 1, "an obstacle-free rectangle is a single cell")

    def test_merge_keeps_partition_and_reduces_cells(self):
        occ, view = build_world(lshape_plan(), cell_cm=10.0, robot_cm=34.0)
        labels, cells = decompose_bcd(view.reachable, occ.width, occ.height, "x")
        before = len(cells)
        m_labels, m_cells = merge_cells(labels, cells, occ.width, occ.height)
        labelled = sum(1 for i, lab in enumerate(m_labels)
                       if lab and view.reachable[i])
        self.assertEqual(labelled, G.count(view.reachable),
                         "merging must not drop or duplicate cells")
        self.assertLessEqual(len(m_cells), before)
        for cell in m_cells.values():
            # a merged cell is still monotone on at least one axis
            monotone = (all(len(v) <= 1 for v in cell.rows.values())
                        or all(len(v) <= 1 for v in cell.cols.values()))
            self.assertTrue(monotone, f"cell {cell.cid} is not monotone on any axis")

    def test_spacing_modes(self):
        for mode, spacing in (("exact", 2 * self.view.radius_cells + 1),
                              ("overlap", 2 * self.view.radius_cells)):
            res = plan_coverage(self.occ, self.view, spacing_mode=mode)
            self.assertEqual(res.stats["sweep_spacing_cells"], spacing)
            self.assertGreaterEqual(res.stats["coverage_reachable"], 0.99, mode)


# --------------------------------------------------- coverage / metrics / sim
class TestCoverageAndMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.occ, cls.view = build_world(lshape_plan(), cell_cm=10.0, robot_cm=34.0)
        cls.result = plan_coverage(cls.occ, cls.view, target_cov=0.995)
        cls.coverage = analyse_coverage(cls.occ, cls.view, cls.result.covered)
        cls.sim = simulate_motion(cls.result, cls.occ,
                                  G.count(cls.view.cleanable), MotionParams())

    def test_coverage_partition(self):
        cov = self.coverage
        self.assertEqual(cov["covered_cells"] + cov["dirty_cells"],
                         cov["cleanable_cells"])
        self.assertEqual(cov["cleanable_cells"] + cov["dead_cells"],
                         cov["walkable_cells"])
        self.assertLessEqual(cov["coverage_walkable"], 1.0)
        self.assertLessEqual(cov["coverage_reachable"], 1.0)

    def test_area_accounting_matches_cells(self):
        area = self.occ.cell_area_m2
        self.assertAlmostEqual(self.coverage["walkable_area_m2"],
                               self.coverage["walkable_cells"] * area, places=1)

    def test_metrics_are_sane(self):
        m = efficiency_metrics(self.occ, self.coverage, self.result, self.sim)
        self.assertGreater(m["steps"], 0)
        self.assertGreater(m["path_length_m"], 0)
        self.assertGreaterEqual(m["repeat_rate"], 0.0)
        self.assertLessEqual(m["effective_ratio"], 1.0)
        self.assertGreater(m["area_per_meter"], 0.0)

    def test_simulation_matches_path(self):
        self.assertEqual(len(self.sim.poses), len(self.result.path))
        self.assertEqual(self.sim.times, sorted(self.sim.times), "time must be monotonic")
        # first pose is the first path cell, in metres
        y, x = divmod(self.result.path[0], self.occ.width)
        px, py, _ = self.sim.poses[0]
        self.assertAlmostEqual(px, x * self.occ.cell_m, places=6)
        self.assertAlmostEqual(py, y * self.occ.cell_m, places=6)

    def test_timeline_is_monotonic_and_final(self):
        curve = self.sim.coverage_curve
        self.assertGreater(len(curve), 1)
        values = [v for _, v in curve]
        self.assertEqual(values, sorted(values), "coverage may never decrease")
        self.assertAlmostEqual(values[-1], self.coverage["coverage_reachable"], places=3)

    def test_replay_payload_roundtrip(self):
        payload = replay_payload(self.occ, self.result, self.result.path[0])
        dirs = G.base64.b64decode(payload["dirs_b64"])
        self.assertEqual(len(dirs), len(self.result.path) - 1)
        w = self.occ.width
        x, y = payload["start"]
        rebuilt = [y * w + x]
        for code in dirs:
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1)][code]
            x += dx
            y += dy
            rebuilt.append(y * w + x)
        self.assertEqual(rebuilt, list(self.result.path),
                         "browser replay must reconstruct the exact route")

    def test_format_duration(self):
        self.assertEqual(format_duration(45), "45 秒")
        self.assertEqual(format_duration(125), "2 分 05 秒")
        self.assertIn("小时", format_duration(7200))


# ------------------------------------------------------------- real dataset
class TestRealFloorplans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.lib = FloorplanLibrary()
        except FileNotFoundError:
            cls.lib = None

    def setUp(self):
        if self.lib is None:
            self.skipTest("HouseExpo dataset not available")

    def test_first_floorplan_end_to_end(self):
        plan = self.lib.get(self.lib.ids()[0])
        occ, view = build_world(plan, cell_cm=10.0, robot_cm=34.0)
        result = plan_coverage(occ, view, target_cov=0.99)
        self.assertTrue(result.path)
        self.assertGreaterEqual(result.stats["coverage_reachable"], 0.99)
        w = occ.width
        for a, b in zip(result.path, result.path[1:]):
            ay, ax = divmod(a, w)
            by, bx = divmod(b, w)
            self.assertLessEqual(max(abs(ax - bx), abs(ay - by)), 1)

    def test_loader_fields(self):
        plan = self.lib.get(self.lib.ids()[0])
        self.assertTrue(plan.id)
        self.assertGreaterEqual(len(plan.verts), 3)
        self.assertGreater(plan.area_m2, 0)
        self.assertGreaterEqual(plan.room_num, 1)


# ------------------------------------------------------- image upload pipeline
class TestImagePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.lib = FloorplanLibrary()
        except FileNotFoundError:
            cls.lib = None

    def setUp(self):
        if self.lib is None:
            self.skipTest("HouseExpo dataset not available")

    def test_otsu_separates_bimodal_histogram(self):
        gray = bytes([10] * 300 + [240] * 700)
        t = otsu_threshold(gray)
        self.assertTrue(10 <= t < 240, f"threshold {t} not between the two modes")

    def test_render_parse_roundtrip_matches_vector(self):
        from tools.test_image_pipeline import render_floorplan_image
        plan = self.lib.get(self.lib.ids()[0])
        gray, w, h, _ = render_floorplan_image(plan)
        real_w = max(v[0] for v in plan.verts) - min(v[0] for v in plan.verts)
        fine, diag = parse_floorplan_image(gray, w, h,
                                           ImageParseParams(real_width_m=real_w))
        self.assertGreater(diag["free_px"], 0)
        self.assertEqual(diag["warnings"], [], "clean synthetic plan must not warn")

        occ_img, _ = build_world(fine, cell_cm=10.0, robot_cm=34.0)
        occ_vec, _ = build_world(plan, cell_cm=10.0, robot_cm=34.0)
        img_area = occ_img.area_m2(occ_img.walkable)
        vec_area = occ_vec.area_m2(occ_vec.walkable)
        self.assertLess(abs(img_area - vec_area) / vec_area, 0.05,
                        f"image parse {img_area}m2 vs vector {vec_area}m2")

    def test_blank_image_is_rejected(self):
        gray = bytes([255] * (60 * 60))
        with self.assertRaises(ValueError):
            parse_floorplan_image(gray, 60, 60, ImageParseParams(real_width_m=6.0))

    def test_scale_axis(self):
        from tools.test_image_pipeline import render_floorplan_image
        plan = self.lib.get(self.lib.ids()[0])
        gray, w, h, _ = render_floorplan_image(plan)
        size_y = max(v[1] for v in plan.verts) - min(v[1] for v in plan.verts)
        fine, _ = parse_floorplan_image(
            gray, w, h, ImageParseParams(real_height_m=size_y, scale_axis="y"))
        self.assertAlmostEqual(fine.size_y_m, size_y, delta=0.3)


class TestUploadStore(unittest.TestCase):
    def setUp(self):
        # a workspace-local scratch dir: the sandbox refuses to delete system temp dirs
        import shutil
        import uuid
        base = Path(__file__).resolve().parent / "_tmp"
        base.mkdir(exist_ok=True)
        self.dir = base / f"case-{uuid.uuid4().hex[:8]}"
        self._shutil = shutil
        self.store = UploadStore(self.dir)

    def tearDown(self):
        self._shutil.rmtree(self.dir, ignore_errors=True)

    def _library(self):
        try:
            return FloorplanLibrary()
        except FileNotFoundError:
            self.skipTest("HouseExpo dataset not available")

    def test_houseexpo_upload_roundtrip(self):
        lib = self._library()
        src = lib.get(lib.ids()[0])
        up = self.store.add_houseexpo("mine.json", src.path.read_text(encoding="utf-8"))
        self.assertTrue(up.id.startswith("up-"))
        self.assertEqual(len(self.store.list_meta()), 1)
        self.assertEqual(self.store.list_meta()[0]["custom"], True)
        reloaded = self.store.source(up.id)
        self.assertGreaterEqual(len(reloaded.verts), 3)
        self.assertTrue(self.store.delete(up.id))
        self.assertEqual(self.store.list_meta(), [])

    def test_rejects_malformed_json(self):
        for bad in ('{"id": "x"}', "not json at all", '{"id":"a","bbox":{},"verts":[]}'):
            with self.assertRaises(ValueError):
                self.store.add_houseexpo("bad.json", bad)

    def test_image_upload_and_reparse(self):
        from tools.test_image_pipeline import render_floorplan_image
        lib = self._library()
        plan = lib.get(lib.ids()[1])
        gray, w, h, _ = render_floorplan_image(plan)
        real_w = max(v[0] for v in plan.verts) - min(v[0] for v in plan.verts)
        up = self.store.add_image("scan.png", bytes(gray), w, h,
                                  ImageParseParams(real_width_m=real_w))
        self.assertEqual(up.kind, "image")
        self.assertGreater(up.meta["size_x_m"], 1.0)
        self.assertEqual(up.diagnostics["warnings"], [])
        original_width = up.meta["size_x_m"]          # capture: reparse mutates in place

        wider = self.store.reparse_image(up.id, ImageParseParams(real_width_m=real_w * 2))
        self.assertAlmostEqual(wider.meta["size_x_m"], original_width * 2, delta=0.2)
        self.assertGreater(self.store.source(up.id).width, 8)

    def test_upload_survives_reload(self):
        lib = self._library()
        src = lib.get(lib.ids()[0])
        up = self.store.add_houseexpo("mine.json", src.path.read_text(encoding="utf-8"))
        reopened = UploadStore(self.dir)
        self.assertIn(up.id, reopened.items)


if __name__ == "__main__":
    unittest.main(verbosity=2)
