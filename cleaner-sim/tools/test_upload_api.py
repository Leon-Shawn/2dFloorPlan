"""Integration test for the upload + simulation HTTP API.

Exercises the endpoints a browser would call, using only the standard library:

    python tools/test_upload_api.py [--port 8017]

Covers: health, catalogue, HouseExpo-JSON upload, floorplan-image upload,
stateless parse preview, re-parse with new settings, running a simulation on an
uploaded floorplan, and deletion.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleaner_sim import grid as G
from cleaner_sim.houseexpo import FloorplanLibrary
from cleaner_sim.rasterize import fine_from_plan

from tools.test_image_pipeline import render_floorplan_image  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = ""):
    results.append((name, PASS if ok else FAIL, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def call(port: int, path: str, method: str = "GET", payload=None, timeout=300):
    url = f"http://127.0.0.1:{port}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "json" in ctype:
                return resp.status, json.loads(body.decode("utf-8"))
            return resp.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body.decode("utf-8"))
        except ValueError:
            return exc.code, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8017)
    args = ap.parse_args()
    port = args.port

    print(f"upload API integration test against http://127.0.0.1:{port}\n")

    # ------------------------------------------------------------------ health
    status, health = call(port, "/api/health")
    if not check("GET /api/health", status == 200 and health.get("ok"),
                 f"{health.get('floorplans')} floorplans"):
        return 1

    status, listing = call(port, "/api/floorplans")
    builtin = [f for f in listing["floorplans"] if not f.get("custom")]
    check("GET /api/floorplans", status == 200 and len(builtin) > 0,
          f"{len(builtin)} built-in")

    # --------------------------------------------------- upload a HouseExpo JSON
    lib = FloorplanLibrary()
    src = lib.get(lib.ids()[0])
    raw = json.loads(src.path.read_text(encoding="utf-8"))
    status, res = call(port, "/api/upload/houseexpo", "POST",
                       {"name": "my-floorplan.json", "text": json.dumps(raw)})
    ok = status == 200 and res.get("id")
    check("POST /api/upload/houseexpo", bool(ok), res.get("id", res.get("error", "")))
    json_id = res.get("id") if ok else None

    if json_id:
        status, listing = call(port, "/api/floorplans")
        found = [f for f in listing["floorplans"] if f["id"] == json_id]
        check("uploaded JSON appears in catalogue", status == 200 and len(found) == 1)

        status, run = call(port, f"/api/run?id={json_id}&cell_cm=10&robot_cm=34")
        ok = status == 200 and run.get("metrics", {}).get("steps", 0) > 0
        check("simulate on uploaded JSON", bool(ok),
              f"steps={run.get('metrics', {}).get('steps')} "
              f"cov={run.get('coverage', {}).get('coverage_walkable')}")

        status, png = call(port, f"/api/thumb?id={json_id}&w=168")
        check("thumbnail for uploaded JSON",
              status == 200 and isinstance(png, bytes) and png[:4] == b"\x89PNG")

    # --------------------------------------------------------- upload an image
    plan = lib.get(lib.ids()[3])
    gray, w, h, _fine = render_floorplan_image(plan)
    real_w = max(v[0] for v in plan.verts) - min(v[0] for v in plan.verts)
    img_body = {
        "name": "scanned-floorplan.png",
        "width": w, "height": h,
        "gray_b64": base64.b64encode(bytes(gray)).decode("ascii"),
        "real_width_m": real_w,
        "scale_axis": "x",
        "seal_px": 2,
    }

    status, prev = call(port, "/api/parse/preview", "POST", img_body)
    diag = prev.get("diagnostics", {}) if status == 200 else {}
    check("POST /api/parse/preview",
          status == 200 and prev.get("preview") and diag.get("free_px", 0) > 0,
          f"threshold={diag.get('threshold')} area={diag.get('floor_area_m2')}m2")

    status, res = call(port, "/api/upload/image", "POST", img_body)
    ok = status == 200 and res.get("id")
    check("POST /api/upload/image", bool(ok), res.get("id", res.get("error", "")))
    img_id = res.get("id") if ok else None

    if img_id:
        status, run = call(port, f"/api/run?id={img_id}&cell_cm=10&robot_cm=34")
        ok = status == 200 and run.get("metrics", {}).get("steps", 0) > 0
        cov = run.get("coverage", {}).get("coverage_reachable")
        check("simulate on uploaded image", bool(ok) and cov == 1.0,
              f"steps={run.get('metrics', {}).get('steps')} cov_reach={cov}")

        # the image pipeline must land close to the vector geometry of the same plan
        occ_v, _ = __import__("cleaner_sim.rasterize", fromlist=["build_world"]).build_world(
            plan, cell_cm=10.0, robot_cm=34.0)
        vector_area = occ_v.area_m2(occ_v.walkable)
        image_area = run["coverage"]["walkable_area_m2"]
        delta = abs(image_area - vector_area) / vector_area
        check("image parse matches vector geometry", delta < 0.05,
              f"{image_area:.1f}m2 vs {vector_area:.1f}m2 ({delta * 100:.1f}%)")

        status, res2 = call(port, "/api/upload/reparse", "POST",
                            {"id": img_id, "real_width_m": real_w * 1.2,
                             "threshold": "auto", "seal_px": 2})
        check("POST /api/upload/reparse",
              status == 200 and res2.get("meta", {}).get("size_x_m", 0) > 0,
              f"width={res2.get('meta', {}).get('size_x_m')}m")

    # --------------------------------------------------------------- rejection
    status, res = call(port, "/api/upload/houseexpo", "POST",
                       {"name": "bad.json", "text": '{"id": "x"}'})
    check("rejects malformed floorplan JSON",
          status == 400 and "error" in res, res.get("error", ""))

    status, res = call(port, "/api/upload/image", "POST",
                       {"width": 4, "height": 4, "gray_b64": "AAAA"})
    check("rejects mismatched grayscale buffer", status == 400)

    # ------------------------------------------------------------------ delete
    for uid in (json_id, img_id):
        if not uid:
            continue
        status, res = call(port, "/api/delete", "POST", {"id": uid})
        check(f"POST /api/delete {uid[:16]}", status == 200 and res.get("deleted"))
        status, listing = call(port, "/api/floorplans")
        gone = not any(f["id"] == uid for f in listing["floorplans"])
        check("deleted upload leaves the catalogue", gone)

    failed = [r for r in results if r[1] == FAIL]
    print()
    if failed:
        print(f"FAILED: {len(failed)}/{len(results)} checks")
        for name, _, detail in failed:
            print(f"   - {name}  {detail}")
        return 1
    print(f"PASS: all {len(results)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
