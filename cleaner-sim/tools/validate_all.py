"""Batch validation: run the whole pipeline over every floorplan and report."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleaner_sim.pipeline import RunParams, SimulationService

svc = SimulationService()
metas = svc.floorplans()
print(f"{len(metas)} floorplans from {svc.data_dir()}\n")

header = (f"{'rooms':>5} {'size':>13} {'grid':>10} {'steps':>7} {'cov_reach':>9} "
          f"{'cov_all':>8} {'repeat':>7} {'dead':>6} {'seek':>6} {'total':>7}")
print(header)
print("-" * len(header))

failures = []
worst = []
total_time = 0.0
cov_sum = 0.0

for meta in metas:
    fid = meta["id"]
    try:
        t0 = time.perf_counter()
        p = svc.run(fid, RunParams())
        dt = time.perf_counter() - t0
    except Exception as exc:                                  # noqa: BLE001
        failures.append((fid, f"{type(exc).__name__}: {exc}"))
        print(f"{meta['room_num']:>5} {meta['size_x_m']:>5}x{meta['size_y_m']:<5} "
              f"FAILED  {type(exc).__name__}: {exc}")
        continue

    total_time += dt
    cov_sum += p["coverage"]["coverage_walkable"]
    seek = p["pipeline"]["stages"][2]["ms"] / 1000.0
    print(f"{meta['room_num']:>5} {meta['size_x_m']:>5}x{meta['size_y_m']:<5} "
          f"{p['grid']['w']:>4}x{p['grid']['h']:<4} {p['metrics']['steps']:>7} "
          f"{p['coverage']['coverage_reachable']:>9.4f} "
          f"{p['coverage']['coverage_walkable']:>8.4f} "
          f"{p['metrics']['repeat_rate']:>7.3f} "
          f"{p['coverage']['dead_ratio']:>6.3f} "
          f"{seek:>6.1f} {dt:>6.2f}s")
    worst.append((dt, fid, meta["room_num"]))

print("-" * len(header))
ok = len(metas) - len(failures)
print(f"OK {ok}/{len(metas)}   mean coverage(walkable)={cov_sum/max(1,ok):.4f}   "
      f"total {total_time:.1f}s   mean {total_time/max(1,ok):.2f}s")
worst.sort(reverse=True)
print("slowest:", [(f"{d:.2f}s", r) for d, _, r in worst[:5]])
if failures:
    print("\nFAILURES:")
    for fid, err in failures:
        print(f"  {fid}: {err}")
    sys.exit(1)
print("\nall floorplans planned successfully")
