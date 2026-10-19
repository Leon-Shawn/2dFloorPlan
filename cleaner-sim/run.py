#!/usr/bin/env python3
"""Start the 2D floorplan cleaning-simulation demo.

    python run.py                       # http://127.0.0.1:8017
    python run.py --port 9000
    python run.py --data ../HouseExpo/json     # after full dataset extraction

No third-party packages are required.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cleaner_sim.server import serve          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8017)
    ap.add_argument("--data", default=None,
                    help="directory of HouseExpo floorplan JSONs")
    args = ap.parse_args()
    serve(host=args.host, port=args.port, data_dir=args.data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
