"""Sanity check: every DOM id the front-end script looks up exists in index.html.

Catches the most common front-end runtime error (a typo'd element id) without
needing a browser:

    python tools/check_dom_ids.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"

html = (WEB / "index.html").read_text(encoding="utf-8")
js = (WEB / "app.js").read_text(encoding="utf-8")
css = (WEB / "style.css").read_text(encoding="utf-8")

declared = set(re.findall(r'id="([^"]+)"', html))
used = set(re.findall(r"""\$\(\s*['"]([^'"]+)['"]\s*\)""", js))
used |= set(re.findall(r"""getElementById\(\s*['"]([^'"]+)['"]\s*\)""", js))

# ids created dynamically by the script itself
dynamic = set(re.findall(r"""id="([^"]+)"[^`]*`""", js))

missing = sorted(used - declared - dynamic)
unused = sorted(declared - used)

print(f"index.html ids      : {len(declared)}")
print(f"ids looked up in js : {len(used)}")
print(f"missing             : {missing if missing else 'none'}")
print(f"declared but unused : {unused if unused else 'none'}")

# cheap structural checks on the stylesheet
bad = [ln for ln in css.splitlines() if "considered" in ln or "??" in ln]
print(f"stylesheet smells   : {bad if bad else 'none'}")

if missing:
    sys.exit(1)
print("OK")
