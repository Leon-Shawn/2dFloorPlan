"""Registry for user-uploaded floorplans (HouseExpo JSON, or a floorplan image).

Uploads are kept in memory for the running session and mirrored to disk so they
survive a restart:

    uploads/
      index.json                 metadata for every upload
      <id>.json                  a HouseExpo-format floorplan
      <id>.gray                  the raw grayscale of an uploaded image
      <id>.png                   the parsed mask preview

Both kinds resolve to a floorplan *source* that `rasterize.build_world` accepts:
a `Floorplan` for JSON, a `FineMask` for images. Everything downstream — grid
building, robot modelling, planning, coverage, replay — is shared.
"""
from __future__ import annotations

import functools
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .houseexpo import Floorplan, load_floorplan
from .imageplan import ImageParseParams, load_gray, parse_floorplan_image, save_gray
from .rasterize import FineMask, png_from_masks, thumbnail_from_fine

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_GRAY_PIXELS = 1600 * 1600


def _slug(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "upload").strip())[:40]
    return base.strip("-") or "upload"


@dataclass
class Upload:
    id: str
    name: str
    kind: str                       # "houseexpo" | "image"
    created: float
    meta: dict = field(default_factory=dict)
    parse_params: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    gray_size: list = field(default_factory=list)


def _locked(method):
    """Serialize access to the store with its re-entrant lock.

    The HTTP server is threaded (`ThreadingHTTPServer`), so every public
    read/write of `items` / `index.json` must hold `self._lock`.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class UploadStore:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        self.items: dict[str, Upload] = {}
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------------ index
    def _load(self) -> None:
        if not self.index_path.is_file():
            return
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except ValueError:
            bad = self.index_path.with_name("index.json.bad")
            try:
                bad.write_bytes(self.index_path.read_bytes())
            except OSError:
                pass
            print(f"[uploads] {self.index_path.name} 解析失败，已备份为 {bad.name}，"
                  f"本次从空索引启动")
            return
        except OSError as exc:
            print(f"[uploads] 无法读取 {self.index_path.name}：{exc}（本次从空索引启动）")
            return
        for item in raw.get("uploads", []):
            try:
                up = Upload(**item)
            except TypeError:
                continue
            if self._exists(up):
                self.items[up.id] = up

    def _save(self) -> None:
        """Persist the index atomically (write temp file + os.replace).

        Callers must hold `self._lock`; a half-written index.json used to be
        possible under concurrent uploads, which then silently wiped the
        upload history on the next start (`_load` swallowed the error).
        """
        payload = {"uploads": [asdict(u) for u in self.items.values()]}
        text = json.dumps(payload, ensure_ascii=False, indent=1)
        tmp = self.index_path.with_name(self.index_path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.index_path)

    def _exists(self, up: Upload) -> bool:
        suffix = ".json" if up.kind == "houseexpo" else ".gray"
        return (self.dir / f"{up.id}{suffix}").is_file()

    def _new_id(self, name: str) -> str:
        stamp = time.strftime("%H%M%S")
        base = f"up-{_slug(name)}-{stamp}"
        candidate, n = base, 1
        while candidate in self.items:
            n += 1
            candidate = f"{base}-{n}"
        return candidate

    # -------------------------------------------------------------- houseexpo
    @_locked
    def add_houseexpo(self, name: str, text: str) -> Upload:
        if len(text.encode("utf-8")) > MAX_JSON_BYTES:
            raise ValueError("JSON 文件过大（上限 4 MB）")
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"不是合法的 JSON：{exc}") from exc
        for key in ("id", "bbox", "verts"):
            if key not in data:
                raise ValueError(f"缺少 HouseExpo 必需字段 {key!r}")
        if len(data["verts"]) < 3:
            raise ValueError("轮廓顶点少于 3 个，无法构成户型")

        uid = self._new_id(name)
        data["id"] = uid                      # make the internal id unique
        path = self.dir / f"{uid}.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        plan = load_floorplan(path)
        up = Upload(id=uid, name=name or uid, kind="houseexpo", created=time.time(),
                    meta=plan.meta())
        self.items[uid] = up
        self._save()
        return up

    # ------------------------------------------------------------------ image
    @_locked
    def add_image(self, name: str, gray: bytes, width: int, height: int,
                  params: ImageParseParams) -> Upload:
        if width * height > MAX_GRAY_PIXELS:
            raise ValueError(f"图像过大（上限 {MAX_GRAY_PIXELS} 像素）")
        if len(gray) != width * height:
            raise ValueError("灰度数据长度与尺寸不符")
        uid = self._new_id(name)
        save_gray(self.dir / f"{uid}.gray", gray, width, height)
        up = Upload(id=uid, name=name or uid, kind="image", created=time.time(),
                    gray_size=[width, height])
        self.items[uid] = up
        self.reparse_image(uid, params)
        return self.items[uid]

    @_locked
    def reparse_image(self, uid: str, params: ImageParseParams) -> Upload:
        up = self.items.get(uid)
        if up is None or up.kind != "image":
            raise KeyError(f"unknown image upload: {uid}")
        gray, width, height = load_gray(self.dir / f"{uid}.gray")
        fine, diag = parse_floorplan_image(gray, width, height, params)
        up.parse_params = asdict(params)
        up.diagnostics = diag
        up.meta = dict(fine.meta)
        up.meta["id"] = uid
        up.meta["kind"] = "image"
        up.meta["name"] = up.name
        up.meta["tier"] = "custom"
        # cache a preview so the picker does not have to re-parse
        wall, pw, ph = _preview_masks(fine)
        (self.dir / f"{uid}.png").write_bytes(png_from_masks(wall, pw, ph))
        self._save()
        return up

    # ----------------------------------------------------------------- access
    @_locked
    def contains(self, uid: str) -> bool:
        return uid in self.items

    @_locked
    def get(self, uid: str) -> Upload:
        up = self.items.get(uid)
        if up is None:
            raise KeyError(f"unknown upload: {uid}")
        return up

    @_locked
    def source(self, uid: str):
        """Return the floorplan source (Floorplan or FineMask) for this upload."""
        up = self.get(uid)
        if up.kind == "houseexpo":
            return load_floorplan(self.dir / f"{uid}.json")
        params = ImageParseParams(**up.parse_params)
        gray, width, height = load_gray(self.dir / f"{uid}.gray")
        fine, diag = parse_floorplan_image(gray, width, height, params)
        fine.meta = dict(up.meta)
        up.diagnostics = diag
        return fine

    @_locked
    def thumbnail(self, uid: str, width: int = 168) -> bytes:
        up = self.get(uid)
        if up.kind == "image":
            cached = self.dir / f"{uid}.png"
            if cached.is_file():
                return cached.read_bytes()
            return thumbnail_from_fine(self.source(uid), target_w=width)
        from .rasterize import thumbnail_png
        return thumbnail_png(self.source(uid), target_w=width)

    @_locked
    def list_meta(self) -> list[dict]:
        out = []
        for up in sorted(self.items.values(), key=lambda u: u.created, reverse=True):
            meta = dict(up.meta)
            meta.setdefault("id", up.id)
            meta["kind"] = up.kind
            meta["name"] = up.name
            meta["custom"] = True
            meta.setdefault("tier", "custom")
            meta.setdefault("room_num", 0)
            meta.setdefault("categories", [])
            out.append(meta)
        return out

    @_locked
    def delete(self, uid: str) -> bool:
        up = self.items.pop(uid, None)
        if up is None:
            return False
        for suffix in (".json", ".gray", ".png"):
            p = self.dir / f"{uid}{suffix}"
            if p.is_file():
                p.unlink()
        self._save()
        return True


def _preview_masks(fine: FineMask):
    """Wall mask for the preview image (inside -> floor, otherwise wall)."""
    wall = bytearray(1 if not v else 0 for v in fine.inside)
    return wall, fine.width, fine.height
