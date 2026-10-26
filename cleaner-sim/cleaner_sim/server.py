"""Standard-library HTTP server for the cleaning-simulation demo.

No web framework is used on purpose: the whole project runs on the Python standard
library, so it starts on any machine with Python 3.10+ and nothing installed.

API
---
GET    /                      the single-page UI
GET    /static/<file>         UI assets
GET    /api/health            liveness + dataset location
GET    /api/floorplans        catalogue: built-in dataset + user uploads
GET    /api/thumb?id=&w=      PNG thumbnail of the parsed floorplan
GET    /api/run?id=&...       run the full pipeline and return the demo payload
GET    /api/uploads           list user uploads with their parse diagnostics
POST   /api/upload/houseexpo  register a HouseExpo-format floorplan JSON
POST   /api/upload/image      register a floorplan image (browser-decoded grayscale)
POST   /api/parse/preview     stateless image parse, for the live threshold preview
POST   /api/upload/reparse    re-parse a stored image with new settings
POST   /api/delete            remove an upload
"""
from __future__ import annotations

import json
import mimetypes
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .imageplan import ImageParseParams
from .pipeline import PlanningError, RunParams, SimulationService
from .uploads import MAX_GRAY_PIXELS

WEB_DIR = (Path(__file__).resolve().parent.parent / "web").resolve()
MAX_BODY = 16 * 1024 * 1024
MAX_PREVIEW_PIXELS = 1400 * 1400        # 预览本就要降采样：独立且更小的上限
SPACING_MODES = ("exact", "overlap")


class _PayloadTooLarge(ValueError):
    """Request body exceeds MAX_BODY (mapped to HTTP 413)."""


def _num(query: dict, name: str, default: float, lo: float, hi: float) -> float:
    try:
        raw = query.get(name, [None])[0]
        value = default if raw in (None, "") else float(raw)
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _flag(query: dict, name: str, default: bool = False) -> bool:
    raw = query.get(name, [None])[0]
    if raw is None:
        return default
    return str(raw).lower() in ("1", "true", "yes", "on")


class Handler(BaseHTTPRequestHandler):
    server_version = "cleaner-sim/1.1"
    protocol_version = "HTTP/1.1"
    service: SimulationService = None

    # ------------------------------------------------------------------ plumbing
    def _send(self, code: int, body: bytes, content_type: str, cache: str = "no-store"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def _body(self) -> dict:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None or raw_length.strip() == "":
            length = 0
        else:
            try:
                length = int(raw_length)
            except ValueError:
                # 不排空请求体又不关连接的话，残留字节会被当成下一个请求解析
                self.close_connection = True
                raise ValueError("Content-Length 不是合法整数")
            if length < 0:
                self.close_connection = True
                raise ValueError("Content-Length 不能为负")
        if length == 0:
            return {}
        if length > MAX_BODY:
            # 在读走请求体之前就拒绝：关闭连接，避免残留字节污染 keep-alive 管道
            self.close_connection = True
            raise _PayloadTooLarge(f"请求体过大（上限 {MAX_BODY // (1024 * 1024)} MB）")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"请求体不是合法 JSON：{exc}") from exc

    def log_message(self, fmt, *args):                      # keep the console readable
        if "/api/" in (self.path or ""):
            print(f"  [api] {fmt % args}")

    # --------------------------------------------------------------------- routes
    def do_GET(self):                                       # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(unquote(path[len("/static/"):]))
            if path == "/api/health":
                return self._json({
                    "ok": True,
                    "floorplans": len(self.service.floorplans()),
                    "data_dir": self.service.data_dir(),
                    "upload_dir": self.service.upload_dir(),
                })
            if path == "/api/floorplans":
                return self._json({"floorplans": self.service.floorplans(),
                                   "data_dir": self.service.data_dir()})
            if path == "/api/uploads":
                return self._json({"uploads": self.service.uploads.list_meta()})
            if path == "/api/thumb":
                fid = query.get("id", [""])[0]
                width = int(_num(query, "w", 168, 48, 512))
                return self._send(200, self.service.thumbnail(fid, width),
                                  "image/png", cache="public, max-age=3600")
            if path == "/api/run":
                return self._run(query)
            return self._json({"error": "not found", "path": path}, 404)
        except KeyError as exc:
            return self._json({"error": f"未知户型：{exc}"}, 404)
        except PlanningError as exc:
            return self._json({
                "error": "该户型在当前机器人尺寸下没有可行的清扫区域："
                         "请尝试调整「机器人直径」或重新解析户型图",
                "detail": str(exc),
            }, 422)
        except Exception as exc:                             # noqa: BLE001
            rid = uuid.uuid4().hex[:8]
            print(f"[500] rid={rid} GET {path}")
            traceback.print_exc()
            return self._json({"error": "服务器内部错误", "request_id": rid}, 500)

    def do_DELETE(self):                                    # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/upload/"):
            uid = unquote(parsed.path[len("/api/upload/"):])
            ok = self.service.delete_upload(uid)
            return self._json({"deleted": ok, "id": uid}, 200 if ok else 404)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):                                      # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype and ctype != "application/json":
                return self._json({"error": "仅接受 application/json 请求体"}, 415)
            body = self._body()
            if path == "/api/upload/houseexpo":
                name = str(body.get("name") or "floorplan.json")
                text = body.get("text") or ""
                if not text.strip():
                    return self._json({"error": "缺少 JSON 内容"}, 400)
                return self._json(self.service.add_houseexpo(name, text))
            if path == "/api/upload/image":
                return self._json(self._add_image(body))
            if path == "/api/parse/preview":
                gray, width, height = _gray_from_body(body, MAX_PREVIEW_PIXELS)
                return self._json(self.service.parse_preview(
                    gray, width, height, _image_params(body)))
            if path == "/api/upload/reparse":
                uid = str(body.get("id") or "")
                return self._json(self.service.reparse_image(uid, _image_params(body)))
            if path == "/api/delete":
                uid = str(body.get("id") or "")
                ok = self.service.delete_upload(uid)
                return self._json({"deleted": ok, "id": uid}, 200 if ok else 404)
            return self._json({"error": "not found", "path": path}, 404)
        except _PayloadTooLarge as exc:
            return self._json({"error": str(exc)}, 413)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        except KeyError as exc:
            return self._json({"error": f"未知对象：{exc}"}, 404)
        except Exception as exc:                             # noqa: BLE001
            rid = uuid.uuid4().hex[:8]
            print(f"[500] rid={rid} POST {path}")
            traceback.print_exc()
            return self._json({"error": "服务器内部错误", "request_id": rid}, 500)

    # --------------------------------------------------------------------- helpers
    def _add_image(self, body: dict) -> dict:
        gray, width, height = _gray_from_body(body)
        return self.service.add_image(str(body.get("name") or "floorplan"),
                                      gray, width, height, _image_params(body))

    def _run(self, query: dict):
        fid = query.get("id", [""])[0]
        if not fid:
            return self._json({"error": "missing ?id=<floorplan id>"}, 400)
        spacing = query.get("spacing_mode", ["exact"])[0]
        if spacing not in SPACING_MODES:
            spacing = "exact"
        params = RunParams(
            cell_cm=_num(query, "cell_cm", 10.0, 4.0, 50.0),
            robot_cm=_num(query, "robot_cm", 34.0, 10.0, 120.0),
            target_cov=_num(query, "target_cov", 0.995, 0.80, 1.0),
            max_retreat=int(_num(query, "max_retreat", 4, 0, 10)),
            speed_mps=_num(query, "speed_mps", 0.25, 0.05, 1.5),
            turn_time_s=_num(query, "turn_time_s", 0.6, 0.0, 5.0),
            spacing_mode=spacing,
        )
        return self._json(self.service.run(fid, params))

    def _static(self, relative: str):
        target = (WEB_DIR / relative).resolve()
        if not str(target).startswith(str(WEB_DIR)) or not target.is_file():
            return self._json({"error": "not found", "path": relative}, 404)
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)


def _gray_from_body(body: dict, max_pixels: int = MAX_GRAY_PIXELS):
    """Decode the browser-posted grayscale buffer."""
    import base64
    try:
        width = int(body["width"])
        height = int(body["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("缺少图像尺寸 width/height") from exc
    if width <= 0 or height <= 0:
        raise ValueError("图像尺寸必须为正")
    if width * height > max_pixels:
        # 在 base64 解码与任何全图遍历之前先拒绝：超大图像的纯 Python 解析
        # 会长时间占满一个线程（CPU 耗尽型 DoS）。
        raise ValueError(f"图像过大（上限 {max_pixels} 像素）")
    raw = body.get("gray_b64") or ""
    if not raw:
        raise ValueError("缺少灰度数据 gray_b64")
    try:
        gray = base64.b64decode(raw, validate=True)
    except Exception as exc:                                 # noqa: BLE001
        raise ValueError("灰度数据不是合法 base64") from exc
    if len(gray) != width * height:
        raise ValueError(f"灰度数据长度 {len(gray)} 与 {width}x{height} 不符")
    return gray, width, height


def _image_params(body: dict) -> ImageParseParams:
    def num(key, default, lo, hi):
        try:
            raw = body.get(key)
            value = default if raw in (None, "") else float(raw)
        except (TypeError, ValueError):
            value = default
        return max(lo, min(hi, value))

    thr = body.get("threshold")
    try:
        threshold = None if thr in (None, "", "auto") else max(0, min(255, int(float(thr))))
    except (TypeError, ValueError):
        threshold = None

    return ImageParseParams(
        threshold=threshold,
        invert=bool(body.get("invert")),
        seal_px=int(num("seal_px", 2, 0, 8)),
        min_island_px=int(num("min_island_px", 40, 0, 100000)),
        real_width_m=num("real_width_m", 10.0, 1.0, 200.0),
        real_height_m=(num("real_height_m", 0, 0, 200.0) or None),
        scale_axis="y" if str(body.get("scale_axis", "x")).lower() == "y" else "x",
    )


def _port_in_use(host: str, port: int) -> bool:
    """Is something already accepting connections on this port?

    注意：Windows 下 SO_REUSEADDR 允许同一端口被重复绑定，仅靠 bind 无法发现
    "已有实例在运行"，必须主动连接探测一次，否则会出现两个服务静默共存的局面。
    """
    import socket
    try:
        with socket.create_connection((host, port), timeout=0.35):
            return True
    except OSError:
        return False


def serve(host: str = "127.0.0.1", port: int = 8017, data_dir: str | None = None,
          upload_dir=None, service: SimulationService | None = None) -> int:
    service = service or SimulationService(data_dir, upload_dir=upload_dir)
    Handler.service = service
    if _port_in_use(host, port):
        print(f"端口 {port} 已被占用（可能已有一个 cleaner-sim 实例在运行）——"
              f"请先关闭它，或使用 --port 指定其它端口。")
        return 1
    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        print(f"无法在 {host}:{port} 启动服务：{exc}"
              f"（端口可能被占用，请用 --port 换一个）")
        return 1
    print(f"cleaner-sim  |  {len(service.floorplans())} floorplans from {service.data_dir()}")
    print(f"uploads      |  {service.upload_dir()}")
    print(f"open         |  http://{host}:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()
    return 0
